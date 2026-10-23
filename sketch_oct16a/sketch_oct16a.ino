#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include "DHT.h"
#include <IRremoteESP8266.h>
#include <IRac.h>
#include <IRrecv.h>
#include <IRsend.h>
#include <IRutils.h>

// --- CHẾ ĐỘ WI-FI ---
// 0 = tắt hẳn Wi-Fi/Bluetooth: chạy qua cáp USB với máy tính (EcoClass đọc qua cổng COM).
//     Tắt sóng vô tuyến → mạch bớt nóng, cảm biến đặt cạnh mạch đo bớt lệch.
// 1 = bật Wi-Fi: phát điểm truy cập "ESP32-DHT11" + trang web http://<IP>/ và /data.
#define WIFI_ENABLED 0

// --- THAY TÊN VÀ MẬT KHẨU WI-FI NHÀ BẠN VÀO ĐÂY (chỉ dùng khi WIFI_ENABLED = 1) ---
const char* ssid = "C204";
const char* password = "Tdmu#2026";

#define DHTPIN 4       // Chân DATA (S) của DHT11 nối vào D4 (GPIO4) trên ESP32
#define DHTTYPE DHT11  // Khai báo loại cảm biến DHT11 thực tế

// --- ĐO ---
const unsigned long READ_INTERVAL_MS = 2000;   // DHT11: tối thiểu 2 s giữa hai lần đo cho ổn định
const int MEDIAN_N = 5;                        // lọc trung vị 5 mẫu gần nhất (~10 s)
// Hiệu chỉnh lệch nhiệt độ (°C), cộng vào giá trị đo: đặt nhiệt kế chuẩn cạnh cảm biến,
// lệch = nhiệt kế - ESP32. Mặc định dưới đây; đổi khi đang chạy (không cần nạp lại) bằng cách
// gửi qua Serial Monitor (115200, kết thúc dòng "Newline"):  OFFSET=-1.5   → lưu vào bộ nhớ flash
//                                                          OFFSET?       → xem giá trị hiện tại
const float DEFAULT_TEMP_OFFSET = 0.0;

// --- ĐIỀU KHIỂN MÁY LẠNH BẰNG HỒNG NGOẠI ---
// Chân DAT/S của mạch phát nối D13 (GPIO13), GND nối GND (VCC nếu có thì nối VIN).
// Lệnh qua Serial Monitor (115200, "Newline"):
//   AC_SCAN        → lần lượt gửi lệnh TẮT của mọi hãng (mỗi hãng cách 4 s); máy lạnh tắt thì gõ AC_STOP
//   AC_SCAN=<số>   → quét tiếp từ thứ tự <số>
//   AC_STOP        → dừng quét
//   AC_TRY=<tên>   → gửi lệnh TẮT theo một giao thức (vd AC_TRY=DAIKIN), không lưu
//   AC_USE=<tên>   → lưu giao thức của máy lạnh vào bộ nhớ flash (vd AC_USE=PANASONIC_AC)
//   AC_MODEL=<số>  → lưu đời máy (-1 = mặc định); thử 1, 2, 3... nếu hãng đúng mà không ăn lệnh
//   AC_OFF         → phát lệnh TẮT của MỌI hãng, mỗi hãng đủ các đời remote (~55 s): giao thức đã lưu trước,
//                    rồi các hãng phổ biến ở VN, rồi phần còn lại → tắt được nhiều loại máy lạnh trong một lần
//   AC_OFF_ONE / AC_ON → tắt / bật (Làm lạnh 26 °C) chỉ theo giao thức đã lưu
//   AC_LIST        → in danh sách giao thức hỗ trợ;  AC? → xem giao thức đang lưu
//   AC_LEARN       → học lệnh TẮT từ remote thật trong 30 s (mạch thu: OUT → D14, VCC → 3V3, GND → GND).
//                    Lưu nguyên tín hiệu vào flash; AC_OFF / AC_OFF_ONE phát tín hiệu đã học trước tiên.
//   AC_FORGET      → xoá tín hiệu đã học
#define IR_LED_PIN 13
#define IR_RECV_PIN 14
const unsigned long LEARN_MS = 30000;
const uint16_t MAX_LEARN = 1024;
const unsigned long SCAN_STEP_MS = 4000;
const unsigned long OFF_ALL_STEP_MS = 300;     // khoảng nghỉ giữa hai hãng khi phát AC_OFF
// Thứ tự ưu tiên khi phát AC_OFF; các giao thức còn lại phát sau theo thứ tự của thư viện
const char* const OFF_PRIORITY[] = {
  "MITSUBISHI_AC", "MITSUBISHI136", "MITSUBISHI112", "MITSUBISHI_HEAVY_88", "MITSUBISHI_HEAVY_152",
  "DAIKIN", "DAIKIN2", "DAIKIN216", "DAIKIN160", "DAIKIN176", "DAIKIN128", "DAIKIN152", "DAIKIN64",
  "PANASONIC_AC", "PANASONIC_AC32", "LG", "LG2", "SAMSUNG_AC", "TOSHIBA_AC", "SHARP_AC",
  "GREE", "MIDEA", "COOLIX", "TCL112AC", "CARRIER_AC64", "HITACHI_AC", "FUJITSU_AC", "AIRWELL"};

DHT dht(DHTPIN, DHTTYPE);
Preferences prefs;
float tempOffset = DEFAULT_TEMP_OFFSET;

IRac ac(IR_LED_PIN);
IRsend irsend(IR_LED_PIN);
IRrecv irrecv(IR_RECV_PIN, MAX_LEARN, 50, true);   // bộ đệm lớn: lệnh máy lạnh dài, thường gửi 2 khung liền
decode_results learnRes;
uint16_t learnedRaw[MAX_LEARN];
uint16_t learnedLen = 0;                       // 0 = chưa học
bool learning = false;
unsigned long learnStart = 0;
decode_type_t acProto = decode_type_t::UNKNOWN;
int16_t acModel = -1;
bool scanning = false;
int scanPos = 0;                 // thứ tự hiện tại trong danh sách giao thức hỗ trợ
unsigned long scanLast = 0;
struct AcTarget { decode_type_t proto; int16_t model; };
AcTarget offOrder[200];                        // thứ tự phát của AC_OFF: (hãng, đời remote)
int offCount = 0, offPos = 0;
bool offAll = false;
unsigned long offLast = 0;

// Giao thức máy lạnh thứ i (bắt đầu từ 0) mà thư viện hỗ trợ; UNKNOWN nếu hết danh sách
decode_type_t supportedProto(int i) {
  for (int p = 1; p <= kLastDecodeType; p++) {
    if (IRac::isProtocolSupported((decode_type_t)p) && i-- == 0) return (decode_type_t)p;
  }
  return decode_type_t::UNKNOWN;
}

void sendAc(decode_type_t proto, int16_t model, bool power) {
  ac.next.protocol = proto;
  ac.next.model = model;
  ac.next.power = power;
  ac.next.mode = stdAc::opmode_t::kCool;
  ac.next.celsius = true;
  ac.next.degrees = 26;
  ac.next.fanspeed = stdAc::fanspeed_t::kAuto;
  ac.sendAc();
}

// Các đời remote (model) thư viện hỗ trợ của từng hãng (theo IRsend.h); hãng không có trong bảng: -1 = mặc định
void modelRange(decode_type_t p, int16_t &lo, int16_t &hi) {
  lo = hi = -1;
  switch (p) {
    case decode_type_t::PANASONIC_AC: lo = 1; hi = 6; break;   // LKE, NKE, DKE/PKR, JKE, CKP, RKR
    case decode_type_t::FUJITSU_AC:   lo = 1; hi = 6; break;   // ARRAH2E, ARDB1, ARREB1E, ARJW2, ARRY4, ARREW4E
    case decode_type_t::LG:
    case decode_type_t::LG2:          lo = 1; hi = 5; break;   // GE6711AR2853M, AKB75215403, AKB74955603, ...
    case decode_type_t::GREE:         lo = 1; hi = 3; break;   // YAW1F, YBOFB, YX1FSF
    case decode_type_t::SHARP_AC:     lo = 1; hi = 3; break;   // A907, A705, A903
    case decode_type_t::TOSHIBA_AC:   lo = 0; hi = 1; break;   // remote A, remote B (WA-TH03A...)
    case decode_type_t::TCL112AC:     lo = 1; hi = 2; break;
    case decode_type_t::WHIRLPOOL_AC: lo = 1; hi = 2; break;
    case decode_type_t::HITACHI_AC1:  lo = 1; hi = 2; break;
    case decode_type_t::HAIER_AC176:  lo = 1; hi = 2; break;
    case decode_type_t::MIRAGE:       lo = 1; hi = 2; break;
    case decode_type_t::ARGO:         lo = 1; hi = 2; break;
    case decode_type_t::VOLTAS:       lo = 0; hi = 1; break;
    default: break;
  }
}

void addOffTarget(decode_type_t p, int16_t m) {
  for (int i = 0; i < offCount; i++) if (offOrder[i].proto == p && offOrder[i].model == m) return;
  if (offCount < (int)(sizeof(offOrder) / sizeof(offOrder[0]))) offOrder[offCount++] = {p, m};
}

void addOff(decode_type_t p) {
  if (!IRac::isProtocolSupported(p)) return;
  int16_t lo, hi;
  modelRange(p, lo, hi);
  for (int16_t m = lo; m <= hi; m++) addOffTarget(p, m);
}

// Giao thức + đời đã lưu → danh sách ưu tiên (mọi đời) → mọi giao thức còn lại
void buildOffOrder() {
  offCount = 0;
  if (IRac::isProtocolSupported(acProto)) {
    if (acModel != -1) addOffTarget(acProto, acModel);
    addOff(acProto);
  }
  for (const char* name : OFF_PRIORITY) addOff(strToDecodeType(name));
  for (int i = 0; supportedProto(i) != decode_type_t::UNKNOWN; i++) addOff(supportedProto(i));
}

void printAcConfig() {
  Serial.print("AC_PROTO="); Serial.print(typeToString(acProto));
  Serial.print(" AC_MODEL="); Serial.print(acModel);
  Serial.print(" AC_LEARNED_LEN="); Serial.println(learnedLen);
}

void sendLearned() {
  if (learnedLen) irsend.sendRaw(learnedRaw, learnedLen, 38);
}

// Đang học: nhận được tín hiệu → lưu nguyên dạng thô (phát lại y hệt remote), giao thức nhận ra được thì lưu luôn
void handleLearning() {
  if (!learning) return;
  if (irrecv.decode(&learnRes)) {
    uint16_t len = getCorrectedRawLength(&learnRes);
    if (len < 20) {                            // nhiễu/tín hiệu quá ngắn: bỏ, chờ tiếp
      irrecv.resume();
      return;
    }
    uint16_t* raw = resultToRawArray(&learnRes);
    learnedLen = min(len, MAX_LEARN);
    memcpy(learnedRaw, raw, learnedLen * sizeof(uint16_t));
    delete[] raw;
    prefs.putBytes("irraw", learnedRaw, learnedLen * sizeof(uint16_t));
    prefs.putUShort("irrawlen", learnedLen);
    Serial.print("AC_LEARNED len="); Serial.print(learnedLen);
    Serial.print(" proto="); Serial.println(typeToString(learnRes.decode_type));
    String desc = IRAcUtils::resultAcToString(&learnRes);
    if (desc.length()) { Serial.print("AC_LEARNED_STATE "); Serial.println(desc); }
    if (IRac::isProtocolSupported(learnRes.decode_type)) {
      acProto = learnRes.decode_type;
      prefs.putInt("irproto", (int)acProto);
    }
    learning = false;
    irrecv.disableIRIn();
    printAcConfig();
  } else if (millis() - learnStart >= LEARN_MS) {
    learning = false;
    irrecv.disableIRIn();
    Serial.println("AC_LEARN_TIMEOUT - khong nhan duoc tin hieu (kiem tra day mach thu: OUT->D14, VCC->3V3)");
  }
}

float tBuf[MEDIAN_N], hBuf[MEDIAN_N];
int nSamples = 0, bufPos = 0;

// Giá trị đã lọc (đã cộng hiệu chỉnh), cập nhật mỗi 2 giây trong loop()
float lastTemp = NAN, lastHum = NAN;

float median(const float* src, int n) {
  float a[MEDIAN_N];
  for (int i = 0; i < n; i++) a[i] = src[i];
  for (int i = 1; i < n; i++) {                // sắp xếp chèn, n <= 5
    float v = a[i]; int j = i - 1;
    while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
    a[j + 1] = v;
  }
  return (n % 2) ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0f;
}

void handleSerialCommand() {
  static String cmd;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') { cmd += c; continue; }
    cmd.trim();
    if (cmd.startsWith("OFFSET=")) {
      tempOffset = cmd.substring(7).toFloat();
      prefs.putFloat("toffset", tempOffset);
      if (nSamples > 0) lastTemp = median(tBuf, nSamples) + tempOffset;
      Serial.print("OFFSET_SET="); Serial.println(tempOffset, 2);
    } else if (cmd == "OFFSET?") {
      Serial.print("OFFSET="); Serial.println(tempOffset, 2);
    } else if (cmd == "AC_SCAN" || cmd.startsWith("AC_SCAN=")) {
      scanPos = cmd.length() > 8 ? cmd.substring(8).toInt() : 0;
      scanning = true;
      scanLast = millis() - SCAN_STEP_MS;      // gửi hãng đầu tiên ngay
      Serial.println("AC_SCAN_START - may lanh tat thi go AC_STOP");
    } else if (cmd == "AC_STOP") {
      scanning = false;
      Serial.print("AC_SCAN_STOPPED last="); Serial.print(typeToString(supportedProto(scanPos - 1)));
      Serial.print(" prev="); Serial.println(typeToString(supportedProto(scanPos - 2)));
      Serial.println("Thu lai tung cai bang AC_TRY=<ten>, dung cai nao thi AC_USE=<ten>");
    } else if (cmd.startsWith("AC_TRY=") || cmd.startsWith("AC_USE=")) {
      String name = cmd.substring(7);
      name.toUpperCase();
      decode_type_t p = strToDecodeType(name.c_str());
      if (!IRac::isProtocolSupported(p)) {
        Serial.print("AC_UNKNOWN_PROTO="); Serial.println(name);
      } else if (cmd.startsWith("AC_TRY=")) {
        sendAc(p, acModel, false);
        Serial.print("AC_TRY_SENT="); Serial.println(typeToString(p));
      } else {
        acProto = p;
        prefs.putInt("irproto", (int)p);
        printAcConfig();
      }
    } else if (cmd.startsWith("AC_MODEL=")) {
      acModel = cmd.substring(9).toInt();
      prefs.putShort("irmodel", acModel);
      printAcConfig();
    } else if (cmd == "AC_OFF") {
      sendLearned();                           // tín hiệu học từ remote thật: phát đầu tiên
      buildOffOrder();
      scanning = false;
      offAll = true;
      offPos = 0;
      offLast = millis();
      Serial.print("AC_OFF_ALL_START n="); Serial.print(offCount);
      Serial.print(" learned="); Serial.println(learnedLen ? "yes" : "no");
    } else if (cmd == "AC_LEARN") {
      offAll = scanning = false;
      irrecv.enableIRIn();
      learning = true;
      learnStart = millis();
      Serial.println("AC_LEARN_START - chia remote vao mach thu va bam nut TAT (30 s)");
    } else if (cmd == "AC_DUMP") {             // in tín hiệu đã học (µs: bật, tắt, bật, ...) để phân tích
      Serial.print("AC_DUMP len="); Serial.print(learnedLen); Serial.print(" :");
      for (uint16_t i = 0; i < learnedLen; i++) { Serial.print(' '); Serial.print(learnedRaw[i]); }
      Serial.println();
    } else if (cmd == "AC_FORGET") {
      learnedLen = 0;
      prefs.remove("irraw");
      prefs.remove("irrawlen");
      printAcConfig();
    } else if (cmd == "AC_OFF_ONE" && learnedLen) {
      sendLearned();
      Serial.println("AC_OFF_SENT learned");
    } else if (cmd == "AC_OFF_ONE" || cmd == "AC_ON") {
      if (acProto == decode_type_t::UNKNOWN) {
        Serial.println("AC_PROTO_NOT_SET - chay AC_SCAN hoac AC_USE=<ten> truoc");
      } else {
        sendAc(acProto, acModel, cmd == "AC_ON");
        Serial.println(cmd == "AC_ON" ? "AC_ON_SENT" : "AC_OFF_SENT");
      }
    } else if (cmd == "AC_LIST") {
      for (int i = 0; supportedProto(i) != decode_type_t::UNKNOWN; i++) {
        Serial.print(i); Serial.print(": "); Serial.println(typeToString(supportedProto(i)));
      }
    } else if (cmd == "AC?") {
      printAcConfig();
    }
    cmd = "";
  }
}

#if WIFI_ENABLED
WebServer server(80);
bool staGaveUp = false;

void handleData() {
  String json = "{\"temperature\":" + (isnan(lastTemp) ? String("null") : String(lastTemp, 1)) +
                ",\"humidity\":" + (isnan(lastHum) ? String("null") : String(lastHum, 1)) + "}";
  server.send(200, "application/json", json);
}

void handleRoot() {
  float hum = lastHum;
  float temp = lastTemp;

  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<style>body{font-family:Arial; text-align:center; margin-top:50px; background:#f0f2f5;}";
  html += ".card{background:white; padding:30px; border-radius:15px; display:inline-block; box-shadow:0 4px 8px rgba(0,0,0,0.1);}";
  html += "h1{color:#007bff;} .val{font-size:1.8em; font-weight:bold; color:#333;}</style>";
  html += "<script>setInterval(function(){location.reload();}, 2000);</script></head><body>";
  html += "<div class='card'><h1>TRẠM ĐO THỜI TIẾT IOT</h1>";

  if (isnan(temp) || isnan(hum)) {
    html += "<p style='color:red;'>Lỗi: Không đọc được dữ liệu từ cảm biến DHT11!</p>";
  } else {
    html += "<p>🌡️ Nhiệt độ phòng: <span class='val'>" + String(temp, 1) + " °C</span></p>";
    html += "<p>💧 Độ ẩm không khí: <span class='val'>" + String(hum, 1) + " %</span></p>";
  }

  html += "</div></body></html>";
  server.send(200, "text/html", html);
}
#endif

void setup() {
  Serial.begin(115200);
  dht.begin();
  prefs.begin("ecoclass", false);
  tempOffset = prefs.getFloat("toffset", DEFAULT_TEMP_OFFSET);
  Serial.print("OFFSET="); Serial.println(tempOffset, 2);
  acProto = (decode_type_t)prefs.getInt("irproto", (int)decode_type_t::UNKNOWN);
  acModel = prefs.getShort("irmodel", -1);
  learnedLen = min(prefs.getUShort("irrawlen", 0), MAX_LEARN);
  if (learnedLen) prefs.getBytes("irraw", learnedRaw, learnedLen * sizeof(uint16_t));
  irsend.begin();
  printAcConfig();
#if WIFI_ENABLED
  WiFi.mode(WIFI_AP_STA);
  bool apOK = WiFi.softAP("ESP32-DHT11", "Dht112026");
  Serial.print("AP_STARTED="); Serial.println(apOK);
  Serial.print("AP_IP="); Serial.println(WiFi.softAPIP());
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
  WiFi.begin(ssid, password);
#else
  WiFi.mode(WIFI_OFF);
  btStop();
  Serial.println("WIFI=OFF");
#endif
}

void loop() {
  handleSerialCommand();
  handleLearning();
  if (offAll && millis() - offLast >= OFF_ALL_STEP_MS) {
    sendAc(offOrder[offPos].proto, offOrder[offPos].model, false);
    offLast = millis();                        // tính nghỉ từ lúc phát xong
    if (++offPos >= offCount) {
      offAll = false;
      Serial.print("AC_OFF_ALL_DONE n="); Serial.println(offCount);
    }
  }
  if (scanning && millis() - scanLast >= SCAN_STEP_MS) {
    scanLast = millis();
    decode_type_t p = supportedProto(scanPos);
    if (p == decode_type_t::UNKNOWN) {
      scanning = false;
      Serial.println("AC_SCAN_DONE - het danh sach");
    } else {
      sendAc(p, -1, false);
      Serial.print("AC_SCAN "); Serial.print(scanPos); Serial.print(": "); Serial.println(typeToString(p));
      scanPos++;
    }
  }
#if WIFI_ENABLED
  server.handleClient();
  // Không vào được Wi-Fi "C204" sau 30 s → thôi thử lại (thử liên tục làm mạch nóng), giữ điểm truy cập
  if (!staGaveUp && WiFi.status() != WL_CONNECTED && millis() > 30000) {
    WiFi.disconnect(false, false);
    WiFi.mode(WIFI_AP);
    staGaveUp = true;
  }
#endif
  static unsigned long lastRead = 0;
  if (millis() - lastRead < READ_INTERVAL_MS) return;
  lastRead = millis();

#if WIFI_ENABLED
  Serial.print("STA_STATUS="); Serial.print((int)WiFi.status());
  Serial.print(" STA_IP="); Serial.print(WiFi.localIP());
  Serial.print(" AP_CLIENTS="); Serial.print(WiFi.softAPgetStationNum());
  Serial.print(" ");
#endif
  dht.read(true);                              // đo mới (bỏ bộ đệm 2 s của thư viện)
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h) || h <= 0 || h > 100 || t < -10 || t > 60) {
    Serial.println("DHT_READ_ERROR");          // mẫu lỗi/vô lý: bỏ, không đưa vào bộ lọc
    return;
  }
  tBuf[bufPos] = t; hBuf[bufPos] = h;
  bufPos = (bufPos + 1) % MEDIAN_N;
  if (nSamples < MEDIAN_N) nSamples++;
  lastTemp = median(tBuf, nSamples) + tempOffset;
  lastHum = median(hBuf, nSamples);

  // Giữ đúng dạng "Temperature_C=.. Humidity_percent=.." mà EcoClass (ecoclass/edge/esp32.py) đang đọc
  Serial.print("Temperature_C="); Serial.print(lastTemp, 1);
  Serial.print(" Humidity_percent="); Serial.print(lastHum, 1);
  Serial.print(" Raw_C="); Serial.print(t, 1);
  Serial.print(" Offset_C="); Serial.print(tempOffset, 2);
  Serial.print(" Samples="); Serial.println(nSamples);
}
