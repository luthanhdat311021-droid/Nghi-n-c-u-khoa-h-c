#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include "DHT.h"

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

DHT dht(DHTPIN, DHTTYPE);
Preferences prefs;
float tempOffset = DEFAULT_TEMP_OFFSET;

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
