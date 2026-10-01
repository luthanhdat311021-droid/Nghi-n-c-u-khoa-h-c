// EcoClass-Vision — ESP32 + PZEM-004T v3.0
// Đo điện áp/dòng/công suất của một mạch thiết bị và gửi lên MQTT mỗi 1 giây.
// Chỉ dùng để TỰ GÁN NHÃN dữ liệu và ĐỐI CHỨNG khi đánh giá, không dùng trong suy luận.
//
// Thư viện (Arduino Library Manager): PZEM004Tv30 (mandulaj), PubSubClient, ArduinoJson.
// Đấu nối: PZEM TX -> GPIO16 (RX2), PZEM RX -> GPIO17 (TX2), 5V, GND.
// AN TOÀN: phần đấu nối điện 220V phải do người có chuyên môn thực hiện, cắt điện khi lắp.

#include <WiFi.h>
#include <PubSubClient.h>
#include <PZEM004Tv30.h>
#include <ArduinoJson.h>
#include <time.h>

const char* WIFI_SSID   = "YOUR_WIFI";
const char* WIFI_PASS   = "YOUR_PASSWORD";
const char* MQTT_HOST   = "192.168.1.10";
const int   MQTT_PORT   = 1883;
const char* ROOM_ID     = "A1-101";
const char* DEVICE_ID   = "light_1";      // trùng id thiết bị trong configs/room_*.yaml

PZEM004Tv30 pzem(Serial2, 16, 17);
WiFiClient net;
PubSubClient mqtt(net);
char topic[96];

void connectWifi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  configTime(7 * 3600, 0, "pool.ntp.org", "time.google.com");   // đồng bộ thời gian để khớp với khung hình
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String cid = String("pzem-") + ROOM_ID + "-" + DEVICE_ID;
    if (!mqtt.connect(cid.c_str())) delay(2000);
  }
}

double nowEpoch() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec / 1e6;
}

void setup() {
  Serial.begin(115200);
  connectWifi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  snprintf(topic, sizeof(topic), "ecoclass/%s/power/%s", ROOM_ID, DEVICE_ID);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWifi();
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();

  float v = pzem.voltage(), i = pzem.current(), p = pzem.power(), e = pzem.energy();
  if (!isnan(v)) {
    StaticJsonDocument<192> doc;
    doc["ts"] = nowEpoch();
    doc["voltage"] = v;
    doc["current"] = i;
    doc["power"] = p;
    doc["energy"] = e;
    char buf[192];
    size_t n = serializeJson(doc, buf);
    mqtt.publish(topic, (const uint8_t*)buf, n, false);
  }
  delay(1000);
}
