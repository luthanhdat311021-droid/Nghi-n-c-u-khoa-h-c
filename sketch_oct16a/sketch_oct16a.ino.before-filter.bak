#include <WiFi.h>
#include <WebServer.h>
#include "DHT.h"

// --- THAY TÊN VÀ MẬT KHẨU WI-FI NHÀ BẠN VÀO ĐÂY ---
const char* ssid = "C204";      
const char* password = "Tdmu#2026";   

#define DHTPIN 4       // Chân DATA (S) của DHT11 nối vào D4 (GPIO4) trên ESP32
#define DHTTYPE DHT11  // Khai báo loại cảm biến DHT11 thực tế

DHT dht(DHTPIN, DHTTYPE);
WebServer server(80);

// Giá trị đo gần nhất, cập nhật mỗi 1 giây trong loop() (DHT11 đo tối đa 1 lần/giây)
float lastTemp = NAN, lastHum = NAN;

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
  html += "<script>setInterval(function(){location.reload();}, 1000);</script></head><body>";
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

void setup() {
  Serial.begin(115200);
  dht.begin();
  WiFi.mode(WIFI_AP_STA);
  bool apOK = WiFi.softAP("ESP32-DHT11", "Dht112026");
  Serial.print("AP_STARTED="); Serial.println(apOK);
  Serial.print("AP_IP="); Serial.println(WiFi.softAPIP());
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
  WiFi.begin(ssid, password);
}

void loop() {
  server.handleClient();
  static unsigned long lastLog = 0;
  if (millis() - lastLog >= 1000) {
    lastLog = millis();
    Serial.print("STA_STATUS="); Serial.print((int)WiFi.status());
    Serial.print(" STA_IP="); Serial.print(WiFi.localIP());
    Serial.print(" AP_CLIENTS="); Serial.print(WiFi.softAPgetStationNum());
    float t = dht.readTemperature(false, true);   // force: bỏ bộ đệm 2s của thư viện
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) Serial.println(" DHT_READ_ERROR");
    else {
      lastTemp = t; lastHum = h;
      Serial.print(" Temperature_C="); Serial.print(t, 1);
      Serial.print(" Humidity_percent="); Serial.println(h, 1);
    }
  }
}
