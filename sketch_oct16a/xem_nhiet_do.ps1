# Doc nhiet do/do am tu ESP32 (DHT11) qua cong USB va hien thi tai http://localhost:8080
# Chay: powershell -ExecutionPolicy Bypass -File xem_nhiet_do.ps1 [-Port COM4]
param([string]$Port = "COM4", [int]$HttpPort = 8080)

$sp = New-Object System.IO.Ports.SerialPort $Port, 115200
$sp.DtrEnable = $false; $sp.RtsEnable = $false   # khong reset ESP32 khi mo cong
$sp.ReadTimeout = 200
$sp.Open()

$http = New-Object System.Net.HttpListener
$http.Prefixes.Add("http://localhost:$HttpPort/")
$http.Start()
Write-Host "Mo trinh duyet: http://localhost:$HttpPort  (Ctrl+C de dung)"

$temp = $null; $hum = $null; $ip = ""; $updated = $null; $buf = ""
$ctxTask = $http.GetContextAsync()
try {
  while ($true) {
    $buf += $sp.ReadExisting()
    while ($buf.Contains("`n")) {
      $i = $buf.IndexOf("`n"); $line = $buf.Substring(0, $i).Trim(); $buf = $buf.Substring($i + 1)
      if ($line -match 'Temperature_C=([\d.]+) Humidity_percent=([\d.]+)') {
        $temp = $Matches[1]; $hum = $Matches[2]; $updated = Get-Date
        Write-Host ("{0:HH:mm:ss}  Nhiet do: {1} C   Do am: {2} %" -f $updated, $temp, $hum)
      }
      if ($line -match 'STA_IP=(\S+)') { $ip = $Matches[1] }
    }

    if ($ctxTask.Wait(100)) {
      $ctx = $ctxTask.Result
      if ($ctx.Request.Url.AbsolutePath -eq "/data") {
        $body = "{""temperature"":$(if ($temp) { $temp } else { 'null' }),""humidity"":$(if ($hum) { $hum } else { 'null' }),""esp_ip"":""$ip"",""updated"":""$(if ($updated) { $updated.ToString('HH:mm:ss') })""}"
        $ctx.Response.ContentType = "application/json"
      } else {
        $body = @"
<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nhiệt độ phòng</title><style>
body{font-family:Arial,sans-serif;text-align:center;background:#f0f2f5;margin:0;padding:40px 16px}
.card{background:#fff;padding:30px;border-radius:15px;display:inline-block;box-shadow:0 4px 8px rgba(0,0,0,.1);min-width:260px}
h1{color:#007bff;font-size:1.4em}.val{font-size:2.4em;font-weight:bold;color:#333}.sub{color:#888;font-size:.9em}
</style></head><body><div class="card"><h1>NHIỆT ĐỘ PHÒNG (ESP32 + DHT11)</h1>
<p>🌡️ Nhiệt độ</p><div class="val" id="t">--</div><p>💧 Độ ẩm</p><div class="val" id="h">--</div>
<p class="sub" id="u">Đang chờ dữ liệu...</p>
<canvas id="c" width="560" height="220" style="width:100%;max-width:560px;margin-top:10px"></canvas>
<p class="sub"><span style="color:#e4572e">━ Nhiệt độ (°C)</span> &nbsp; <span style="color:#2e86de">━ Độ ẩm (%)</span> · 5 phút gần nhất</p></div><script>
const pts=[];let last='';
async function f(){try{const d=await (await fetch('/data')).json();
if(d.temperature!==null){t.textContent=d.temperature+' °C';h.textContent=d.humidity+' %';
u.textContent='Cập nhật lúc '+d.updated+' · ESP32 IP '+d.esp_ip;
if(d.updated!==last){last=d.updated;pts.push([d.temperature,d.humidity]);if(pts.length>300)pts.shift();draw()}}}
catch(e){u.textContent='Mất kết nối tới máy tính'}}
function line(ctx,i,col,W,H,pad){const v=pts.map(p=>p[i]);let lo=Math.min(...v)-1,hi=Math.max(...v)+1;
ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
v.forEach((y,k)=>{const x=pad+k*(W-2*pad)/Math.max(1,v.length-1),yy=H-pad-(y-lo)/(hi-lo)*(H-2*pad);k?ctx.lineTo(x,yy):ctx.moveTo(x,yy)});
ctx.stroke();ctx.fillStyle=col;ctx.font='12px Arial';ctx.fillText(hi.toFixed(1),i?W-36:4,pad+4);ctx.fillText(lo.toFixed(1),i?W-36:4,H-pad)}
function draw(){const ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
ctx.strokeStyle='#eee';for(let g=0;g<5;g++){const y=20+g*(H-40)/4;ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(W-40,y);ctx.stroke()}
line(ctx,0,'#e4572e',W,H,20);line(ctx,1,'#2e86de',W,H,20)}
f();setInterval(f,1000)</script></body></html>
"@
        $ctx.Response.ContentType = "text/html; charset=utf-8"
      }
      $bytes = [Text.Encoding]::UTF8.GetBytes($body)
      $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
      $ctx.Response.Close()
      $ctxTask = $http.GetContextAsync()
    }
  }
} finally {
  $http.Stop(); $sp.Close()
}
