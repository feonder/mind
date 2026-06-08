"""Mind canlı eğitim görselleştirici — out/train_status.json'u okuyup
code rain + progress + loss grafiği olarak gösterir. Dashboard'dan bağımsız.

    python -m mindllm.viz_server --port 8095   → http://localhost:8095
"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

STATUS_PATH = "out/train_status.json"

PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>Mind · Canlı Eğitim</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#7CFC9A}
  #rain{position:fixed;inset:0}
  #hud{position:fixed;top:0;left:0;right:0;padding:18px 22px;z-index:2;
    background:linear-gradient(#000d,#0000)}
  .row{display:flex;align-items:center;gap:16px}
  .big{font-size:26px;font-weight:700;color:#b8ffd2;min-width:90px}
  #bar{height:12px;background:#0a2a17;border-radius:7px;flex:1;overflow:hidden;
    box-shadow:inset 0 0 6px #000}
  #fill{height:100%;width:0;background:linear-gradient(90deg,#1db954,#7CFC9A);
    transition:width .6s ease;box-shadow:0 0 12px #1db954}
  #meta{color:#46b277;font-size:13px}
  #spark{margin-top:10px}
  #sample{position:fixed;bottom:0;left:0;right:0;padding:14px 22px;z-index:2;
    color:#caffdd;font-size:14px;background:linear-gradient(#0000,#000d);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .off{color:#c8693f}
</style></head><body>
<canvas id="rain"></canvas>
<div id="hud">
  <div class="row">
    <span class="big" id="pct">—</span>
    <div id="bar"><div id="fill"></div></div>
  </div>
  <div id="meta">Mind eğitim bekleniyor…</div>
  <canvas id="spark" width="360" height="40"></canvas>
</div>
<div id="sample"></div>
<script>
const rain=document.getElementById('rain'),rx=rain.getContext('2d');
let W,H,cols,drops,glyphs="MIND.0123456789 abcdefghijklmnopqrstuvwxyz";
function resize(){W=rain.width=innerWidth;H=rain.height=innerHeight;
  cols=Math.floor(W/14);drops=Array(cols).fill(0).map(()=>Math.random()*-H);}
addEventListener('resize',resize);resize();
function draw(){
  rx.fillStyle='rgba(0,0,0,0.07)';rx.fillRect(0,0,W,H);
  rx.font='14px monospace';
  for(let i=0;i<cols;i++){
    const ch=glyphs.charAt(Math.floor(Math.random()*glyphs.length));
    rx.fillStyle=Math.random()>0.96?'#d6ffe6':'#1db954';
    rx.fillText(ch,i*14,drops[i]);
    if(drops[i]>H&&Math.random()>0.975)drops[i]=0;
    drops[i]+=14;
  }
}
setInterval(draw,55);
const sp=document.getElementById('spark'),sc=sp.getContext('2d');
function spark(h){
  sc.clearRect(0,0,360,40);if(!h||h.length<2)return;
  const mn=Math.min(...h),mx=Math.max(...h),r=(mx-mn)||1;
  sc.strokeStyle='#7CFC9A';sc.lineWidth=2;sc.beginPath();
  h.forEach((v,i)=>{const x=i/(h.length-1)*360,y=38-((v-mn)/r)*32;
    i?sc.lineTo(x,y):sc.moveTo(x,y);});
  sc.stroke();
}
async function poll(){
  try{
    const s=await (await fetch('/status?'+Date.now())).json();
    if(s.percent==null){return;}
    document.getElementById('pct').textContent='%'+s.percent;
    document.getElementById('fill').style.width=s.percent+'%';
    document.getElementById('meta').innerHTML=
      `${s.model} · iter ${s.iter}/${s.max_iters} · loss <b>${s.loss}</b> · ⚡${(s.tok_per_sec/1000).toFixed(1)}k tok/s `
      +(s.running?'':'<span class="off">· BİTTİ</span>');
    if(s.sample){glyphs=(s.sample.replace(/\\s+/g,'')||glyphs).slice(0,240);
      document.getElementById('sample').textContent='🌧️ '+s.sample;}
    spark(s.loss_history);
  }catch(e){}
}
poll();setInterval(poll,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/status"):
            try:
                body = open(STATUS_PATH, "rb").read()
            except OSError:
                body = b"{}"
            ctype = "application/json"
        else:
            body = PAGE.encode("utf-8")
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--status", default=STATUS_PATH)
    a = p.parse_args()
    STATUS_PATH = a.status
    print(f"Mind canlı görselleştirici: http://localhost:{a.port}")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
