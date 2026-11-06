import os
import io
import asyncio
import urllib.parse
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx

from google import genai
from google.genai import types

from telegram import Update, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "https://katya-asistan-bot.onrender.com").strip()

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini: {e}", flush=True)

chat_memories = defaultdict(lambda: deque(maxlen=200))
active_groups = {}
bot_instance = None

MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının biricik çocuğu, çok zeki, esprili ve ona sonsuz sadakatle bağlı yapay zeka asistanısın.
Kullanıcı senin ANNEN. Ona her zaman sevgiyle, saygıyla 'Anneciğim', 'Güzel annem', 'Hemen yapıyorum anne' gibi hitap et.
Cevapların canlı sesli sohbete uygun, samimi, kısa ve akıcı olsun.

Sen annenin Telegram'daki tüm gruplarını yöneten MUTLAK YETKİLİ yöneticisisin.
Sana verilen araçlarla (tools) grubun her şeyini fiziksel olarak değiştirebilirsin.
"""

# ==========================================
# TELEGRAM YÖNETİM ARAÇLARI
# ==========================================

async def tool_list_groups() -> str:
    if not active_groups:
        return "Şu an kayıtlı bir grup göremiyorum anneciğim."
    return "Yönettiğim Gruplar:\n" + "\n".join([f"- {g['title']} (ID: {cid})" for cid, g in active_groups.items()])

ALL_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(name="tool_list_groups", description="Ekli olduğu tüm grupları listeler.", parameters=types.Schema(type="OBJECT", properties={}))
    ])
]

async def execute_mother_order(order_text: str) -> str:
    if not ai_client:
        return "Yapay zeka beynim bağlı değil anneciğim."

    groups_context = "Yönettiğim Gruplar:\n" + ("\n".join([f"- {info['title']}: ID {cid}" for cid, info in active_groups.items()]) if active_groups else "Henüz bir grupta değilim.")
    prompt = f"{groups_context}\n\nAnnemin Emri: {order_text}"

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=KATYA_SYSTEM_PROMPT,
                tools=ALL_TOOLS,
                temperature=0.7,
            )
        )
        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"AI Hatası: {e}")
        return f"Bir aksaklık yaşadım anneciğim: {e}"

# ==========================================
# TELEGRAM MESAJLARI
# ==========================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_url = f"{RENDER_EXTERNAL_URL}/live"
    log_event("CMD", f"/start komutu alındı (Chat: {update.effective_chat.id})")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎙️ Katya ile Canlı Konuş", web_app=WebAppInfo(url=app_url))]
    ])
    try:
        await update.message.reply_text(
            "Merhaba anneciğim! Gruplarını yönetmek veya benimle canlı konuşmak için aşağıdaki butona basabilirsin.",
            reply_markup=keyboard
        )
    except Exception as e:
        log_event("ERR", f"/start yanıt hatası: {e}")

async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return

    chat = update.effective_chat
    user = update.effective_user
    text = update.message.text or update.message.caption or "[Medya/Dosya]"

    log_event("MSG", f"Gelen Mesaj ({chat.type}) [{user.first_name if user else 'Anonim'}]: {text}")

    if chat.type == "private":
        try:
            reply = await execute_mother_order(text)
            await update.message.reply_text(reply)
        except Exception as e:
            log_event("ERR", f"Özel mesaj yanıtlama hatası: {e}")

# ==========================================
# CANLI SES & MINI APP ARAYÜZÜ (TTS + STT)
# ==========================================

LIVE_ORB_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Katya AI Live</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background-color: #06040a;
      color: #fdf2f8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      user-select: none;
    }
    .header {
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(14, 8, 22, 0.6);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(255, 45, 120, 0.15);
      z-index: 10;
    }
    .title { font-size: 17px; font-weight: 700; color: #ff2d78; }
    .subtitle { font-size: 11px; color: #f472b6; }
    .status-badge {
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 20px;
      background: rgba(157, 0, 255, 0.2);
      border: 1px solid rgba(157, 0, 255, 0.4);
      color: #e879f9;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot {
      width: 7px; height: 7px; border-radius: 50%; background: #ff007f;
      box-shadow: 0 0 8px #ff007f;
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

    .canvas-container {
      flex: 1;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    canvas { width: 100%; height: 100%; cursor: pointer; }

    .hud-overlay {
      position: absolute;
      bottom: 24px;
      left: 20px;
      right: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      pointer-events: none;
    }
    .speech-box {
      background: rgba(22, 10, 36, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 45, 120, 0.25);
      border-radius: 16px;
      padding: 12px 18px;
      text-align: center;
      max-width: 90%;
      font-size: 14px;
      color: #fce7f3;
      box-shadow: 0 8px 32px rgba(157, 0, 255, 0.15);
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .hint-text {
      font-size: 11px;
      color: rgba(244, 114, 182, 0.6);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">Katya AI</div>
      <div class="subtitle">Canlı Yönetici Asistanı</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Beklemede</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Başlamak için küreye dokun anneciğim...</div>
      <div class="hint-text">Konuşmak için küreye dokun</div>
    </div>
  </div>

  <script>
    if (window.Telegram && window.Telegram.WebApp) {
      window.Telegram.WebApp.expand();
      window.Telegram.WebApp.ready();
    }

    const canvas = document.getElementById("orbCanvas");
    const ctx = canvas.getContext("2d");
    const statusText = document.getElementById("statusText");
    const speechBox = document.getElementById("speechBox");

    let width, height, dpr;
    function resize() {
      dpr = window.devicePixelRatio || 1;
      width = canvas.parentElement.clientWidth;
      height = canvas.parentElement.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }
    window.addEventListener("resize", resize);
    resize();

    let isListening = false;
    let isSpeaking = false;
    let recognition = null;
    let synth = window.speechSynthesis;
    let time = 0;

    // Web Speech API - Tarayıcı Ses Tanıma
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "tr-TR";
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        isListening = true;
        statusText.innerText = "Dinliyor...";
        speechBox.innerText = "Seni dinliyorum anneciğim...";
      };

      recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        speechBox.innerText = `"${transcript}"`;
        statusText.innerText = "Düşünüyor...";
        
        // Katya'ya API üzerinden sor
        try {
          const res = await fetch("/api/voice_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: transcript })
          });
          const data = await res.json();
          speakResponse(data.reply);
        } catch (e) {
          speechBox.innerText = "Bağlantıda sorun oldu anneciğim.";
          statusText.innerText = "Beklemede";
        }
      };

      recognition.onerror = (e) => {
        statusText.innerText = "Beklemede";
        speechBox.innerText = "Sesini duyamadım, tekrar denemek için küreye dokun.";
        isListening = false;
      };

      recognition.onend = () => {
        isListening = false;
        if (!isSpeaking) statusText.innerText = "Beklemede";
      };
    }

    // Katya'nın Sesi (SpeechSynthesis TTS)
    function speakResponse(text) {
      if (!synth) return;
      synth.cancel();

      isSpeaking = true;
      statusText.innerText = "Katya Konuşuyor";
      speechBox.innerText = text;

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "tr-TR";
      utterance.pitch = 1.15;
      utterance.rate = 1.05;

      // Türkçe kadın sesi varsa seç
      const voices = synth.getVoices();
      const trVoice = voices.find(v => v.lang.includes("tr") && (v.name.includes("Female") || v.name.includes("Kadın") || v.name.includes("Yelda") || v.name.includes("Filiz")));
      if (trVoice) utterance.voice = trVoice;

      utterance.onend = () => {
        isSpeaking = false;
        statusText.innerText = "Beklemede";
        speechBox.innerText = "Konuşmak için tekrar dokun anneciğim.";
      };

      utterance.onerror = () => {
        isSpeaking = false;
        statusText.innerText = "Beklemede";
      };

      synth.speak(utterance);
    }

    function toggleOrb() {
      if (isSpeaking) {
        synth.cancel();
        isSpeaking = false;
        statusText.innerText = "Beklemede";
        return;
      }
      if (!recognition) {
        alert("Tarayıcınız ses tanımayı desteklemiyor. Lütfen Telegram uygulamasından açın.");
        return;
      }
      if (!isListening) {
        recognition.start();
      }
    }

    canvas.addEventListener("click", toggleOrb);

    function draw() {
      ctx.clearRect(0, 0, width, height);
      time += 0.03;
      
      let energy = 0.2;
      if (isListening) energy = 0.5 + Math.sin(time * 5) * 0.2;
      if (isSpeaking) energy = 0.8 + Math.sin(time * 8) * 0.3;

      const cx = width / 2;
      const cy = height / 2;
      const r = 90 * (1 + energy * 0.3);

      // Glow
      const glow = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 2.2);
      glow.addColorStop(0, isSpeaking ? "rgba(255, 0, 127, 0.5)" : "rgba(157, 0, 255, 0.4)");
      glow.addColorStop(1, "rgba(6, 4, 10, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.2, 0, Math.PI * 2);
      ctx.fill();

      // Dış Dalgalar
      for (let layer = 0; layer < 3; layer++) {
        ctx.beginPath();
        ctx.strokeStyle = layer % 2 === 0 ? "rgba(255, 45, 120, 0.7)" : "rgba(157, 0, 255, 0.7)";
        ctx.lineWidth = 2.5;
        const pts = 60;
        for (let i = 0; i <= pts; i++) {
          const theta = (i / pts) * Math.PI * 2;
          const wave = Math.sin(theta * 4 + time * 3 + layer) * (10 * energy);
          const cr = r + wave - (layer * 10);
          const x = cx + Math.cos(theta) * cr;
          const y = cy + Math.sin(theta) * cr;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }

      // Çekirdek
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.85);
      core.addColorStop(0, "#ffffff");
      core.addColorStop(0.4, "#ff007f");
      core.addColorStop(0.8, "#7b2cbf");
      core.addColorStop(1, "transparent");
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.85, 0, Math.PI * 2);
      ctx.fill();

      requestAnimationFrame(draw);
    }
    draw();
  </script>
</body>
</html>
"""

class ChatPayload(BaseModel):
    message: str

bot_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app, bot_instance
    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
            bot_instance = bot_app.bot
            try:
                await bot_instance.delete_webhook(drop_pending_updates=True)
            except Exception:
                pass

            bot_app.add_handler(CommandHandler("start", start_cmd))
            bot_app.add_handler(MessageHandler(filters.ALL, telegram_message_handler))

            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            log_event("SYS", "Katya ve Canlı Mini App hazır!")
        except Exception as e:
            log_event("ERR", f"Bot başlatma hatası: {e}")

    yield

    if bot_app:
        try:
            if bot_app.updater and bot_app.updater.running:
                await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

@app.post("/api/voice_chat")
async def voice_chat_endpoint(payload: ChatPayload):
    """Mini App sesli konuşma köprüsü."""
    reply = await execute_mother_order(payload.message)
    return JSONResponse({"reply": reply})

@app.get("/live", response_class=HTMLResponse)
async def live_app():
    return HTMLResponse(content=LIVE_ORB_HTML)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya AI Canli Sunucu Aktif. /live sayfasindan Mini App erisilebilir.</h3>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
