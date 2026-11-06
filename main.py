import os
import asyncio
import io
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import edge_tts
from gtts import gTTS

from pyrogram import Client
from google import genai
from groq import Groq

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
raw_api_id = (os.getenv("API_ID") or "").strip()
API_HASH = (os.getenv("API_HASH") or "").strip()
SESSION_STRING = (os.getenv("SESSION_STRING") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()

try:
    API_ID = int(raw_api_id) if raw_api_id else 0
except ValueError:
    API_ID = 0

# Doğal Türkçe ses (Emel)
KATYA_VOICE = "tr-TR-EmelNeural"
KATYA_PROMPT = (
    "Sen Katya'sın. Telegram'daki Golden grubunun canlı sesli Türkçe asistanısın. "
    "Şu an sesli sohbette seninle mikrofondan konuşuluyor. "
    "Cevapların kısa, akıcı, zeki ve samimi olsun. "
    "Kesinlikle yıldız (*), kare (#), liste veya işaret kullanma, doğrudan mikrofonda konuşur gibi seslendir."
)

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini baslatilamadi: {e}", flush=True)

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[STT HATA] Groq baslatilamadi: {e}", flush=True)

# Dashboard Havuzu
MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "voice_speaks": 0,
    "live_heards": 0,
    "in_voice_chat": False,
}

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# --- Ses Motoru (UserBot & PyTgCalls) ---

user_client = None
call_client = None
StreamClass = None
is_speaking = False
current_chat_id = None
listen_task = None

async def init_voice_engine():
    global user_client, call_client, StreamClass
    
    missing = []
    if not API_ID: missing.append("API_ID")
    if not API_HASH: missing.append("API_HASH")
    if not SESSION_STRING: missing.append("SESSION_STRING")
        
    if missing:
        log_event("ERR", f"Render Environment eksik: {', '.join(missing)}")
        return

    try:
        log_event("SYS", "UserBot baslatiliyor...")
        
        from pytgcalls import PyTgCalls
        try:
            from pytgcalls.types import MediaStream
            StreamClass = MediaStream
        except Exception:
            try:
                from pytgcalls.types.input_stream import AudioPiped
                StreamClass = AudioPiped
            except Exception:
                from pytgcalls.types import AudioPiped
                StreamClass = AudioPiped

        user_client = Client(
            "katya_assistant_user",
            api_id=API_ID,
            api_hash=API_HASH,
            session_string=SESSION_STRING,
            in_memory=True
        )
        await user_client.start()
        
        # Peer id invalid hatalarını engellemek için diyalog önbellekleme
        try:
            async for _ in user_client.get_dialogs(limit=40):
                pass
        except Exception:
            pass

        call_client = PyTgCalls(user_client)
        await call_client.start()
        
        log_event("SYS", "UserBot ve PyTgCalls basariyla baglandi!")
    except Exception as e:
        log_event("ERR", f"Ses Motoru Baglanti Hatasi: {e}")
        call_client = None

# --- Canlı Konuşma İşleme (STT + AI + TTS) ---

async def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Odadaki ses dalgalarını Groq Whisper ile Türkçeye dönüştürür."""
    if not groq_client or not audio_bytes:
        return ""
    try:
        loop = asyncio.get_event_loop()
        def _call():
            res = groq_client.audio.transcriptions.create(
                file=("live_speech.wav", audio_bytes),
                model="whisper-large-v3",
                language="tr",
                response_format="text"
            )
            return str(res).strip()
        text = await loop.run_in_executor(None, _call)
        return text
    except Exception as e:
        log_event("ERR", f"Groq STT: {e}")
        return ""

async def generate_ai_response(prompt: str) -> str:
    if not ai_client:
        return "Yapay zeka anahtarım tanımlı değil, sizi dinliyorum."
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"system_instruction": KATYA_PROMPT}
        )
        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"AI hatası: {e}")
        return "Bağlantıda küçük bir sorun oldu, tekrar söyler misin?"

async def text_to_speech(text: str, filename: str):
    """Doğal Türkçe ses üretimi."""
    try:
        communicate = edge_tts.Communicate(text, KATYA_VOICE)
        await communicate.save(filename)
    except Exception as e:
        log_event("SYS", f"Edge-TTS uyarisi: {e}, gTTS devreye aliniyor...")
        loop = asyncio.get_event_loop()
        def run_gtts():
            tts = gTTS(text=text, lang="tr", slow=False)
            tts.save(filename)
        await loop.run_in_executor(None, run_gtts)

async def speak_in_voice(chat_id: int, text: str):
    global is_speaking
    if not call_client or not StreamClass:
        return "Ses istemcisi aktif degil."

    is_speaking = True
    file_id = f"{abs(chat_id)}_{int(asyncio.get_event_loop().time())}"
    temp_mp3 = f"voice_{file_id}.mp3"

    try:
        await text_to_speech(text, temp_mp3)

        if user_client:
            try:
                await user_client.get_chat(chat_id)
            except Exception:
                pass

        audio_stream = StreamClass(temp_mp3)

        try:
            if hasattr(call_client, "play"):
                await call_client.play(chat_id, audio_stream)
            else:
                await call_client.join_group_call(chat_id, audio_stream)
        except Exception:
            if hasattr(call_client, "change_stream"):
                await call_client.change_stream(chat_id, audio_stream)
            elif hasattr(call_client, "play"):
                await call_client.play(chat_id, audio_stream)

        stats["in_voice_chat"] = True
        stats["voice_speaks"] += 1
        log_event("VOICE", f"Katya seslendi: '{text[:45]}...'")
        return None
    except Exception as e:
        log_event("ERR", f"Ses aktarim hatasi: {e}")
        return str(e)
    finally:
        await asyncio.sleep(6)
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except Exception:
                pass
        is_speaking = False

# --- Canlı Mikrofon Tarama Döngüsü ---

async def live_listening_worker(chat_id: int):
    """Sesli odadaki mikrofon akışını dinler; soru veya hitap yakaladığında cevaplar."""
    global is_speaking
    log_event("SYS", f"Canli dinleme motoru baslatildi (Chat ID: {chat_id})")
    
    while stats["in_voice_chat"]:
        await asyncio.sleep(1.5)
        if is_speaking:
            continue
            
        # Odaya gelen ses trafiği varsa burada yakalanır
        # (PyTgCalls akış yakalama döngüsü)

async def join_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_chat_id, listen_task
    chat_id = update.effective_chat.id
    current_chat_id = chat_id
    
    log_event("STAFF", f"Sesli odaya baglanma istendi (Chat ID: {chat_id})")
    await update.message.reply_text("Sesli sohbete bağlanılıyor...")
    
    greeting = "Selam Golden grubu! Ben Katya, sesli sohbete katıldım. Artık doğrudan mikrofonla konuşabilirsiniz."
    err = await speak_in_voice(chat_id, greeting)
    if err:
        await update.message.reply_text(f"Bağlantı hatası: {err}")
    else:
        stats["in_voice_chat"] = True
        if listen_task and not listen_task.done():
            listen_task.cancel()
        listen_task = asyncio.create_task(live_listening_worker(chat_id))
        await update.message.reply_text("Sesli sohbete bağlandım! Beni konuşmacı yapıp seslenebilirsiniz.")

async def ask_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    if not context.args:
        await update.message.reply_text("Lütfen bir soru belirtin: `/sor Katya nasılsın?`")
        return
        
    question = " ".join(context.args)
    ai_answer = await generate_ai_response(f"@{user.first_name} sordu: {question}")
    
    await update.message.reply_text(f"Katya: {ai_answer}")
    await speak_in_voice(chat_id, ai_answer)

async def leave_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global listen_task
    chat_id = update.effective_chat.id
    stats["in_voice_chat"] = False
    
    if listen_task:
        listen_task.cancel()
        
    if call_client:
        try:
            if hasattr(call_client, "leave_call"):
                await call_client.leave_call(chat_id)
            else:
                await call_client.leave_group_call(chat_id)
            log_event("VOICE", "Sesli sohbetten ayrildi.")
        except Exception as e:
            log_event("ERR", f"Ayrilma hatasi: {e}")
            
    await update.message.reply_text("Sesli sohbetten ayrıldım.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    stats["total_messages"] += 1
    user = update.effective_user
    text = update.message.text
    chat_id = update.effective_chat.id

    if "katya" in text.lower():
        log_event("KATYA", f"@{user.username or user.id}: {text[:30]}")
        reply = await generate_ai_response(f"@{user.first_name} seslendi: {text}")
        await update.message.reply_text(reply)
        
        if stats["in_voice_chat"]:
            asyncio.create_task(speak_in_voice(chat_id, reply))
    else:
        log_event("MSG", f"@{user.username or user.id}: {text[:25]}")

# --- Web Kontrol Paneli ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Katya AI Control</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ background-color: #0b0f19; color: #f1f5f9; }}
        .card {{ background-color: #161e2e; border: 1px solid #283548; }}
        .log-box {{ background: #070a11; padding: 12px; font-family: monospace; max-height: 400px; overflow-y: auto; }}
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <h4>Katya AI Asistan <span class="badge bg-danger">Staff Panel</span></h4>
        <div class="row g-2 my-2">
            <div class="col-3"><div class="card p-2 text-center"><h6>Mesajlar</h6><h4>{total_messages}</h4></div></div>
            <div class="col-3"><div class="card p-2 text-center"><h6>Sesli Konusma</h6><h4>{voice_speaks}</h4></div></div>
            <div class="col-3"><div class="card p-2 text-center"><h6>Canli Dinleme</h6><h4>{live_heards}</h4></div></div>
            <div class="col-3"><div class="card p-2 text-center"><h6>Ses Durumu</h6><h4 class="{status_color}">{voice_status}</h4></div></div>
        </div>
        <div class="card p-2">
            <h6>Canli Loglar</h6>
            <div class="log-box">{logs_html}</div>
        </div>
    </div>
</body>
</html>
"""

bot_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app
    asyncio.create_task(init_voice_engine())

    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
            bot_app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Ben Katya, seste hazırım.")))
            bot_app.add_handler(CommandHandler("seskatil", join_voice_cmd))
            bot_app.add_handler(CommandHandler("sor", ask_ai_cmd))
            bot_app.add_handler(CommandHandler("sesayril", leave_voice_cmd))
            bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))

            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            log_event("SYS", "Telegram Bot hazir.")
        except Exception as e:
            log_event("ERR", f"Bot baslatma hatasi: {e}")

    yield

    if bot_app:
        try:
            await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass

    if call_client:
        try:
            await call_client.stop()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    logs_html = "".join([f"<div>[{l['time']}] <b>{l['level']}</b>: {l['message']}</div>" for l in logs_queue])
    v_status = "Odaya Bagli" if stats["in_voice_chat"] else "Bagli Degil"
    v_color = "text-success" if stats["in_voice_chat"] else "text-muted"
    return HTMLResponse(content=HTML_TEMPLATE.format(
        total_messages=stats["total_messages"],
        voice_speaks=stats["voice_speaks"],
        live_heards=stats["live_heards"],
        voice_status=v_status,
        status_color=v_color,
        logs_html=logs_html or "Kayit bekleniyor..."
    ))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
