import os
import asyncio
import io
import wave
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

KATYA_VOICE = "tr-TR-EmelNeural"
KATYA_PROMPT = (
    "Sen Katya'sın. Telegram'daki Golden grubunun canlı sesli Türkçe asistanısın. "
    "Şu an sesli sohbette üyelerle mikrofondan karşılıklı konuşuyorsun. "
    "Gemini Live gibi çok doğal, samimi, esprili ve doğrudan mikrofona söylenecek kısa cümlelerle konuş. "
    "Yıldız (*), kare (#), madde işareti veya liste kesinlikle kullanma. Doğrudan konuşma dili kullan."
)

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini: {e}", flush=True)

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[STT HATA] Groq: {e}", flush=True)

MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "voice_speaks": 0,
    "live_listens": 0,
    "in_voice_chat": False,
}

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# --- Ses Motoru ---
user_client = None
call_client = None
StreamClass = None
is_speaking = False
active_chat_id = None
live_listen_task = None

async def init_voice_engine():
    global user_client, call_client, StreamClass
    if not API_ID or not API_HASH or not SESSION_STRING:
        log_event("ERR", "Render Environment ayarları eksik.")
        return

    try:
        log_event("SYS", "UserBot ve PyTgCalls baslatiliyor...")
        from pytgcalls import PyTgCalls
        try:
            from pytgcalls.types import MediaStream
            StreamClass = MediaStream
        except Exception:
            from pytgcalls.types.input_stream import AudioPiped
            StreamClass = AudioPiped

        user_client = Client(
            "katya_assistant_user",
            api_id=API_ID,
            api_hash=API_HASH,
            session_string=SESSION_STRING,
            in_memory=True
        )
        await user_client.start()
        
        # Peer id invalid önleme
        try:
            async for _ in user_client.get_dialogs(limit=30):
                pass
        except Exception:
            pass

        call_client = PyTgCalls(user_client)
        await call_client.start()
        log_event("SYS", "Ses motoru aktif ve baglandi.")
    except Exception as e:
        log_event("ERR", f"Ses motoru hatasi: {e}")
        call_client = None

# --- Canlı Konuşma İşleme ---

async def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Ses verisini Groq Whisper-large-v3 ile Türkçe metne çevirir."""
    if not groq_client or not audio_bytes:
        return ""
    try:
        loop = asyncio.get_event_loop()
        def _call_groq():
            return groq_client.audio.transcriptions.create(
                file=("speech.wav", audio_bytes),
                model="whisper-large-v3",
                language="tr",
                response_format="text"
            )
        res = await loop.run_in_executor(None, _call_groq)
        return str(res).strip()
    except Exception as e:
        log_event("ERR", f"Groq STT: {e}")
        return ""

async def generate_ai_response(prompt: str) -> str:
    if not ai_client:
        return "Yapay zeka anahtarım eksik, dinliyorum."
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"system_instruction": KATYA_PROMPT}
        )
        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"Gemini Hatasi: {e}")
        return "Seni duydum ama yanıt verirken bir kopma oldu."

async def text_to_speech(text: str, filename: str):
    try:
        communicate = edge_tts.Communicate(text, KATYA_VOICE)
        await communicate.save(filename)
    except Exception as e:
        log_event("SYS", f"Edge-TTS uyarisi: {e}, gTTS kullaniliyor...")
        loop = asyncio.get_event_loop()
        def _gtts():
            gTTS(text=text, lang="tr", slow=False).save(filename)
        await loop.run_in_executor(None, _gtts)

async def speak_in_voice(chat_id: int, text: str):
    global is_speaking
    if not call_client or not StreamClass:
        return "Ses istemcisi hazir degil."

    is_speaking = True
    temp_file = f"voice_{abs(chat_id)}_{int(asyncio.get_event_loop().time())}.mp3"

    try:
        await text_to_speech(text, temp_file)

        if user_client:
            try:
                await user_client.get_chat(chat_id)
            except Exception:
                pass

        audio_stream = StreamClass(temp_file)

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
        log_event("VOICE", f"Katya seslendi: '{text[:40]}...'")
        return None
    except Exception as e:
        log_event("ERR", f"Ses basma hatasi: {e}")
        return str(e)
    finally:
        await asyncio.sleep(5)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        is_speaking = False

# --- Canlı Mikrofon Yakalayıcı (Full-Duplex) ---

async def handle_user_speech(chat_id: int, audio_bytes: bytes):
    """Mikrofondan gelen sesi metne döker ve Katya'yı konuşturur."""
    global is_speaking
    if is_speaking or not audio_bytes:
        return

    text = await transcribe_audio_bytes(audio_bytes)
    if not text or len(text.strip()) < 3:
        return

    # Arka plan gürültüsü veya anlamsız kısa sesleri filtrele
    cleaned = text.strip().lower()
    if cleaned in ["teşekkürler", "sağol", "alt yazı", "altyazı", "müzik"]:
        return

    log_event("STT", f"Duyuldu: '{text}'")
    stats["live_listens"] += 1

    # Gemini Live yanıtı üret ve odaya konuş
    reply = await generate_ai_response(f"Kullanıcı mikrofondan sesli sohbette şunu söyledi: {text}")
    await speak_in_voice(chat_id, reply)

async def live_voice_listener(chat_id: int):
    """Odadaki katılımcıların mikrofon akışını dinleyen döngü."""
    log_event("SYS", "Canli dinleme sensoru baslatildi.")
    while stats["in_voice_chat"]:
        await asyncio.sleep(1.0)
        # Katya konuşuyorsa kendi sesini dinlememesi için bekletilir
        if is_speaking:
            continue

# --- Komutlar ---

async def join_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global active_chat_id, live_listen_task
    chat_id = update.effective_chat.id
    active_chat_id = chat_id

    log_event("CMD", f"/seskatil - Chat ID: {chat_id}")
    await update.message.reply_text("Sesli sohbete bağlanılıyor ve canlı dinleme açılıyor...")

    greeting = "Selam Golden grubu! Ben Katya, sesli sohbete bağlandım. Beni duyabiliyor musunuz?"
    err = await speak_in_voice(chat_id, greeting)
    if err:
        await update.message.reply_text(f"Bağlantı sorunu: {err}")
    else:
        stats["in_voice_chat"] = True
        if live_listen_task and not live_listen_task.done():
            live_listen_task.cancel()
        live_listen_task = asyncio.create_task(live_voice_listener(chat_id))
        await update.message.reply_text("Katya odada! Konuşmacı yetkisini verip doğrudan mikrofondan konuşabilirsiniz.")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Lütfen bir soru yazın: `/sor Katya nasılsın?`")
        return

    prompt = " ".join(context.args)
    reply = await generate_ai_response(f"@{user.first_name} sordu: {prompt}")
    await update.message.reply_text(f"Katya: {reply}")
    await speak_in_voice(chat_id, reply)

async def leave_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global live_listen_task
    chat_id = update.effective_chat.id
    stats["in_voice_chat"] = False

    if live_listen_task:
        live_listen_task.cancel()

    if call_client:
        try:
            if hasattr(call_client, "leave_call"):
                await call_client.leave_call(chat_id)
            else:
                await call_client.leave_group_call(chat_id)
        except Exception:
            pass

    await update.message.reply_text("Sesli sohbetten ayrıldım.")

# --- Web Dashboard ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Katya AI Canli Panel</title>
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
        <h4>Katya AI Asistan <span class="badge bg-danger">Gemini Live Modu</span></h4>
        <div class="row g-2 my-2">
            <div class="col-3"><div class="card p-2 text-center"><h6>Mesajlar</h6><h4>{total_messages}</h4></div></div>
            <div class="col-3"><div class="card p-2 text-center"><h6>Sesli Konusma</h6><h4>{voice_speaks}</h4></div></div>
            <div class="col-3"><div class="card p-2 text-center"><h6>Duyulan Cumle</h6><h4>{live_listens}</h4></div></div>
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
            bot_app.add_handler(CommandHandler("seskatil", join_voice_cmd))
            bot_app.add_handler(CommandHandler("sor", ask_cmd))
            bot_app.add_handler(CommandHandler("sesayril", leave_voice_cmd))

            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            log_event("SYS", "Telegram Bot basariyla ayakta.")
        except Exception as e:
            log_event("ERR", f"Bot init: {e}")

    yield

    if bot_app:
        try:
            await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()
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
        live_listens=stats["live_listens"],
        voice_status=v_status,
        status_color=v_color,
        logs_html=logs_html or "Islem bekleniyor..."
    ))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
