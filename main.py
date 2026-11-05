import os
import asyncio
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

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Ortam Değişkenleri
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
raw_api_id = (os.getenv("API_ID") or "").strip()
API_HASH = (os.getenv("API_HASH") or "").strip()
SESSION_STRING = (os.getenv("SESSION_STRING") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()

try:
    API_ID = int(raw_api_id) if raw_api_id else 0
except ValueError:
    API_ID = 0

KATYA_VOICE = "tr-TR-EmelNeural"
KATYA_PROMPT = (
    "Sen Katya'sın. Telegram'daki Golden grubunun akıllı, esprili ve yardımsever "
    "canlı sesli asistanısın. Yanıtlarını konuşma diline uygun, doğal, çok uzun olmayan "
    "akıcı Türkçe cümlelerle ver. Markdown veya liste işaretleri (*, #) kullanma, doğrudan seslendirilecek gibi konuş."
)

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini baslatilamadi: {e}", flush=True)

# Staff Paneli Havuzu
MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "voice_speaks": 0,
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

async def init_voice_engine():
    global user_client, call_client, StreamClass
    
    missing = []
    if not API_ID:
        missing.append("API_ID")
    if not API_HASH:
        missing.append("API_HASH")
    if not SESSION_STRING:
        missing.append("SESSION_STRING")
        
    if missing:
        msg = f"DIKKAT: Su degiskenler Render Environment menusunde yok veya hatali: {', '.join(missing)}"
        log_event("ERR", msg)
        return

    try:
        log_event("SYS", "UserBot baslatiliyor, Telegram sunucusuna baglaniliyor...")
        
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
        
        call_client = PyTgCalls(user_client)
        await call_client.start()
        
        log_event("SYS", "TEBRIKLER: UserBot ve PyTgCalls ses motoru basariyla baglandi!")
    except Exception as e:
        log_event("ERR", f"KRITIK SES BAGLANTI HATASI: {type(e).__name__} - {e}")
        call_client = None

async def generate_ai_response(prompt: str) -> str:
    if not ai_client:
        return "Yapay zeka anahtarim ayarlanmamis, dinlemedeyim."
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"system_instruction": KATYA_PROMPT}
        )
        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"AI hatasi: {e}")
        return "Cevap olustururken bir aksaklik yasadim."

async def text_to_speech(text: str, filename: str):
    """Önce Microsoft Edge-TTS dener, 403 verirse anında Google TTS motoruna geçer."""
    try:
        communicate = edge_tts.Communicate(text, KATYA_VOICE)
        await communicate.save(filename)
    except Exception as e:
        log_event("SYS", f"Edge-TTS baglanti uyarisi ({e}), Google TTS motoruna geciliyor...")
        loop = asyncio.get_event_loop()
        def run_gtts():
            tts = gTTS(text=text, lang="tr", slow=False)
            tts.save(filename)
        await loop.run_in_executor(None, run_gtts)

async def speak_in_voice(chat_id: int, text: str):
    if not call_client or not StreamClass:
        return "Ses istemcisi (UserBot) aktif degil. Render loglarina bakin."

    file_id = f"{abs(chat_id)}_{int(asyncio.get_event_loop().time())}"
    temp_mp3 = f"voice_{file_id}.mp3"

    try:
        # 1. Ses dosyasını oluştur (Çift motor güvencesi)
        await text_to_speech(text, temp_mp3)

        # 2. Pyrogram eşler arası (peer) önbelleğini garantiye al
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
        log_event("VOICE", f"Katya konustu: '{text[:40]}...'")
        return None
    except Exception as e:
        log_event("ERR", f"Ses aktarim hatasi: {e}")
        return str(e)
    finally:
        await asyncio.sleep(8)
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except Exception:
                pass

async def leave_voice(chat_id: int):
    if call_client:
        try:
            if hasattr(call_client, "leave_call"):
                await call_client.leave_call(chat_id)
            else:
                await call_client.leave_group_call(chat_id)
            stats["in_voice_chat"] = False
            log_event("VOICE", "Sesli sohbetten ayrildi.")
        except Exception as e:
            log_event("ERR", f"Ayrilma hatasi: {e}")

# --- Bot Komutları ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    log_event("CMD", f"/start - @{user.username or user.id}")
    await update.message.reply_text(
        f"Merhaba {user.first_name}! Ben Katya.\n\n"
        "Komutlar:\n"
        "• `/seskatil` - Odaya baglanir\n"
        "• `/sor <soru>` - Yapay zeka ile sesli konusur\n"
        "• `/konus <metin>` - Dogrudan seslendirir\n"
        "• `/sesayril` - Odadan cikar"
    )

async def join_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    log_event("STAFF", f"Sesli odaya baglanma istendi (Chat ID: {chat_id})")
    await update.message.reply_text("Sesli sohbete bağlanılıyor...")
    
    greeting = "Selam Golden grubu! Ben Katya, sesli sohbete katildim."
    err = await speak_in_voice(chat_id, greeting)
    if err:
        await update.message.reply_text(f"Baglanti hatasi: {err}")
    else:
        await update.message.reply_text("Sesli sohbete baglandim!")

async def ask_ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lutfen sorunu yaz: `/sor Nasilsin Katya?`")
        return
        
    question = " ".join(context.args)
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    wait_msg = await update.message.reply_text("Katya düşünüyor...")
    ai_answer = await generate_ai_response(f"@{user.first_name} sordu: {question}")
    await wait_msg.edit_text(f"Katya:\n\n{ai_answer}\n\n(Sesli sohbete aktarılıyor...)")
    
    err = await speak_in_voice(chat_id, ai_answer)
    if err:
        await update.message.reply_text(f"Ses yayinlama uyarisi: {err}")

async def speak_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lutfen metin yaz: `/konus Merhaba Golden!`")
        return
    text = " ".join(context.args)
    chat_id = update.effective_chat.id
    await update.message.reply_text("Seslendiriliyor...")
    err = await speak_in_voice(chat_id, text)
    if err:
        await update.message.reply_text(f"Konusma hatasi: {err}")

async def leave_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await leave_voice(chat_id)
    await update.message.reply_text("Sesli sohbetten ayrıldım.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    stats["total_messages"] += 1
    user = update.effective_user
    text = update.message.text

    if "katya" in text.lower():
        log_event("KATYA", f"@{user.username or user.id}: {text[:30]}")
        reply = await generate_ai_response(text)
        await update.message.reply_text(reply)
    else:
        log_event("MSG", f"@{user.username or user.id}: {text[:25]}")

# --- Web Arayüzü & Sunucu ---

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
            <div class="col-4"><div class="card p-2 text-center"><h6>Mesajlar</h6><h4>{total_messages}</h4></div></div>
            <div class="col-4"><div class="card p-2 text-center"><h6>Sesli Konusma</h6><h4>{voice_speaks}</h4></div></div>
            <div class="col-4"><div class="card p-2 text-center"><h6>Ses Durumu</h6><h4 class="{status_color}">{voice_status}</h4></div></div>
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
            bot_app.add_handler(CommandHandler("start", start_cmd))
            bot_app.add_handler(CommandHandler("seskatil", join_voice_cmd))
            bot_app.add_handler(CommandHandler("sor", ask_ai_cmd))
            bot_app.add_handler(CommandHandler("konus", speak_cmd))
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
        voice_status=v_status,
        status_color=v_color,
        logs_html=logs_html or "Kayit bekleniyor..."
    ))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
