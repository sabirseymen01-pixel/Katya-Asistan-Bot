import os
import asyncio
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import edge_tts

from pyrogram import Client

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Ortam Değişkenleri
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH", "")
SESSION_STRING = os.getenv("SESSION_STRING", "")

KATYA_VOICE = "tr-TR-EmelNeural"

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

# --- Ses Motoru (İzole, Çökmeyi Önler) ---

user_client = None
call_client = None
StreamClass = None

async def init_voice_engine():
    global user_client, call_client, StreamClass
    if not (SESSION_STRING and API_ID and API_HASH):
        log_event("SYS", "UserBot değişkenleri eksik, ses devre dışı.")
        return

    try:
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
        call_client = PyTgCalls(user_client)
        await user_client.start()
        await call_client.start()
        log_event("SYS", "UserBot ve PyTgCalls aktif.")
    except Exception as e:
        log_event("ERR", f"Ses motoru yükleme uyarısı: {e}")
        call_client = None

async def speak_in_voice(chat_id: int, text: str):
    if not call_client or not StreamClass:
        return "Ses istemcisi aktif değil veya kütüphane uyumsuz."

    file_id = f"{abs(chat_id)}_{int(asyncio.get_event_loop().time())}"
    temp_mp3 = f"voice_{file_id}.mp3"

    try:
        communicate = edge_tts.Communicate(text, KATYA_VOICE)
        await communicate.save(temp_mp3)

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
        log_event("VOICE", f"Sesli odada konuşuldu: '{text[:35]}...'")
        return None
    except Exception as e:
        log_event("ERR", f"Ses yayın hatası: {e}")
        return str(e)
    finally:
        await asyncio.sleep(5)
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
            log_event("VOICE", "Sesli sohbetten ayrıldı.")
        except Exception as e:
            log_event("ERR", f"Ayrılma hatası: {e}")

# --- Telegram Bot Komutları ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    log_event("CMD", f"/start - @{user.username or user.id}")
    await update.message.reply_text(
        f"Merhaba {user.first_name}! Ben Katya Asistan.\n"
        "Golden grubunu yönetmek ve sesli sohbete katılmak için buradayım.\n\n"
        "Komutlar:\n"
        "• `/seskatil` - Sesli odaya bağlanır.\n"
        "• `/konus <mesaj>` - Canlı sesli sohbette konuşur.\n"
        "• `/sesayril` - Sesli sohbetten çıkar."
    )

async def join_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    log_event("STAFF", f"Sesli odaya bağlanma istendi (Chat ID: {chat_id})")
    await update.message.reply_text("Sesli sohbete bağlanılıyor...")
    err = await speak_in_voice(chat_id, "Golden grubu selamlar, Katya sesli sohbete katıldı!")
    if err:
        await update.message.reply_text(f"Bağlantı hatası: {err}")
    else:
        await update.message.reply_text("Sesli sohbete bağlandım!")

async def speak_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lütfen konuşturmak istediğin metni yaz: `/konus Merhaba Golden!`")
        return
    text = " ".join(context.args)
    chat_id = update.effective_chat.id
    await update.message.reply_text("Seslendiriliyor...")
    err = await speak_in_voice(chat_id, text)
    if err:
        await update.message.reply_text(f"Konuşma hatası: {err}")

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
        await update.message.reply_text("Golden grubu için dinlemedeyim!")
    else:
        log_event("MSG", f"@{user.username or user.id}: {text[:25]}")

# --- FastAPI & Staff Paneli ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Katya Staff Suite</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ background-color: #0b0f19; color: #f1f5f9; font-family: system-ui, sans-serif; }}
        .card {{ background-color: #161e2e; border: 1px solid #283548; border-radius: 10px; }}
        .log-box {{ background: #070a11; border-radius: 8px; padding: 12px; font-family: monospace; max-height: 430px; overflow-y: auto; font-size: 0.85rem; }}
        .badge-voice {{ background-color: #8b5cf6; }}
        .badge-staff {{ background-color: #ef4444; }}
        .badge-katya {{ background-color: #06b6d4; }}
        .badge-cmd {{ background-color: #3b82f6; }}
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary">
            <div>
                <h4 class="mb-0 fw-bold">Katya Asistan <span class="badge bg-danger">Staff Panel</span></h4>
                <small class="text-secondary">Golden Grubu Ses & Chat Yönetim Arayüzü</small>
            </div>
            <span class="badge bg-success px-3 py-2">Çevrimiçi</span>
        </div>

        <div class="row g-2 mb-3">
            <div class="col-md-4">
                <div class="card p-3">
                    <span class="text-secondary small">Toplam Mesaj</span>
                    <h3 class="fw-bold text-info mb-0">{total_messages}</h3>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-3">
                    <span class="text-secondary small">Canlı Ses Konuşmaları</span>
                    <h3 class="fw-bold text-warning mb-0">{voice_speaks}</h3>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-3">
                    <span class="text-secondary small">Ses Kanalı Durumu</span>
                    <h3 class="fw-bold mb-0 {status_color}">{voice_status}</h3>
                </div>
            </div>
        </div>

        <div class="card p-3">
            <h6 class="mb-2">Canlı Konsol ve Olay Kayıtları</h6>
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
    
    # 1. Ses motorunu arka planda başlat (ana sunucuyu asla engellemez)
    asyncio.create_task(init_voice_engine())

    # 2. Telegram Botunu Başlat
    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
            bot_app.add_handler(CommandHandler("start", start_cmd))
            bot_app.add_handler(CommandHandler("seskatil", join_voice_cmd))
            bot_app.add_handler(CommandHandler("konus", speak_cmd))
            bot_app.add_handler(CommandHandler("sesayril", leave_voice_cmd))
            bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))

            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling()
            log_event("SYS", "Katya Bot dinlemede.")
        except Exception as e:
            log_event("ERR", f"Bot hatası: {e}")

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
    badge_map = {
        "VOICE": "badge-voice",
        "STAFF": "badge-staff",
        "KATYA": "badge-katya",
        "CMD": "badge-cmd",
        "SYS": "bg-secondary",
        "MSG": "bg-dark",
    }
    logs_html_list = []
    for log in logs_queue:
        badge_cls = badge_map.get(log["level"], "bg-secondary")
        logs_html_list.append(
            f"<div class='mb-1'><span class='text-secondary'>[{log['time']}]</span> "
            f"<span class='badge {badge_cls}'>{log['level']}</span> {log['message']}</div>"
        )

    v_status = "Odaya Bağlı" if stats["in_voice_chat"] else "Bağlı Değil"
    v_color = "text-success" if stats["in_voice_chat"] else "text-secondary"

    rendered = HTML_TEMPLATE.format(
        total_messages=stats["total_messages"],
        voice_speaks=stats["voice_speaks"],
        voice_status=v_status,
        status_color=v_color,
        logs_html="".join(logs_html_list) if logs_html_list else "<span class='text-muted'>Veri bekleniyor...</span>",
    )
    return HTMLResponse(content=rendered)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
