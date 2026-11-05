import os
import asyncio
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Ses motorunu içeren harici modül
import voice

# Ortam Değişkenleri
BOT_TOKEN = os.getenv("BOT_TOKEN")

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
    
    err = await voice.join_and_speak(chat_id, "Golden grubu selamlar, Katya sesli sohbete katıldı!")
    if err:
        log_event("ERR", f"Bağlantı hatası: {err}")
        await update.message.reply_text(f"Bağlantı hatası: {err}")
    else:
        stats["in_voice_chat"] = True
        stats["voice_speaks"] += 1
        log_event("VOICE", "Sesli sohbete katılındı.")
        await update.message.reply_text("Sesli sohbete bağlandım!")

async def speak_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lütfen konuşturmak istediğin metni yaz: `/konus Merhaba Golden!`")
        return
        
    text = " ".join(context.args)
    chat_id = update.effective_chat.id
    await update.message.reply_text("Seslendiriliyor...")
    
    err = await voice.join_and_speak(chat_id, text)
    if err:
        log_event("ERR", f"Ses yayın hatası: {err}")
        await update.message.reply_text(f"Konuşma hatası: {err}")
    else:
        stats["in_voice_chat"] = True
        stats["voice_speaks"] += 1
        log_event("VOICE", f"Sesli odada konuşuldu: '{text[:35]}...'")

async def leave_voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await voice.leave_voice(chat_id)
    stats["in_voice_chat"] = False
    log_event("VOICE", "Sesli sohbetten ayrıldı.")
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
    
    # 1. Ses motorunu arka planda non-blocking olarak başlat
    asyncio.create_task(voice.start_voice_client())

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
            log_event("SYS", "Katya Telegram bot dinlemede.")
        except Exception as e:
            log_event("ERR", f"Telegram bot başlatma hatası: {e}")

    yield

    # Kapanış işlemleri
    if bot_app:
        try:
            await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass

    await voice.stop_voice_client()

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
