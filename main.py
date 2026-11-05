import os
import logging
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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

# Ortam Değişkenleri
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Bellek içi log ve istatistik havuzu (Staff Paneli için)
MAX_LOGS = 100
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "total_commands": 0,
    "unique_users": set(),
}

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "level": level, "message": message}
    logs_queue.appendleft(entry)

# --- Telegram Bot Handler Fonksiyonları ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    
    welcome_text = (
        f"Merhaba {user.first_name}! Ben Katya Asistan.\n"
        f"Golden grubu için görev başındayım. Size nasıl yardımcı olabilirim?"
    )
    log_event("CMD", f"/start komutu çalıştırıldı: @{user.username or user.id}")
    await update.message.reply_text(welcome_text)

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)

    report_text = (
        "📊 Katya Sistem Durum Raporu:\n"
        f"• Aktif Başlangıç: {stats['start_time']}\n"
        f"• Toplam İşlenen Mesaj: {stats['total_messages']}\n"
        f"• Tekil Kullanıcı: {len(stats['unique_users'])}"
    )
    log_event("CMD", f"Rapor istendi: @{user.username or user.id}")
    await update.message.reply_text(report_text)

async def handle_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    stats["total_messages"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    text = update.message.text
    
    log_event("MSG", f"@{user.username or user.id}: {text[:40]}")
    
    # Basit asistan yanıt mekanizması
    reply = f"Katya: '{text}' mesajını aldım ve kaydettim."
    await update.message.reply_text(reply)

# --- FastAPI & Dashboard (Staff Panel) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Katya Asistan - Staff Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="10">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; }
        .log-box { background: #020617; border-radius: 8px; padding: 15px; font-family: monospace; max-height: 400px; overflow-y: auto; }
        .badge-cmd { background-color: #3b82f6; }
        .badge-msg { background-color: #10b981; }
        .stat-value { font-size: 2rem; font-weight: bold; color: #38bdf8; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <header class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom border-secondary">
            <h1 class="h3">Katya Asistan — Kontrol Paneli</h1>
            <span class="badge bg-success">Canlı / Aktif</span>
        </header>

        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Başlangıç Zamanı</div>
                    <div class="fs-5 mt-1">{start_time}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Toplam Mesaj</div>
                    <div class="stat-value">{total_messages}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Çalıştırılan Komut</div>
                    <div class="stat-value">{total_commands}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Aktif Kullanıcı</div>
                    <div class="stat-value">{unique_users_count}</div>
                </div>
            </div>
        </div>

        <div class="card p-3">
            <h5 class="card-title mb-3">Sistem & Etkileşim Logları (Son 100)</h5>
            <div class="log-box">
                {logs_html}
            </div>
        </div>
    </div>
</body>
</html>
"""

bot_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app
    if BOT_TOKEN:
        bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
        bot_app.add_handler(CommandHandler("start", start_command))
        bot_app.add_handler(CommandHandler("rapor", report_command))
        bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_messages))

        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()
        log_event("SYS", "Katya Asistan başarıyla başlatıldı.")
    else:
        log_event("ERR", "BOT_TOKEN tanımlanmadı!")
    
    yield
    
    if bot_app:
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    logs_html_list = []
    for log in logs_queue:
        badge_cls = "badge-cmd" if log["level"] == "CMD" else ("badge-msg" if log["level"] == "MSG" else "bg-secondary")
        logs_html_list.append(
            f"<div><span class='text-secondary'>[{log['time']}]</span> "
            f"<span class='badge {badge_cls}'>{log['level']}</span> {log['message']}</div>"
        )
    
    rendered_html = HTML_TEMPLATE.format(
        start_time=stats["start_time"],
        total_messages=stats["total_messages"],
        total_commands=stats["total_commands"],
        unique_users_count=len(stats["unique_users"]),
        logs_html="".join(logs_html_list) if logs_html_list else "<span class='text-muted'>Henüz bir kayıt bulunmuyor...</span>"
    )
    return HTMLResponse(content=rendered_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
