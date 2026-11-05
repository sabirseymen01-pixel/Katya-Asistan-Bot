import os
import asyncio
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import edge_tts

from telegram import Update, ChatPermissions
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Ortam Değişkenleri
BOT_TOKEN = os.getenv("BOT_TOKEN")
KATYA_VOICE = "tr-TR-EmelNeural"  # Doğal Türkçe kadın sesi

# Staff Paneli Havuzu
MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "voice_interactions": 0,
    "moderation_actions": 0,
    "warnings": {},  # {user_id: count}
    "unique_users": set(),
}

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})

# --- Ses Üretme (TTS) Yardımcısı ---

async def generate_voice(text: str, output_path: str = "katya_reply.mp3"):
    communicate = edge_tts.Communicate(text, KATYA_VOICE)
    await communicate.save(output_path)
    return output_path

# --- Yetkilendirme & Moderasyon ---

async def is_admin(update: Update, user_id: int) -> bool:
    try:
        admins = await update.effective_chat.get_administrators()
        return any(admin.user.id == user_id for admin in admins)
    except Exception:
        return False

async def warn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.effective_user.id):
        await update.message.reply_text("Bu komut sadece Staff/Yöneticiler içindir.")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("Uyarılacak kişinin mesajını yanıtlayarak `/uyar` yazın.")
        return

    target = update.message.reply_to_message.from_user
    count = stats["warnings"].get(target.id, 0) + 1
    stats["warnings"][target.id] = count
    stats["moderation_actions"] += 1

    log_event("STAFF", f"UYARI: @{target.username or target.id} ({count}/3)")

    if count >= 3:
        # 3. uyarıda 1 saat mute
        until = datetime.now() + timedelta(hours=1)
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until
        )
        stats["warnings"][target.id] = 0
        await update.message.reply_text(f"⚠️ {target.first_name} 3 uyarı sınırına ulaştığı için 1 saat susturuldu.")
    else:
        await update.message.reply_text(f"⚠️ {target.first_name} uyarıldı! (Uyarı: {count}/3)")

# --- Sesli & Metin Etkileşim Motoru ---

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gruptan gelen ses kayıtlarını yakalar ve yapay zeka ile sesli döner."""
    stats["voice_interactions"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    
    log_event("VOICE", f"@{user.username or user.id} sesli mesaj iletti.")
    
    # Simüle edilen LLM cevabı (Buraya Whisper/OpenAI API bağlanabilir)
    cevap_metni = f"Merhaba {user.first_name}, sesli mesajınızı dinledim. Golden grubunun asistanı olarak kaydettim."
    
    audio_file = f"voice_{user.id}.mp3"
    await generate_voice(cevap_metni, audio_file)
    
    await update.message.reply_voice(voice=open(audio_file, "rb"), caption="Katya Sesli Yanıt")
    if os.path.exists(audio_file):
        os.remove(audio_file)

async def handle_text_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Metin filtreleme, Katya çağrıları ve genel grup akışı."""
    if not update.message or not update.message.text:
        return
        
    stats["total_messages"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    text = update.message.text

    # Reklam/Spam Link Denetimi
    if ("t.me/" in text or "http://" in text or "https://" in text) and not await is_admin(update, user.id):
        await update.message.delete()
        log_event("STAFF", f"SPAM SİLİNDİ: @{user.username or user.id}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⚠️ @{user.username or user.first_name}, grupta izinsiz link paylaşımı yasaktır.")
        return

    # İsmi ile seslenme tetikleyicisi
    if "katya" in text.lower():
        log_event("AI", f"Katya etkileşimi: @{user.username or user.id}")
        yanit = f"Buradayım {user.first_name}! Golden grubunu yönetiyor ve dinliyorum. Komutlar için /yardim yazabilirsin."
        
        # İsterse sesli yanıt üretip gönderme seçeneği
        if "sesli" in text.lower() or "konuş" in text.lower():
            ses_dosyasi = f"katya_talk_{user.id}.mp3"
            await generate_voice(yanit, ses_dosyasi)
            await update.message.reply_voice(voice=open(ses_dosyasi, "rb"))
            if os.path.exists(ses_dosyasi):
                os.remove(ses_dosyasi)
        else:
            await update.message.reply_text(yanit)
    else:
        log_event("MSG", f"@{user.username or user.id}: {text[:30]}")

# --- Web Staff Dashboard (Arayüz) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Katya Staff Suite — Golden Admin</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ background-color: #080c14; color: #e2e8f0; font-family: system-ui, sans-serif; }}
        .card {{ background-color: #0f172a; border: 1px solid #1e293b; border-radius: 10px; }}
        .log-box {{ background: #020617; border-radius: 8px; padding: 12px; font-family: monospace; max-height: 420px; overflow-y: auto; font-size: 0.85rem; }}
        .badge-voice {{ background-color: #8b5cf6; }}
        .badge-staff {{ background-color: #ef4444; }}
        .badge-ai {{ background-color: #06b6d4; }}
        .badge-cmd {{ background-color: #3b82f6; }}
        .stat-num {{ font-size: 1.8rem; font-weight: bold; color: #38bdf8; }}
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-dark">
            <div>
                <h4 class="mb-0 fw-bold">Katya Asistan <span class="badge bg-danger">Staff Panel</span></h4>
                <small class="text-secondary">Golden Grubu Sesli ve Metin Denetim Merkezi</small>
            </div>
            <span class="badge bg-success px-3 py-2">Çevrimiçi & Dinlemede</span>
        </div>

        <div class="row g-2 mb-3">
            <div class="col-md-3">
                <div class="card p-3">
                    <span class="text-secondary small">Toplam Mesaj</span>
                    <div class="stat-num">{total_messages}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <span class="text-secondary small">Sesli İletişim (TTS/Ses)</span>
                    <div class="stat-num text-info">{voice_interactions}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <span class="text-secondary small">Moderasyon İşlemleri</span>
                    <div class="stat-num text-warning">{moderation_actions}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <span class="text-secondary small">Aktif Kullanıcı</span>
                    <div class="stat-num text-light">{unique_users_count}</div>
                </div>
            </div>
        </div>

        <div class="card p-3">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h6 class="mb-0">Canlı Sistem, Ses & Güvenlik Akışı</h6>
                <small class="text-secondary">5 sn aralıkla yenilenir</small>
            </div>
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
        
        # Moderasyon & Komutlar
        bot_app.add_handler(CommandHandler("uyar", warn_command))
        
        # Mesaj & Ses Algılayıcılar
        bot_app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_messages))

        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()
        log_event("SYS", "Katya ses ve yönetim motoru başlatıldı.")
    else:
        log_event("ERR", "BOT_TOKEN eksik!")
        
    yield
    
    if bot_app:
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    badge_map = {
        "VOICE": "badge-voice",
        "STAFF": "badge-staff",
        "AI": "badge-ai",
        "CMD": "badge-cmd",
        "SYS": "bg-secondary",
        "MSG": "bg-dark"
    }
    logs_html_list = []
    for log in logs_queue:
        badge_cls = badge_map.get(log["level"], "bg-secondary")
        logs_html_list.append(
            f"<div class='mb-1'><span class='text-secondary'>[{log['time']}]</span> "
            f"<span class='badge {badge_cls}'>{log['level']}</span> {log['message']}</div>"
        )

    rendered = HTML_TEMPLATE.format(
        total_messages=stats["total_messages"],
        voice_interactions=stats["voice_interactions"],
        moderation_actions=stats["moderation_actions"],
        unique_users_count=len(stats["unique_users"]),
        logs_html="".join(logs_html_list) if logs_html_list else "<span class='text-muted'>Veri bekleniyor...</span>"
    )
    return HTMLResponse(content=rendered)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
