import os
import asyncio
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from telegram import Update, ChatPermissions
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")

# Staff Paneli Veri Havuzu
MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)
stats = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_messages": 0,
    "total_commands": 0,
    "moderation_actions": 0,
    "unique_users": set(),
}

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})

# --- Telegram Komutları & Yetenekleri ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    
    msg = (
        f"👑 Merhaba {user.first_name}! Ben **Katya Asistan**.\n\n"
        f"**Golden** grubunun asistanı ve moderatörüyüm.\n"
        f"Komutlar ve yardım için: /yardim"
    )
    log_event("CMD", f"/start kullandı: @{user.username or user.id}")
    await update.message.reply_text(msg, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    help_text = (
        "🛠 **Katya Asistan Komut Listesi**\n\n"
        "• /kurallar - Golden grubu kurallarını listeler.\n"
        "• /rapor - Bot ve grup istatistiklerini gösterir.\n\n"
        "👮 **Yönetici (Staff) Komutları:**\n"
        "*(Bir kullanıcının mesajına yanıt vererek kullanın)*\n"
        "• `/ban` - Kullanıcıyı gruptan yasaklar.\n"
        "• `/mute <dakika>` - Kullanıcıyı geçici olarak susturur (örn: `/mute 10`).\n"
        "• `/unmute` - Kullanıcının susturmasını kaldırır."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    rules_text = (
        "📜 **Golden Grubu Kuralları:**\n"
        "1. Karşılıklı saygı ve nezaket esastır.\n"
        "2. Reklam, spam ve izinsiz bağlantı paylaşımı yasaktır.\n"
        "3. Grup huzurunu bozan davranışlar doğrudan susturma/yasaklama sebebidir."
    )
    await update.message.reply_text(rules_text, parse_mode="Markdown")

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    report_text = (
        "📊 **Katya Sistem & Moderasyon Raporu**\n\n"
        f"• Aktif Başlangıç: `{stats['start_time']}`\n"
        f"• Toplam Mesaj: `{stats['total_messages']}`\n"
        f"• Tekil Etkileşim: `{len(stats['unique_users'])}`\n"
        f"• Moderasyon İşlemi: `{stats['moderation_actions']}`"
    )
    await update.message.reply_text(report_text, parse_mode="Markdown")

# --- Moderasyon Fonksiyonları ---

async def is_admin(update: Update, user_id: int) -> bool:
    chat_admins = await update.effective_chat.get_administrators()
    return any(admin.user.id == user_id for admin in chat_admins)

async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    
    if not await is_admin(update, user.id):
        await update.message.reply_text("Bu komutu sadece grup yöneticileri kullanabilir.")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("Yasaklamak istediğin kullanıcının mesajını yanıtlayarak `/ban` yazmalısın.")
        return

    target_user = update.message.reply_to_message.from_user
    try:
        await context.bot.ban_chat_member(chat_id=update.effective_chat.id, user_id=target_user.id)
        stats["moderation_actions"] += 1
        log_event("STAFF", f"BAN: @{target_user.username or target_user.id} yasaklandı. (Uygulayan: @{user.username or user.id})")
        await update.message.reply_text(f"⛔ {target_user.first_name} gruptan uzaklaştırıldı.")
    except Exception as e:
        await update.message.reply_text(f"Hata: Kullanıcı yasaklanamadı ({e}). Botun yönetici yetkilerini kontrol edin.")

async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    
    if not await is_admin(update, user.id):
        await update.message.reply_text("Bu komutu sadece grup yöneticileri kullanabilir.")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("Susturmak istediğin kullanıcının mesajını yanıtlayarak `/mute <dakika>` yazmalısın.")
        return

    target_user = update.message.reply_to_message.from_user
    minutes = 10
    if context.args and context.args[0].isdigit():
        minutes = int(context.args[0])

    until_date = datetime.now() + timedelta(minutes=minutes)
    permissions = ChatPermissions(can_send_messages=False)

    try:
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            permissions=permissions,
            until_date=until_date
        )
        stats["moderation_actions"] += 1
        log_event("STAFF", f"MUTE: @{target_user.username or target_user.id} {minutes} dk susturuldu.")
        await update.message.reply_text(f"🔇 {target_user.first_name} {minutes} dakika süreyle susturuldu.")
    except Exception as e:
        await update.message.reply_text(f"Hata: {e}")

async def unmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["total_commands"] += 1
    user = update.effective_user
    
    if not await is_admin(update, user.id):
        await update.message.reply_text("Bu komutu sadece grup yöneticileri kullanabilir.")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("Susturmasını kaldırmak istediğin kullanıcının mesajını yanıtlayarak `/unmute` yaz.")
        return

    target_user = update.message.reply_to_message.from_user
    permissions = ChatPermissions(
        can_send_messages=True,
        can_send_media_messages=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True
    )
    try:
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            permissions=permissions
        )
        stats["moderation_actions"] += 1
        log_event("STAFF", f"UNMUTE: @{target_user.username or target_user.id} susturması kaldırıldı.")
        await update.message.reply_text(f"🔊 {target_user.first_name} artık mesaj gönderebilir.")
    except Exception as e:
        await update.message.reply_text(f"Hata: {e}")

# --- Grup Üye Giriş/Çıkış Takibi ---

async def member_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.new_chat_members:
        for member in update.message.new_chat_members:
            if member.id == context.bot.id:
                await update.message.reply_text("Golden grubu selamlar! Ben Katya Asistan. Yönetim ve asistanlık için hazırım.")
                log_event("SYS", "Katya gruba eklendi.")
            else:
                log_event("JOIN", f"Yeni Üye: @{member.username or member.id}")
                await update.message.reply_text(f"Aramıza hoş geldin, {member.first_name}! /kurallar yazarak grup kurallarını okuyabilirsin.")

    elif update.message.left_chat_member:
        left_user = update.message.left_chat_member
        log_event("LEAVE", f"Üye Ayrıldı: @{left_user.username or left_user.id}")

async def handle_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    stats["total_messages"] += 1
    user = update.effective_user
    stats["unique_users"].add(user.id)
    text = update.message.text
    
    # Katya ismi geçerse yanıt ver
    if "katya" in text.lower():
        log_event("KATYA", f"Etiketlendi: @{user.username or user.id}")
        await update.message.reply_text("Dinliyorum, Golden grubu için buradayım!")
    else:
        log_event("MSG", f"@{user.username or user.id}: {text[:35]}")

# --- Staff Dashboard (Web Arayüzü) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Katya Asistan — Staff Paneli</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="7">
    <style>
        body { background-color: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .card { background-color: #161e2e; border: 1px solid #283548; border-radius: 12px; }
        .log-box { background: #070a11; border-radius: 8px; padding: 15px; font-family: 'SFMono-Regular', Consolas, monospace; max-height: 480px; overflow-y: auto; font-size: 0.9rem; }
        .stat-val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
        .badge-cmd { background-color: #3b82f6; }
        .badge-staff { background-color: #ef4444; }
        .badge-join { background-color: #10b981; }
        .badge-leave { background-color: #f59e0b; }
        .badge-katya { background-color: #8b5cf6; }
        .badge-msg { background-color: #64748b; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <header class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom border-secondary">
            <div>
                <h2 class="h4 mb-0 fw-bold">Katya Asistan <span class="badge bg-primary fs-6">Staff Dashboard</span></h2>
                <small class="text-secondary">Golden Grubu Yönetim Konsolu</small>
            </div>
            <span class="badge bg-success px-3 py-2">Çevrimiçi (Canlı)</span>
        </header>

        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Sunucu Başlangıcı</div>
                    <div class="fs-6 mt-1 text-light">{start_time}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Toplam Grup Mesajı</div>
                    <div class="stat-val">{total_messages}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Çalıştırılan Komut</div>
                    <div class="stat-val">{total_commands}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3">
                    <div class="text-secondary small">Moderasyon Eylemleri</div>
                    <div class="stat-val text-warning">{moderation_actions}</div>
                </div>
            </div>
        </div>

        <div class="card p-3">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="card-title mb-0">Canlı Staff & Sistem Logları</h5>
                <small class="text-secondary">Her 7 saniyede bir otomatik yenilenir</small>
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
        
        # Komut Dinleyicileri
        bot_app.add_handler(CommandHandler("start", start_command))
        bot_app.add_handler(CommandHandler("yardim", help_command))
        bot_app.add_handler(CommandHandler("kurallar", rules_command))
        bot_app.add_handler(CommandHandler("rapor", report_command))
        bot_app.add_handler(CommandHandler("ban", ban_command))
        bot_app.add_handler(CommandHandler("mute", mute_command))
        bot_app.add_handler(CommandHandler("unmute", unmute_command))
        
        # Giriş/Çıkış ve Mesaj Takibi
        bot_app.add_handler(MessageHandler(filters.StatusUpdate.ALL, member_activity))
        bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_messages))

        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()
        log_event("SYS", "Katya Asistan aktif hale getirildi.")
    else:
        log_event("ERR", "BOT_TOKEN bulunamadı!")
    
    yield
    
    if bot_app:
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    badge_map = {
        "STAFF": "badge-staff",
        "CMD": "badge-cmd",
        "JOIN": "badge-join",
        "LEAVE": "badge-leave",
        "KATYA": "badge-katya",
        "MSG": "badge-msg"
    }
    
    logs_html_list = []
    for log in logs_queue:
        badge_cls = badge_map.get(log["level"], "bg-secondary")
        logs_html_list.append(
            f"<div class='mb-1'><span class='text-secondary'>[{log['time']}]</span> "
            f"<span class='badge {badge_cls}'>{log['level']}</span> {log['message']}</div>"
        )
    
    rendered_html = HTML_TEMPLATE.format(
        start_time=stats["start_time"],
        total_messages=stats["total_messages"],
        total_commands=stats["total_commands"],
        moderation_actions=stats["moderation_actions"],
        logs_html="".join(logs_html_list) if logs_html_list else "<span class='text-muted'>Kayıt bekleniyor...</span>"
    )
    return HTMLResponse(content=rendered_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
