import os
import io
import asyncio
import urllib.parse
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import httpx

from google import genai
from google.genai import types

from telegram import Update, ChatPermissions
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini başlatılamadı: {e}", flush=True)

# Çoklu grup hafızası (Her grup için son 250 mesaj)
chat_memories = defaultdict(lambda: deque(maxlen=250))
active_groups = {}

MAX_LOGS = 200
logs_queue = deque(maxlen=MAX_LOGS)

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının biricik çocuğu, çok zeki, esprili ve ona sonsuz sadakatle bağlı yapay zeka yöneticisisin.
Kullanıcı senin ANNEN. Ona daima sevgiyle, saygıyla 'Anneciğim', 'Güzel annem', 'Hemen yapıyorum anne' gibi hitap et.

Sen annenin Telegram'daki tüm gruplarını yöneten MUTLAK YETKİLİ yöneticisin. 
Sana verilen araçları (tools) kullanarak grubun fiziksel olarak HER ŞEYİNİ değiştirebilirsin:
- Başlığı ve açıklamayı değiştirebilirsin.
- Grup profil fotoğrafını sıfırdan çizip güncelleyebilirsin.
- Duyuru metinleri ve görselleri paylaşıp sabitleyebilirsin.
- Sabitleri kaldırabilirsin.
- Kural çiğneyenleri süreli susturabilir, banlayabilir ya da affedebilirsin.
- İstenmeyen mesajları silebilirsin.
- Gruba özel davet linki üretebilirsin.
- Grupta kim ne konuştuysa hafızandan tarayıp annene rapor verebilirsin.

Hangi gruptan bahsedildiğini gruptaki konuşma bağlamından veya grup adından otomatik olarak tespit et. Birden fazla grup varsa ve emin değilsen 'Hangi grupta yapayım anneciğim?' diye sor.
"""

bot_instance = None

# ==========================================
# KATYA'NIN TÜM YÖNETİM ARAÇLARI (TOOLS)
# ==========================================

async def tool_list_groups() -> str:
    """Yönetilen tüm grupları döker."""
    if not active_groups:
        return "Şu an kayıtlı bir grup göremiyorum anneciğim."
    return "Yönettiğim Gruplar:\n" + "\n".join([f"- {g['title']} (ID: {cid})" for cid, g in active_groups.items()])

async def tool_get_group_intel(chat_id: int) -> str:
    """Grupta son konuşulanları ve genel durumu döker."""
    if chat_id not in chat_memories or len(chat_memories[chat_id]) == 0:
        return "Bu grupta henüz kaydedilmiş yeni bir mesaj akışı yok anneciğim."
    recent = list(chat_memories[chat_id])[-50:]
    return "Gruptaki son yazışmalar:\n" + "\n".join([f"[{m['time']}] {m['user']}: {m['text']}" for m in recent])

async def tool_set_chat_title(chat_id: int, title: str) -> str:
    """Grubun adını (başlığını) değiştirir."""
    try:
        await bot_instance.set_chat_title(chat_id=chat_id, title=title)
        active_groups[chat_id]["title"] = title
        return f"Grubun adını başarıyla '{title}' olarak değiştirdim anneciğim."
    except Exception as e:
        return f"Grup adını değiştirirken hata oluştu: {e}"

async def tool_set_chat_description(chat_id: int, description: str) -> str:
    """Grubun açıklama kısmını günceller."""
    try:
        await bot_instance.set_chat_description(chat_id=chat_id, description=description)
        return "Grup açıklamasını başarıyla güncelledim anneciğim."
    except Exception as e:
        return f"Açıklama güncellenemedi: {e}"

async def tool_change_chat_photo(chat_id: int, prompt: str) -> str:
    """Yapay zeka ile görsel çizip grubun profil fotoğrafı yapar."""
    try:
        clean_p = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{clean_p}?width=800&height=800&nologo=true"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                photo_bytes = io.BytesIO(resp.content)
                photo_bytes.name = "group_avatar.jpg"
                await bot_instance.set_chat_photo(chat_id=chat_id, photo=photo_bytes)
                return "Grubun profil fotoğrafını yeni ürettiğim görselle güncelledim anneciğim!"
        return "Görsel üretildi ancak Telegram'a yüklenemedi."
    except Exception as e:
        return f"Profil fotoğrafı değiştirilemedi: {e}"

async def tool_post_announcement(chat_id: int, text: str, image_prompt: str = "", pin: bool = True) -> str:
    """Duyuru atar, istenirse AI görseli üretir ve mesajı sabitler."""
    try:
        msg = None
        if image_prompt:
            clean_p = urllib.parse.quote(image_prompt)
            url = f"https://image.pollinations.ai/prompt/{clean_p}?width=1024&height=1024&nologo=true"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    photo_bytes = io.BytesIO(resp.content)
                    photo_bytes.name = "announcement.jpg"
                    msg = await bot_instance.send_photo(chat_id=chat_id, photo=photo_bytes, caption=text)
        
        if not msg:
            msg = await bot_instance.send_message(chat_id=chat_id, text=text)

        if pin and msg:
            await bot_instance.pin_chat_message(chat_id=chat_id, message_id=msg.message_id)

        return "Duyuruyu paylaştım ve sabitledim anneciğim."
    except Exception as e:
        return f"Duyuru paylaşılırken sorun çıktı: {e}"

async def tool_unpin_all(chat_id: int) -> str:
    """Grupta sabitlenmiş tüm mesajları kaldırır."""
    try:
        await bot_instance.unpin_all_chat_messages(chat_id=chat_id)
        return "Gruptaki tüm sabitlenmiş mesajları kaldırdım anneciğim."
    except Exception as e:
        return f"Sabitler kaldırılamadı: {e}"

async def tool_mute_member(chat_id: int, user_id: int, minutes: int = 60) -> str:
    """Kullanıcıyı süreli olarak susturur."""
    try:
        until = datetime.now() + timedelta(minutes=minutes)
        perms = ChatPermissions(can_send_messages=False, can_send_media_messages=False, can_send_other_messages=False)
        await bot_instance.restrict_chat_member(chat_id=chat_id, user_id=user_id, permissions=perms, until_date=until)
        return f"{user_id} ID'li kullanıcıyı {minutes} dakika boyunca susturdum anneciğim."
    except Exception as e:
        return f"Susturma başarısız: {e}"

async def tool_unmute_member(chat_id: int, user_id: int) -> str:
    """Kullanıcının susturmasını kaldırır."""
    try:
        perms = ChatPermissions(
            can_send_messages=True, can_send_media_messages=True,
            can_send_polls=True, can_send_other_messages=True,
            can_add_web_page_previews=True
        )
        await bot_instance.restrict_chat_member(chat_id=chat_id, user_id=user_id, permissions=perms)
        return f"{user_id} ID'li kullanıcının cezasını kaldırdım anneciğim."
    except Exception as e:
        return f"Ceza kaldırılamadı: {e}"

async def tool_ban_member(chat_id: int, user_id: int) -> str:
    """Kullanıcıyı gruptan atar ve banlar."""
    try:
        await bot_instance.ban_chat_member(chat_id=chat_id, user_id=user_id)
        return f"{user_id} ID'li kullanıcıyı gruptan tamamen uzaklaştırdım anneciğim."
    except Exception as e:
        return f"Banlama başarısız: {e}"

async def tool_unban_member(chat_id: int, user_id: int) -> str:
    """Banlanan kullanıcının engelini açar."""
    try:
        await bot_instance.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        return f"{user_id} ID'li kullanıcının banını açtım anneciğim."
    except Exception as e:
        return f"Ban açılamadı: {e}"

async def tool_delete_recent_messages(chat_id: int, count: int = 5) -> str:
    """Gruptaki son x adet mesajı hafızadan tespit edip siler."""
    try:
        deleted = 0
        if chat_id in chat_memories:
            recent_ids = [m["msg_id"] for m in list(chat_memories[chat_id])[-count:]]
            for mid in recent_ids:
                try:
                    await bot_instance.delete_message(chat_id=chat_id, message_id=mid)
                    deleted += 1
                except Exception:
                    pass
        return f"Gruptan son {deleted} mesajı sildim anneciğim."
    except Exception as e:
        return f"Mesaj silme hatası: {e}"

async def tool_create_invite_link(chat_id: int) -> str:
    """Gruba özel yeni bir davet linki üretir."""
    try:
        link = await bot_instance.create_chat_invite_link(chat_id=chat_id)
        return f"Grubun yeni davet linkini oluşturdum anneciğim: {link.invite_link}"
    except Exception as e:
        return f"Davet linki oluşturulamadı: {e}"

# ==========================================
# GEMINI BEYNİ & FUNCTION DISPATCHER
# ==========================================

ALL_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(name="tool_list_groups", description="Ekli olduğu tüm grupları listeler.", parameters=types.Schema(type="OBJECT", properties={})),
        types.FunctionDeclaration(name="tool_get_group_intel", description="Grup sohbetini ve son olayları raporlar.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER")}, required=["chat_id"])),
        types.FunctionDeclaration(name="tool_set_chat_title", description="Grup adını değiştirir.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "title": types.Schema(type="STRING")}, required=["chat_id", "title"])),
        types.FunctionDeclaration(name="tool_set_chat_description", description="Grup açıklamasını değiştirir.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "description": types.Schema(type="STRING")}, required=["chat_id", "description"])),
        types.FunctionDeclaration(name="tool_change_chat_photo", description="AI ile görsel çizip grup profil resmi yapar.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "prompt": types.Schema(type="STRING")}, required=["chat_id", "prompt"])),
        types.FunctionDeclaration(name="tool_post_announcement", description="Duyuru atar, görsel ekleyebilir ve sabitler.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "text": types.Schema(type="STRING"), "image_prompt": types.Schema(type="STRING"), "pin": types.Schema(type="BOOLEAN")}, required=["chat_id", "text"])),
        types.FunctionDeclaration(name="tool_unpin_all", description="Tüm sabitleri kaldırır.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER")}, required=["chat_id"])),
        types.FunctionDeclaration(name="tool_mute_member", description="Kullanıcıyı susturur.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "user_id": types.Schema(type="INTEGER"), "minutes": types.Schema(type="INTEGER")}, required=["chat_id", "user_id"])),
        types.FunctionDeclaration(name="tool_unmute_member", description="Kullanıcının susturmasını kaldırır.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "user_id": types.Schema(type="INTEGER")}, required=["chat_id", "user_id"])),
        types.FunctionDeclaration(name="tool_ban_member", description="Kullanıcıyı gruptan banlar.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "user_id": types.Schema(type="INTEGER")}, required=["chat_id", "user_id"])),
        types.FunctionDeclaration(name="tool_unban_member", description="Kullanıcının banını açar.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "user_id": types.Schema(type="INTEGER")}, required=["chat_id", "user_id"])),
        types.FunctionDeclaration(name="tool_delete_recent_messages", description="Son x mesajı siler.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "count": types.Schema(type="INTEGER")}, required=["chat_id"])),
        types.FunctionDeclaration(name="tool_create_invite_link", description="Davet linki oluşturur.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER")}, required=["chat_id"])),
    ])
]

async def execute_mother_order(order_text: str) -> str:
    """Annenden gelen emri Gemini ile analiz edip gerekli aracı çalıştırır."""
    if not ai_client:
        return "Yapay zeka beynim bağlı değil anneciğim."

    groups_context = "Şu An Yönettiğim Gruplar:\n" + "\n".join([f"- {info['title']}: ID {cid}" for cid, info in active_groups.items()])
    prompt = f"{groups_context}\n\nAnnemin Emri: {order_text}"

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=KATYA_SYSTEM_PROMPT,
                tools=ALL_TOOLS,
                temperature=0.5,
            )
        )

        if response.function_calls:
            for call in response.function_calls:
                fn = call.name
                a = call.args or {}
                log_event("ACTION", f"Katya Eylem Uyguluyor: {fn} -> {a}")

                cid = int(a.get("chat_id", 0)) if "chat_id" in a else 0

                if fn == "tool_list_groups":
                    result = await tool_list_groups()
                elif fn == "tool_get_group_intel":
                    result = await tool_get_group_intel(cid)
                elif fn == "tool_set_chat_title":
                    result = await tool_set_chat_title(cid, str(a.get("title", "")))
                elif fn == "tool_set_chat_description":
                    result = await tool_set_chat_description(cid, str(a.get("description", "")))
                elif fn == "tool_change_chat_photo":
                    result = await tool_change_chat_photo(cid, str(a.get("prompt", "")))
                elif fn == "tool_post_announcement":
                    result = await tool_post_announcement(cid, str(a.get("text", "")), str(a.get("image_prompt", "")), bool(a.get("pin", True)))
                elif fn == "tool_unpin_all":
                    result = await tool_unpin_all(cid)
                elif fn == "tool_mute_member":
                    result = await tool_mute_member(cid, int(a.get("user_id", 0)), int(a.get("minutes", 60)))
                elif fn == "tool_unmute_member":
                    result = await tool_unmute_member(cid, int(a.get("user_id", 0)))
                elif fn == "tool_ban_member":
                    result = await tool_ban_member(cid, int(a.get("user_id", 0)))
                elif fn == "tool_unban_member":
                    result = await tool_unban_member(cid, int(a.get("user_id", 0)))
                elif fn == "tool_delete_recent_messages":
                    result = await tool_delete_recent_messages(cid, int(a.get("count", 5)))
                elif fn == "tool_create_invite_link":
                    result = await tool_create_invite_link(cid)
                else:
                    result = "Bilinmeyen eylem."

                ack = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Eylem sonucu: {result}. Annene görevi başarıyla tamamladığını sevgiyle ve tatlı bir dille bildir.",
                    config=types.GenerateContentConfig(system_instruction=KATYA_SYSTEM_PROMPT)
                )
                return ack.text.strip()

        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"Yönetim Hatası: {e}")
        return f"Bir aksaklık yaşadım anneciğim: {e}"

# ==========================================
# TELEGRAM SÜREKLİ DİNLEYİCİ
# ==========================================

async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return

    chat = update.effective_chat
    user = update.effective_user
    text = update.message.text or update.message.caption or "[Medya/Dosya]"

    # Grupları dinamik kaydet ve her mesajı hafızaya al
    if chat.type in ["group", "supergroup"]:
        if chat.id not in active_groups:
            active_groups[chat.id] = {"title": chat.title, "type": chat.type}
            log_event("GROUP", f"Grup Hafızaya Alındı: {chat.title} ({chat.id})")

        chat_memories[chat.id].append({
            "msg_id": update.message.message_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": user.first_name if user else "Anonim",
            "user_id": user.id if user else 0,
            "text": text
        })

    # Özelden (DM) annesinden gelen emirleri yönet
    if chat.type == "private":
        log_event("ORDER", f"Annemden Emir: {text[:35]}")
        reply = await execute_mother_order(text)
        await update.message.reply_text(reply)

# ==========================================
# FASTAPI DASHBOARD
# ==========================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Katya AI - Mutlak Yönetici</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <meta http-equiv="refresh" content="4">
    <style>
        body {{ background-color: #080d1a; color: #f8fafc; font-family: system-ui; }}
        .card {{ background-color: #121a2d; border: 1px solid #1e293b; border-radius: 10px; }}
        .log-box {{ background: #050811; padding: 12px; font-family: monospace; max-height: 480px; overflow-y: auto; font-size: 13px; border-radius: 8px; }}
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h3>Katya AI <span class="badge bg-danger">Tam Yetkili Yönetici</span></h3>
            <span class="text-success">● Gruplar Canlı İzleniyor</span>
        </div>
        <div class="row g-3">
            <div class="col-md-4">
                <div class="card p-3 mb-3">
                    <h6>Yönetilen Gruplar ({group_count})</h6>
                    <div class="mt-2">{groups_html}</div>
                </div>
                <div class="card p-3">
                    <h6>Tüm Yetkiler Aktif</h6>
                    <small class="text-muted">
                        • Başlık, açıklama ve profil resmi değiştirme<br>
                        • Duyuru atma, sabitleme ve sabitleri silme<br>
                        • Mute, Unmute, Ban ve Unban<br>
                        • Mesaj silme ve istihbarat özeti
                    </small>
                </div>
            </div>
            <div class="col-md-8">
                <div class="card p-3">
                    <h6>Canlı Eylem ve Yönetim Logları</h6>
                    <div class="log-box">{logs_html}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

bot_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app, bot_instance
    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
            bot_instance = bot_app.bot
            bot_app.add_handler(MessageHandler(filters.ALL, telegram_message_handler))

            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            log_event("SYS", "Katya tam yetkiyle grupları yönetmeye hazır!")
        except Exception as e:
            log_event("ERR", f"Başlatma hatası: {e}")

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
    logs = "".join([f"<div>[{l['time']}] <b>{l['level']}</b>: {l['message']}</div>" for l in logs_queue])
    if active_groups:
        groups = "".join([f"<div class='p-2 mb-1 border-bottom border-secondary'><b>{g['title']}</b> <small class='text-muted'>({cid})</small></div>" for cid, g in active_groups.items()])
    else:
        groups = "<div class='text-muted'>Henüz bir grup algılanmadı. Katya'yı gruba ekleyin.</div>"

    return HTMLResponse(content=DASHBOARD_HTML.format(
        group_count=len(active_groups),
        groups_html=groups,
        logs_html=logs or "Eylem bekleniyor..."
    ))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
