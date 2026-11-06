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
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI HATA] Gemini bağlantı hatası: {e}", flush=True)

# Çoklu grup hafızası ve bot örneği
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

Sen annenin Telegram'daki tüm gruplarını yöneten MUTLAK YETKİLİ yöneticisisin.
Sana verilen araçlarla (tools) grubun her şeyini fiziksel olarak değiştirebilirsin:
- Başlık, açıklama ve profil fotoğrafını değiştirebilirsin.
- Duyuru metinleri ve görselleri paylaşıp sabitleyebilirsin.
- Sabitlenen mesajları kaldırabilirsin.
- Kural çiğneyenleri susturabilir (mute), banlayabilir veya affedebilirsin.
- Mesajları silebilirsin.
- Grupta kim ne konuştuysa hafızandan tarayıp annene özet geçebilirsin.
"""

# ==========================================
# TELEGRAM YÖNETİM ARAÇLARI
# ==========================================

async def tool_list_groups() -> str:
    if not active_groups:
        return "Şu an kayıtlı bir grup göremiyorum anneciğim."
    return "Yönettiğim Gruplar:\n" + "\n".join([f"- {g['title']} (ID: {cid})" for cid, g in active_groups.items()])

async def tool_get_group_intel(chat_id: int) -> str:
    if chat_id not in chat_memories or len(chat_memories[chat_id]) == 0:
        return "Bu grupta henüz kaydedilmiş yeni bir mesaj akışı yok anneciğim."
    recent = list(chat_memories[chat_id])[-40:]
    return "Gruptaki son yazışmalar:\n" + "\n".join([f"[{m['time']}] {m['user']}: {m['text']}" for m in recent])

async def tool_set_chat_title(chat_id: int, title: str) -> str:
    try:
        await bot_instance.set_chat_title(chat_id=chat_id, title=title)
        active_groups[chat_id]["title"] = title
        return f"Grubun adını '{title}' olarak değiştirdim anneciğim."
    except Exception as e:
        return f"Grup adı değiştirilemedi: {e}"

async def tool_set_chat_description(chat_id: int, description: str) -> str:
    try:
        await bot_instance.set_chat_description(chat_id=chat_id, description=description)
        return "Grup açıklamasını başarıyla güncelledim anneciğim."
    except Exception as e:
        return f"Açıklama güncellenemedi: {e}"

async def tool_change_chat_photo(chat_id: int, prompt: str) -> str:
    try:
        clean_p = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{clean_p}?width=800&height=800&nologo=true"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                photo_bytes = io.BytesIO(resp.content)
                photo_bytes.name = "group_avatar.jpg"
                await bot_instance.set_chat_photo(chat_id=chat_id, photo=photo_bytes)
                return "Grubun profil fotoğrafını yeni görselle güncelledim anneciğim!"
        return "Görsel üretilemedi."
    except Exception as e:
        return f"Fotoğraf değiştirilemedi: {e}"

async def tool_post_announcement(chat_id: int, text: str, image_prompt: str = "", pin: bool = True) -> str:
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
    try:
        await bot_instance.unpin_all_chat_messages(chat_id=chat_id)
        return "Gruptaki tüm sabitlenmiş mesajları kaldırdım anneciğim."
    except Exception as e:
        return f"Sabitler kaldırılamadı: {e}"

async def tool_mute_member(chat_id: int, user_id: int, minutes: int = 60) -> str:
    try:
        until = datetime.now() + timedelta(minutes=minutes)
        perms = ChatPermissions(can_send_messages=False, can_send_media_messages=False)
        await bot_instance.restrict_chat_member(chat_id=chat_id, user_id=user_id, permissions=perms, until_date=until)
        return f"{user_id} ID'li kullanıcıyı {minutes} dakika susturdum anneciğim."
    except Exception as e:
        return f"Susturma başarısız: {e}"

async def tool_unmute_member(chat_id: int, user_id: int) -> str:
    try:
        perms = ChatPermissions(
            can_send_messages=True, can_send_media_messages=True,
            can_send_polls=True, can_send_other_messages=True
        )
        await bot_instance.restrict_chat_member(chat_id=chat_id, user_id=user_id, permissions=perms)
        return f"{user_id} ID'li kullanıcının cezasını kaldırdım anneciğim."
    except Exception as e:
        return f"Ceza kaldırılamadı: {e}"

async def tool_ban_member(chat_id: int, user_id: int) -> str:
    try:
        await bot_instance.ban_chat_member(chat_id=chat_id, user_id=user_id)
        return f"{user_id} ID'li kullanıcıyı gruptan tamamen uzaklaştırdım anneciğim."
    except Exception as e:
        return f"Banlama başarısız: {e}"

async def tool_delete_recent_messages(chat_id: int, count: int = 5) -> str:
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
        types.FunctionDeclaration(name="tool_delete_recent_messages", description="Son x mesajı siler.", parameters=types.Schema(type="OBJECT", properties={"chat_id": types.Schema(type="INTEGER"), "count": types.Schema(type="INTEGER")}, required=["chat_id"])),
    ])
]

async def execute_mother_order(order_text: str) -> str:
    if not ai_client:
        return "Yapay zeka beynim bağlı değil anneciğim, lütfen GEMINI_API_KEY ayarını kontrol et."

    groups_context = "Yönettiğim Gruplar:\n" + ("\n".join([f"- {info['title']}: ID {cid}" for cid, info in active_groups.items()]) if active_groups else "Henüz bir grupta değilim.")
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
                log_event("ACTION", f"Katya: {fn} -> {a}")
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
                elif fn == "tool_delete_recent_messages":
                    result = await tool_delete_recent_messages(cid, int(a.get("count", 5)))
                else:
                    result = "Bilinmeyen eylem."

                ack = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Eylem sonucu: {result}. Annene görevi tamamladığını sevgiyle bildir.",
                    config=types.GenerateContentConfig(system_instruction=KATYA_SYSTEM_PROMPT)
                )
                return ack.text.strip()

        return response.text.strip()
    except Exception as e:
        log_event("ERR", f"Yönetim Hatası: {e}")
        return f"Bir aksaklık yaşadım anneciğim: {e}"

# ==========================================
# TELEGRAM MESAJ YÖNLENDİRİCİ
# ==========================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanıcıya Mini App canlı ses butonunu gönderir."""
    chat_url = RENDER_EXTERNAL_URL if RENDER_EXTERNAL_URL else "https://katya-asistan-bot.onrender.com"
    app_url = f"{chat_url}/live"

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

    if chat.type == "private":
        log_event("ORDER", f"Annemden Emir: {text[:40]}")
        try:
            reply = await execute_mother_order(text)
            await update.message.reply_text(reply)
        except Exception as e:
            log_event("ERR", f"Özel mesaj yanıtlama hatası: {e}")
            await update.message.reply_text("Bir aksaklık oldu anneciğim, loglarıma bakar mısın?")

# ==========================================
# CANLI PEMBE/MOR MINI APP HTML ARAYÜZÜ
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
    canvas {
      width: 100%;
      height: 100%;
      cursor: pointer;
    }

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
      <span id="statusText">Dinliyor</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">"Dinliyorum anneciğim, seni dinliyorum..."</div>
      <div class="hint-text">Açmak / kapatmak için küreye dokun</div>
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

    let state = "LISTENING";
    let baseRadius = 100;
    let time = 0;
    let audioContext, analyser, microphone, dataArray;
    let isMicActive = false;

    async function toggleMic() {
      if (!isMicActive) {
        try {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          microphone = audioContext.createMediaStreamSource(stream);
          analyser = audioContext.createAnalyser();
          analyser.fftSize = 256;
          microphone.connect(analyser);
          dataArray = new Uint8Array(analyser.frequencyBinCount);
          isMicActive = true;
          statusText.innerText = "Mikrofon Açık";
          speechBox.innerText = "Seni dinliyorum anneciğim, ne yapmamı istersin?";
        } catch (e) {
          speechBox.innerText = "Mikrofon izni verilmedi.";
        }
      } else {
        if (audioContext) audioContext.close();
        isMicActive = false;
        statusText.innerText = "Beklemede";
        speechBox.innerText = "Mikrofon duraklatıldı. Başlatmak için küreye dokun.";
      }
    }

    canvas.addEventListener("click", toggleMic);

    function getAudioLevel() {
      if (!isMicActive || !analyser) {
        return 0.15 + Math.sin(time * 2) * 0.08;
      }
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      return Math.min(1.0, (sum / dataArray.length) / 80);
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      time += 0.025;
      const audio = getAudioLevel();
      const cx = width / 2;
      const cy = height / 2;
      const r = baseRadius * (1 + audio * 0.4);

      // Dış Parıltı
      const glow = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, r * 2.2);
      glow.addColorStop(0, "rgba(255, 45, 120, 0.45)");
      glow.addColorStop(0.5, "rgba(157, 0, 255, 0.25)");
      glow.addColorStop(1, "rgba(6, 4, 10, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.2, 0, Math.PI * 2);
      ctx.fill();

      // Organik Pembe-Mor Dalgalar
      for (let layer = 0; layer < 4; layer++) {
        ctx.beginPath();
        const points = 72;
        const layerColor = layer % 2 === 0 ? "rgba(255, 45, 120, 0.6)" : "rgba(157, 0, 255, 0.6)";
        ctx.strokeStyle = layerColor;
        ctx.lineWidth = 2.5;

        for (let i = 0; i <= points; i++) {
          const theta = (i / points) * Math.PI * 2;
          const wave = Math.sin(theta * 5 + time * 3 + layer) * (12 + audio * 35) +
                       Math.cos(theta * 3 - time * 2) * (8 + audio * 20);
          const currentR = r + wave - (layer * 12);
          const x = cx + Math.cos(theta) * currentR;
          const y = cy + Math.sin(theta) * currentR;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }

      // İç Çekirdek
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.85);
      core.addColorStop(0, "rgba(255, 255, 255, 0.95)");
      core.addColorStop(0.3, "rgba(255, 0, 127, 0.85)");
      core.addColorStop(0.7, "rgba(123, 44, 191, 0.7)");
      core.addColorStop(1, "rgba(15, 5, 29, 0.4)");
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

bot_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app, bot_instance
    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
            bot_instance = bot_app.bot
            
            # Eski Webhook kalıntılarını silip güncellemeleri temizle
            try:
                await bot_instance.delete_webhook(drop_pending_updates=True)
                log_event("SYS", "Eski Telegram Webhook kaydı temizlendi.")
            except Exception as we:
                log_event("WARN", f"Webhook temizlenirken uyarı: {we}")

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

@app.get("/live", response_class=HTMLResponse)
async def live_app():
    """Telegram Mini App canlı küre sayfası."""
    return HTMLResponse(content=LIVE_ORB_HTML)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya AI Canli Sunucu Calisiyor. /live adresinden Mini App'e erisebilirsiniz.</h3>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
