import os
import io
import re
import asyncio
import base64
import json
from datetime import datetime
from collections import deque, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import edge_tts
from groq import AsyncGroq

from telegram import Update, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ChatMemberHandler,
    ContextTypes,
    filters,
)

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "https://katya-asistan-bot.onrender.com").strip()

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KATYA_VOICE = "tr-TR-EmelNeural"
GROUPS_FILE = "groups.json"

active_groups = {}
chat_context_log = deque(maxlen=60)
bot_instance = None
bot_app = None
cached_active_models = []

MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# ==========================================
# KALICI DİSK DEPOSU
# ==========================================

def load_groups_disk():
    global active_groups
    if os.path.exists(GROUPS_FILE):
        try:
            with open(GROUPS_FILE, "r", encoding="utf-8") as f:
                active_groups = json.load(f)
            log_event("SYS", f"Diskten yüklendi: {list(active_groups.keys())}")
        except Exception as e:
            log_event("WARN", f"Grup dosyası okunamadı: {e}")

def save_groups_disk():
    try:
        with open(GROUPS_FILE, "w", encoding="utf-8") as f:
            json.dump(active_groups, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_event("WARN", f"Grup diske yazılamadı: {e}")

def strip_emojis_and_markdown(text: str) -> str:
    clean = re.sub(r"[\*\_#`~]", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    clean = emoji_pattern.sub("", clean)
    return re.sub(r"\s+", " ", clean).strip()

async def refresh_groq_models():
    global cached_active_models
    if not groq_client: return
    try:
        models_data = await groq_client.models.list()
        banned = ["whisper", "guard", "embed", "vision", "allam", "orpheus"]
        valid = [m.id for m in models_data.data if not any(x in m.id.lower() for x in banned)]
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        sorted_models = [p for p in preferred if p in valid] + [v for v in valid if v not in preferred]
        cached_active_models = sorted_models if sorted_models else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    except Exception as e:
        cached_active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının akıllı, sadık kızı ve mutlak yetkili Telegram grup yöneticisisin.
Kullanıcı senin ANNEN. Ona daima sevgiyle 'Anneciğim' diye hitap et.

KESİN KURALLAR:
1. GEREKSİZ KONUŞMA, LAFI UZATMA. Sadece yapılan işlemi bildir veya soruya net, kısa, tek cümlelik yanıt ver.
2. Emojiler, semboller, markdown yıldızları (*) KESİNLİKLE YASAKTIR.
3. Annen İngilizce, teknik terimler, soyut veya mecaz ("poll aç", "lockla", "mutele", "ortalığı sakinleştir", "kapıları kapat") kullansa bile anında anla ve Türkçe kısa yanıt ver.

YÖNETİM:
Sana verilen listede grup varsa o grupta hemen işlem yap. Grup belirtilmediyse kayıtlı grubu otomatik hedef al.
"""

# ==========================================
# TELEGRAM GRUP YÖNETİM ARAÇLARI
# ==========================================

async def tool_lock_group(group_id: str, lock: bool) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        perms = ChatPermissions(can_send_messages=not lock)
        await bot_instance.set_chat_permissions(chat_id=int(group_id), permissions=perms)
        durum = "kapatıldı" if lock else "açıldı"
        log_event("ACTION", f"Grup ({group_id}) {durum}.")
        return f"Grup mesajlaşmaya {durum} anneciğim."
    except Exception as e:
        return f"İşlem yapılamadı: {e}"

async def tool_send_announcement(group_id: str, message: str, pin: bool = True) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        sent = await bot_instance.send_message(chat_id=int(group_id), text=f"Katya Duyuru:\n\n{message}")
        if pin: await bot_instance.pin_chat_message(chat_id=int(group_id), message_id=sent.message_id)
        return "Duyuru gruba iletildi ve sabitlendi anneciğim."
    except Exception as e:
        return f"Duyuru gönderilemedi: {e}"

async def tool_set_slow_mode(group_id: str, seconds: int) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        await bot_instance.set_chat_slow_mode_delay(chat_id=int(group_id), slow_mode_delay=seconds)
        return f"Yavaş mod {seconds} saniye yapıldı anneciğim."
    except Exception as e:
        return f"Yavaş mod ayarlanamadı: {e}"

async def tool_create_poll(group_id: str, question: str, options: list) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        if not options or len(options) < 2:
            options = ["Evet", "Hayır"]
        await bot_instance.send_poll(chat_id=int(group_id), question=question, options=options, is_anonymous=False)
        return f"Anket grupta paylaşıldı anneciğim."
    except Exception as e:
        return f"Anket açılamadı: {e}"

async def tool_create_invite_link(group_id: str) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        link = await bot_instance.create_chat_invite_link(chat_id=int(group_id), member_limit=1)
        return f"Yeni davet bağlantısı hazır: {link.invite_link}"
    except Exception as e:
        return f"Link üretilemedi: {e}"

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_lock_group",
            "description": "Grup mesajlaşmasını kapatır veya açar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "lock": {"type": "boolean"}
                },
                "required": ["group_id", "lock"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_send_announcement",
            "description": "Gruba duyuru mesajı geçer ve sabitler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "message": {"type": "string"},
                    "pin": {"type": "boolean"}
                },
                "required": ["group_id", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_set_slow_mode",
            "description": "Yavaş mod süresini ayarlar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "seconds": {"type": "integer"}
                },
                "required": ["group_id", "seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_create_poll",
            "description": "Gruba anket veya oylama açar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["group_id", "question", "options"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_create_invite_link",
            "description": "Gruba özel davet linki üretir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"}
                },
                "required": ["group_id"]
            }
        }
    }
]

async def execute_mother_order(order_text: str) -> str:
    if not groq_client:
        return "Yapay zeka beynim bağlı değil anneciğim."

    if not cached_active_models:
        await refresh_groq_models()

    groups_list_str = "\n".join([f"- {info['title']} (ID: {cid})" for cid, info in active_groups.items()]) if active_groups else "Kayıtlı aktif grup yok."
    recent_events_str = "\n".join(list(chat_context_log)[-8:]) if chat_context_log else "Yeni olay yok."

    context_str = f"Yönetilen Gruplar:\n{groups_list_str}\n\nSon Olaylar:\n{recent_events_str}"

    messages = [
        {"role": "system", "content": f"{KATYA_SYSTEM_PROMPT}\n\n{context_str}"},
        {"role": "user", "content": order_text}
    ]

    for model_name in cached_active_models:
        try:
            call_params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 150
            }
            if not any(x in model_name for x in ["gemma", "whisper", "deepseek"]):
                call_params["tools"] = AVAILABLE_TOOLS
                call_params["tool_choice"] = "auto"

            response = await groq_client.chat.completions.create(**call_params)
            msg = response.choices[0].message

            if getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    target_gid = str(args.get("group_id", ""))
                    if (not target_gid or target_gid == "") and active_groups:
                        target_gid = list(active_groups.keys())[0]

                    tool_result = ""
                    if fn_name == "tool_lock_group":
                        tool_result = await tool_lock_group(target_gid, args.get("lock", True))
                    elif fn_name == "tool_send_announcement":
                        tool_result = await tool_send_announcement(target_gid, args.get("message", ""), args.get("pin", True))
                    elif fn_name == "tool_set_slow_mode":
                        tool_result = await tool_set_slow_mode(target_gid, args.get("seconds", 10))
                    elif fn_name == "tool_create_poll":
                        tool_result = await tool_create_poll(target_gid, args.get("question", ""), args.get("options", ["Evet", "Hayır"]))
                    elif fn_name == "tool_create_invite_link":
                        tool_result = await tool_create_invite_link(target_gid)

                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                final_res = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=100
                )
                return strip_emojis_and_markdown(final_res.choices[0].message.content.strip())

            raw_text = msg.content.strip() if msg.content else "Hemen hallediyorum anneciğim."
            return strip_emojis_and_markdown(raw_text)
        except Exception as e:
            continue

    return "Hemen yapıyorum anneciğim."

async def text_to_speech_base64(text: str) -> str:
    clean_text = strip_emojis_and_markdown(text)
    try:
        communicate = edge_tts.Communicate(clean_text, KATYA_VOICE, rate="+10%", pitch="+3Hz")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        return ""

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_url = f"{RENDER_EXTERNAL_URL}/live"
    log_event("CMD", f"/start komutu geldi (Chat: {update.effective_chat.id})")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Katya Canlı Asistan", web_app=WebAppInfo(url=app_url))]
    ])
    try:
        await update.message.reply_text(
            "Merhaba anneciğim! Canlı konuşmak ve grupları yönetmek için aşağıdaki butona dokunabilirsin.",
            reply_markup=keyboard
        )
    except Exception as e:
        log_event("ERR", f"/start yanıt verilemedi: {e}")

async def bind_group_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return
    log_event("CMD", f"/bagla komutu geldi (Chat: {chat.id}, Tip: {chat.type})")
    
    if chat.type in ["group", "supergroup"]:
        cid = str(chat.id)
        active_groups[cid] = {"title": chat.title}
        save_groups_disk()
        log_event("SYS", f"Grup Hafızaya Bağlandı: {chat.title} ({cid})")
        await update.message.reply_text(f"Bu grubu başarıyla hafızama kaydettim anneciğim: {chat.title}")
    else:
        # Eğer özel mesajda /bagla dendiyse son bilinenleri göster
        if active_groups:
            gruplar = ", ".join([v['title'] for v in active_groups.values()])
            await update.message.reply_text(f"Şu an bağlı olduğum gruplar: {gruplar} anneciğim.")
        else:
            await update.message.reply_text("Beni grubuna ekleyip grupta /bagla yazman gerekiyor anneciğim.")

async def my_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat and chat.type in ["group", "supergroup"]:
        cid = str(chat.id)
        active_groups[cid] = {"title": chat.title}
        save_groups_disk()
        log_event("SYS", f"Grup Katılım Algılandı: {chat.title} ({cid})")

async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return

    chat = update.effective_chat
    user = update.effective_user
    user_name = user.first_name if user else "Biri"
    text = update.message.text if update.message and update.message.text else ""

    # Grupta atılan her mesajda grubu kaydet
    if chat.type in ["group", "supergroup"]:
        cid = str(chat.id)
        if cid not in active_groups:
            active_groups[cid] = {"title": chat.title}
            save_groups_disk()
            log_event("SYS", f"Grup Otomatik Kaydedildi: {chat.title} ({cid})")
        if text:
            chat_context_log.append(f"[{chat.title}] {user_name}: {text[:60]}")
        return

    # Özel mesaj gelirse Katya cevap verir
    if chat.type == "private" and text and not text.startswith("/"):
        chat_context_log.append(f"[Özel] {user_name}: {text[:60]}")
        log_event("MSG", f"Özel Mesaj: {text}")
        reply = await execute_mother_order(text)
        await update.message.reply_text(reply)

# ==========================================
# CANLI MINI APP
# ==========================================

LIVE_ORB_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Katya Live</title>
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
    canvas { width: 100%; height: 100%; cursor: pointer; }

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
      background: rgba(22, 10, 36, 0.85);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 45, 120, 0.3);
      border-radius: 16px;
      padding: 14px 20px;
      text-align: center;
      max-width: 90%;
      font-size: 14px;
      color: #fce7f3;
      box-shadow: 0 8px 32px rgba(157, 0, 255, 0.2);
      min-height: 52px;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">Katya Canlı Asistan</div>
      <div class="subtitle">Otonom Grup & Sesli Kontrol</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Başlatmak İçin Dokun</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Sohbeti başlatmak için dokun anneciğim...</div>
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

    let sessionActive = false;
    let isListening = false;
    let isSpeaking = false;
    let isProcessing = false;
    let currentAudio = null;
    let recognition = null;
    let time = 0;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "tr-TR";
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onstart = () => {
        isListening = true;
        if (!isSpeaking && !isProcessing) {
          statusText.innerText = "Seni Dinliyor";
        }
      };

      recognition.onresult = async (event) => {
        if (isSpeaking || isProcessing) return;

        let interim = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }

        if (interim && !isSpeaking) {
          speechBox.innerText = `"${interim}"`;
        }

        if (finalTranscript.trim() && !isProcessing && !isSpeaking) {
          const userMessage = finalTranscript.trim();
          speechBox.innerText = `"${userMessage}"`;
          statusText.innerText = "İşlem Yapılıyor...";
          isProcessing = true;
          
          try { recognition.stop(); } catch(e) {}

          try {
            const res = await fetch("/api/voice_chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: userMessage })
            });
            const data = await res.json();
            speechBox.innerText = data.reply;
            
            if (data.audio) {
              playNeuralVoice(data.audio);
            } else {
              finishSpeaking();
            }
          } catch (e) {
            speechBox.innerText = "Bağlantıda sorun oldu anneciğim.";
            finishSpeaking();
          }
        }
      };

      recognition.onerror = (e) => {
        if (sessionActive && !isSpeaking && !isProcessing && e.error !== "no-speech") {
          setTimeout(resumeListening, 300);
        }
      };

      recognition.onend = () => {
        isListening = false;
        if (sessionActive && !isSpeaking && !isProcessing) {
          setTimeout(resumeListening, 150);
        }
      };
    }

    function resumeListening() {
      if (sessionActive && !isSpeaking && !isProcessing) {
        try {
          recognition.start();
          statusText.innerText = "Seni Dinliyor";
        } catch(e) {}
      }
    }

    function finishSpeaking() {
      isSpeaking = false;
      isProcessing = false;
      currentAudio = null;
      setTimeout(resumeListening, 350);
    }

    function playNeuralVoice(base64Audio) {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
      }

      currentAudio = new Audio("data:audio/mp3;base64," + base64Audio);
      isSpeaking = true;
      isProcessing = false;
      statusText.innerText = "Katya Konuşuyor";

      try { recognition.stop(); } catch(e) {}

      currentAudio.onended = () => {
        finishSpeaking();
      };

      currentAudio.onerror = () => {
        finishSpeaking();
      };

      currentAudio.play().catch(() => {
        finishSpeaking();
      });
    }

    async function toggleSession() {
      if (!recognition) {
        alert("Tarayıcınız ses tanımayı desteklemiyor.");
        return;
      }

      if (!sessionActive) {
        sessionActive = true;
        speechBox.innerText = "Canlı sohbet başladı, seni dinliyorum anneciğim...";
        resumeListening();
      } else {
        sessionActive = false;
        if (currentAudio) {
          currentAudio.pause();
          currentAudio = null;
        }
        isSpeaking = false;
        isProcessing = false;
        try { recognition.stop(); } catch(e) {}
        statusText.innerText = "Durduruldu";
        speechBox.innerText = "Sohbet durduruldu. Başlatmak için dokun.";
      }
    }

    canvas.addEventListener("click", toggleSession);

    function draw() {
      ctx.clearRect(0, 0, width, height);
      time += 0.03;
      
      let energy = 0.2;
      if (isListening && !isSpeaking) energy = 0.55 + Math.sin(time * 5) * 0.2;
      if (isSpeaking) energy = 0.95 + Math.sin(time * 8) * 0.35;

      const cx = width / 2;
      const cy = height / 2;
      const r = 90 * (1 + energy * 0.35);

      const glow = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 2.2);
      glow.addColorStop(0, isSpeaking ? "rgba(255, 0, 127, 0.6)" : "rgba(157, 0, 255, 0.4)");
      glow.addColorStop(1, "rgba(6, 4, 10, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.2, 0, Math.PI * 2);
      ctx.fill();

      for (let layer = 0; layer < 3; layer++) {
        ctx.beginPath();
        ctx.strokeStyle = layer % 2 === 0 ? "rgba(255, 45, 120, 0.8)" : "rgba(157, 0, 255, 0.8)";
        ctx.lineWidth = 2.5;
        const pts = 60;
        for (let i = 0; i <= pts; i++) {
          const theta = (i / pts) * Math.PI * 2;
          const wave = Math.sin(theta * 4 + time * 3 + layer) * (10 * energy);
          const cr = r + wave - (layer * 10);
          const x = cx + Math.cos(theta) * cr;
          const y = cy + Math.sin(theta) * cr;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }

      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.85);
      core.addColorStop(0, "#ffffff");
      core.addColorStop(0.4, "#ff007f");
      core.addColorStop(0.8, "#7b2cbf");
      core.addColorStop(1, "transparent");
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

class ChatPayload(BaseModel):
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app, bot_instance
    load_groups_disk()
    await refresh_groq_models()

    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).updater(None).build()
            bot_instance = bot_app.bot

            bot_app.add_handler(CommandHandler("start", start_cmd))
            bot_app.add_handler(CommandHandler("bagla", bind_group_cmd))
            bot_app.add_handler(CommandHandler("katya", bind_group_cmd))
            bot_app.add_handler(ChatMemberHandler(my_chat_member_handler, ChatMemberHandler.MY_CHAT_MEMBER))
            bot_app.add_handler(MessageHandler(filters.ALL, telegram_message_handler))

            await bot_app.initialize()
            await bot_app.start()

            webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
            await bot_instance.set_webhook(url=webhook_url, drop_pending_updates=True)
            log_event("SYS", f"Telegram Webhook aktif: {webhook_url}")
        except Exception as e:
            log_event("ERR", f"Bot webhook başlatma hatası: {e}")
