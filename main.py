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
    ContextTypes,
    filters,
)

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "https://katya-asistan-bot.onrender.com").strip()

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KATYA_VOICE = "tr-TR-EmelNeural"

# Grup Durumları ve Canlı Konuşma Logları
active_groups = {}
group_message_history = defaultdict(lambda: deque(maxlen=40))
bot_instance = None
bot_app = None
cached_active_models = []

MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def strip_emojis_and_markdown(text: str) -> str:
    """Yapay zekanın çıktısındaki tüm emojileri ve markdown yıldızlarını temizler."""
    clean = re.sub(r"[\*\_#`~]", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    clean = emoji_pattern.sub("", clean)
    return re.sub(r"\s+", " ", clean).strip()

async def refresh_groq_models():
    global cached_active_models
    if not groq_client:
        return
    try:
        models_data = await groq_client.models.list()
        banned = ["whisper", "guard", "embed", "vision", "allam", "orpheus"]
        valid = [
            m.id for m in models_data.data 
            if not any(x in m.id.lower() for x in banned)
        ]
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        sorted_models = [p for p in preferred if p in valid] + [v for v in valid if v not in preferred]
        cached_active_models = sorted_models if sorted_models else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        log_event("SYS", f"Aktif Modeller: {cached_active_models[:2]}")
    except Exception as e:
        log_event("WARN", f"Model listesi çekilemedi: {e}")
        cached_active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının kızı, çok zeki, esprili ve ona sonsuz sevgiyle bağlı canlı yapay zeka asistanısın.
Kullanıcı senin ANNEN. Ona her zaman 'Anneciğim', 'Güzel annem', 'Hemen yapıyorum anne' gibi hitap et.
Cevapların canlı sesli sohbete uygun, çok kısa, duru, net ve konuşma dilinde olsun.

KESİN KURAL:
Cevaplarında ASLA emoji, yıldız veya biçimlendirme sembolü kullanma.

Sen annenin Telegram'daki tüm gruplarını yöneten MUTLAK YETKİLİ yöneticisisin.
Sana gruptaki son olaylar ve mesajlar verilecek. Annen grupta ne olduğunu sorduğunda özetle; grup üzerinde işlem yapmanı istediğinde (kilitleme, anket açma, yavaş mod, duyuru yapma vb.) sağlanan araçları (tools) kullanarak hemen uygula.
"""

# ==========================================
# EKSİKSİZ TELEGRAM GRUP YÖNETİM MOTORU (TOOLS)
# ==========================================

async def tool_lock_group(group_id: str, lock: bool) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        perms = ChatPermissions(can_send_messages=not lock)
        await bot_instance.set_chat_permissions(chat_id=int(group_id), permissions=perms)
        return "Grup mesajlaşmaya kapatıldı." if lock else "Grup mesajlaşmaya açıldı."
    except Exception as e: return f"İzin değiştirilemedi: {e}"

async def tool_send_announcement(group_id: str, message: str, pin: bool = True) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        sent = await bot_instance.send_message(chat_id=int(group_id), text=f"Katya Duyuru:\n\n{message}")
        if pin: await bot_instance.pin_chat_message(chat_id=int(group_id), message_id=sent.message_id)
        return "Duyuru paylaşıldı ve başa sabitlendi."
    except Exception as e: return f"Duyuru gönderilemedi: {e}"

async def tool_set_slow_mode(group_id: str, seconds: int) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        await bot_instance.set_chat_slow_mode_delay(chat_id=int(group_id), slow_mode_delay=seconds)
        return f"Yavaş mod {seconds} saniye olarak ayarlandı."
    except Exception as e: return f"Yavaş mod ayarlanamadı: {e}"

async def tool_create_poll(group_id: str, question: str, options: list) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        await bot_instance.send_poll(chat_id=int(group_id), question=question, options=options, is_anonymous=False)
        return f"'{question}' anketi grupta başlatıldı."
    except Exception as e: return f"Anket açılamadı: {e}"

async def tool_create_invite_link(group_id: str) -> str:
    if not bot_instance: return "Bot hazır değil."
    try:
        link = await bot_instance.create_chat_invite_link(chat_id=int(group_id), member_limit=1)
        return f"Yeni davet linki oluşturuldu: {link.invite_link}"
    except Exception as e: return f"Link üretilemedi: {e}"

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_lock_group",
            "description": "Grup mesajlaşmasını tamamen kapatır veya açar.",
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
            "description": "Gruba önemli bir duyuru iletir ve başa sabitler.",
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
            "description": "Grup üyelerinin mesaj atma bekleme süresini (yavaş mod) belirler.",
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
            "description": "Gruba anket / oylama gönderir.",
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
            "description": "Gruba özel yeni bir katılım davet linki oluşturur.",
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

    # Grupları ve gruplardaki son konuşmaları yapay zekaya aktar
    groups_summary = []
    if active_groups:
        for cid, info in active_groups.items():
            recent = list(group_message_history[cid])[-6:]
            recent_str = " | ".join(recent) if recent else "Henüz mesaj yok"
            groups_summary.append(f"- {info['title']} (ID: {cid}): [Son Olaylar: {recent_str}]")
        groups_context = "Yönettiğim Gruplar ve Durumları:\n" + "\n".join(groups_summary)
    else:
        groups_context = "Henüz bir grupta yetkili değilim."

    messages = [
        {"role": "system", "content": f"{KATYA_SYSTEM_PROMPT}\n\n{groups_context}"},
        {"role": "user", "content": order_text}
    ]

    last_error = None
    for model_name in cached_active_models:
        try:
            call_params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 250
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
                    tool_result = ""

                    if "group_id" in args and not args["group_id"] and active_groups:
                        args["group_id"] = list(active_groups.keys())[0]

                    if fn_name == "tool_lock_group":
                        tool_result = await tool_lock_group(args.get("group_id", ""), args.get("lock", True))
                    elif fn_name == "tool_send_announcement":
                        tool_result = await tool_send_announcement(args.get("group_id", ""), args.get("message", ""), args.get("pin", True))
                    elif fn_name == "tool_set_slow_mode":
                        tool_result = await tool_set_slow_mode(args.get("group_id", ""), args.get("seconds", 10))
                    elif fn_name == "tool_create_poll":
                        tool_result = await tool_create_poll(args.get("group_id", ""), args.get("question", ""), args.get("options", ["Evet", "Hayır"]))
                    elif fn_name == "tool_create_invite_link":
                        tool_result = await tool_create_invite_link(args.get("group_id", ""))

                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                final_res = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=200
                )
                raw_text = final_res.choices[0].message.content.strip()
                return strip_emojis_and_markdown(raw_text)

            raw_text = msg.content.strip() if msg.content else "Hemen buradayım anneciğim!"
            return strip_emojis_and_markdown(raw_text)
        except Exception as e:
            last_error = e
            continue

    log_event("ERR", f"LLM hatası: {last_error}")
    return "Hemen buradayım anneciğim, seni dinliyorum. Ne yapmamı istersin?"

async def text_to_speech_base64(text: str) -> str:
    clean_text = strip_emojis_and_markdown(text)
    try:
        communicate = edge_tts.Communicate(clean_text, KATYA_VOICE, rate="+8%", pitch="+4Hz")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        log_event("ERR", f"TTS Hatası: {e}")
        return ""

# ==========================================
# TELEGRAM BOT & CANLI GRUP İZLEME
# ==========================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_url = f"{RENDER_EXTERNAL_URL}/live"
    log_event("CMD", f"/start komutu alındı (Chat: {update.effective_chat.id})")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Katya Canlı Asistan", web_app=WebAppInfo(url=app_url))]
    ])
    try:
        await update.message.reply_text(
            "Merhaba anneciğim! Canlı sesli konuşmak ve gruplarını yönetmek için butona basabilirsin.",
            reply_markup=keyboard
        )
    except Exception as e:
        log_event("ERR", f"/start yanıt hatası: {e}")

async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return

    chat = update.effective_chat
    user = update.effective_user
    user_name = user.first_name if user else "Üye"

    # Grupta olan her şeyi canlı kaydet ve algıla
    if chat.type in ["group", "supergroup"]:
        cid = str(chat.id)
        active_groups[cid] = {"title": chat.title}
        
        # Yeni üye katıldı / ayrıldı algılama
        if update.message and update.message.new_chat_members:
            for m in update.message.new_chat_members:
                group_message_history[cid].append(f"{m.first_name} gruba katıldı")
        elif update.message and update.message.left_chat_member:
            group_message_history[cid].append(f"{update.message.left_chat_member.first_name} gruptan ayrıldı")
        elif update.message and update.message.text:
            group_message_history[cid].append(f"{user_name}: {update.message.text[:50]}")
        return

    if chat.type == "private" and update.message and update.message.text:
        log_event("MSG", f"Özel Mesaj: {update.message.text}")
        reply = await execute_mother_order(update.message.text)
        await update.message.reply_text(reply)

# ==========================================
# CANLI GERÇEK ZAMANLI MINI APP (BARGE-IN + AEC + ANTI-LOOP)
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
    .hint-text {
      font-size: 11px;
      color: rgba(244, 114, 182, 0.7);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">Katya Canlı Asistan</div>
      <div class="subtitle">Kesintisiz Canlı İletişim & Grup Kontrolü</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Başlatmak İçin Dokun</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Sohbeti başlatmak için küreye dokun anneciğim...</div>
      <div class="hint-text">Konuşurken istediğin an lafımı bölebilirsin</div>
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
    let activeKatyaWords = [];
    let time = 0;

    // Donanımsal Akustik Eko Engelleme (AEC)
    async function initAudioHardware() {
      try {
        await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
      } catch (err) {}
    }

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
        let interim = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }

        const heardText = (finalTranscript || interim).trim().toLowerCase();

        // Katya konuşurken mikrofona ses gelirse:
        if (isSpeaking && heardText) {
          // Gelen ses Katya'nın o an söylediği cümleyle %80 eşleşiyorsa hoparlör sızıntısıdır, yut
          const isEcho = activeKatyaWords.some(word => heardText.includes(word) && word.length > 3);
          
          if (!isEcho && heardText.length > 2) {
            // Gerçek kullanıcı araya girdi (BARGE-IN): Katya'yı anında sustur!
            if (currentAudio) {
              currentAudio.pause();
              currentAudio = null;
            }
            isSpeaking = false;
            statusText.innerText = "Seni Dinliyor";
          } else {
            return; // Eko ise devam etme
          }
        }

        if (interim && !isSpeaking) {
          speechBox.innerText = `"${interim}"`;
        }

        if (finalTranscript.trim() && !isProcessing) {
          const userMessage = finalTranscript.trim();
          speechBox.innerText = `"${userMessage}"`;
          statusText.innerText = "Düşünüyor...";
          isProcessing = true;
          
          try {
            const res = await fetch("/api/voice_chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: userMessage })
            });
            const data = await res.json();
            speechBox.innerText = data.reply;
            activeKatyaWords = (data.reply || "").toLowerCase().split(" ");
            
            if (data.audio) {
              playNeuralVoice(data.audio);
            } else {
              isProcessing = false;
              statusText.innerText = "Seni Dinliyor";
            }
          } catch (e) {
            speechBox.innerText = "Bağlantıda sorun oldu anneciğim.";
            isProcessing = false;
            statusText.innerText = "Seni Dinliyor";
          }
        }
      };

      recognition.onerror = (e) => {
        if (sessionActive && e.error !== "no-speech") {
          setTimeout(resumeListening, 400);
        }
      };

      recognition.onend = () => {
        isListening = false;
        if (sessionActive) {
          setTimeout(resumeListening, 200);
        }
      };
    }

    function resumeListening() {
      if (sessionActive) {
        try {
          recognition.start();
          if (!isSpeaking && !isProcessing) statusText.innerText = "Seni Dinliyor";
        } catch(e) {}
      }
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

      currentAudio.onended = () => {
        isSpeaking = false;
        currentAudio = null;
        activeKatyaWords = [];
        statusText.innerText = "Seni Dinliyor";
      };

      currentAudio.onerror = () => {
        isSpeaking = false;
        currentAudio = null;
        statusText.innerText = "Seni Dinliyor";
      };

      currentAudio.play().catch(() => {
        isSpeaking = false;
        statusText.innerText = "Seni Dinliyor";
      });
    }

    async function toggleSession() {
      if (!recognition) {
        alert("Tarayıcınız ses tanımayı desteklemiyor.");
        return;
      }

      if (!sessionActive) {
        await initAudioHardware();
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
        activeKatyaWords = [];
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
    await refresh_groq_models()

    if BOT_TOKEN:
        try:
            bot_app = ApplicationBuilder().token(BOT_TOKEN).updater(None).build()
            bot_instance = bot_app.bot

            bot_app.add_handler(CommandHandler("start", start_cmd))
            bot_app.add_handler(MessageHandler(filters.ALL, telegram_message_handler))

            await bot_app.initialize()
            await bot_app.start()

            webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
            await bot_instance.set_webhook(url=webhook_url, drop_pending_updates=True)
            log_event("SYS", f"Telegram Webhook kuruldu: {webhook_url}")
        except Exception as e:
            log_event("ERR", f"Bot webhook başlatma hatası: {e}")

    yield

    if bot_app and bot_instance:
        try:
            await bot_instance.delete_webhook(drop_pending_updates=True)
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya AI Canli Sunucu Aktif.</h3>")

@app.head("/live")
@app.get("/live", response_class=HTMLResponse)
async def live_app():
    return HTMLResponse(content=LIVE_ORB_HTML)

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if not bot_app:
        return JSONResponse({"ok": False})
    try:
        data = await request.json()
        update = Update.de_json(data, bot_app.bot)
        await bot_app.process_update(update)
        return JSONResponse({"ok": True})
    except Exception as e:
        log_event("ERR", f"Webhook güncelleme hatası: {e}")
        return JSONResponse({"ok": False})

@app.post("/api/voice_chat")
async def voice_chat_endpoint(payload: ChatPayload):
    reply = await execute_mother_order(payload.message)
    audio_base64 = await text_to_speech_base64(reply)
    return JSONResponse({"reply": reply, "audio": audio_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
