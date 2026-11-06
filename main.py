import os
import re
import json
import asyncio
import base64
import urllib.parse
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import edge_tts
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
MONGO_URI = (os.getenv("MONGO_URI") or "").strip()
DEFAULT_GROUP_ID = (os.getenv("TELEGRAM_GROUP_ID") or "").strip()

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KATYA_VOICE = "tr-TR-EmelNeural"

# MongoDB Bağlantısı
mongo_client = None
db = None
if MONGO_URI:
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client["katya_db"]

scheduled_tasks = {}
cached_active_models = []
polling_running = False

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# ==========================================
# MONGODB ÇOKLU GRUP & KALICI HAFIZA MOTORU
# ==========================================

async def init_db_indexes():
    """MongoDB TTL İndeksi: Mesajları tam 48 saat (Telegram silme sınırı) saklar, sonra otomatik temizler."""
    if db is None:
        return
    try:
        # 48 saat = 172800 saniye sonra kendiliğinden silinen mesaj koleksiyonu
        await db.messages.create_index("created_at", expireAfterSeconds=172800)
        await db.messages.create_index([("group_id", 1), ("message_id", 1)])
        await db.groups.create_index("group_id", unique=True)
        await db.warns.create_index([("group_id", 1), ("user_id", 1)], unique=True)
        log_event("SYS", "MongoDB indeksleri ve 48 saatlik mesaj deposu hazirlandi.")
    except Exception as e:
        log_event("WARN", f"MongoDB indeks hatasi: {e}")

async def db_save_group(group_id: str, title: str):
    if db is None: return
    await db.groups.update_one(
        {"group_id": str(group_id)},
        {"$set": {"title": title, "updated_at": datetime.now(timezone.utc)}},
        upsert=True
    )

async def db_get_groups():
    if db is None: return {}
    cursor = db.groups.find({})
    groups = {}
    async for doc in cursor:
        groups[doc["group_id"]] = {"title": doc.get("title", "Grup")}
    return groups

async def db_save_rule(group_id: str, rule_text: str):
    if db is None: return
    await db.rules.insert_one({
        "group_id": str(group_id),
        "rule": rule_text,
        "created_at": datetime.now(timezone.utc)
    })

async def db_get_rules(group_id: str):
    if db is None: return []
    cursor = db.rules.find({"$or": [{"group_id": str(group_id)}, {"group_id": "GLOBAL"}]})
    rules = []
    async for doc in cursor:
        rules.append(doc["rule"])
    return rules

async def db_record_message(group_id: str, message_id: int, user_id: int, user_name: str, text: str):
    if db is None: return
    await db.messages.insert_one({
        "group_id": str(group_id),
        "message_id": message_id,
        "user_id": user_id,
        "user_name": user_name,
        "text": text,
        "created_at": datetime.now(timezone.utc)
    })

async def db_get_recent_messages(group_id: str, limit: int = 50, user_id: int = None, text_contains: str = None):
    if db is None: return []
    query = {"group_id": str(group_id)}
    if user_id:
        query["user_id"] = user_id
    if text_contains:
        query["text"] = {"$regex": re.escape(text_contains), "$options": "i"}

    cursor = db.messages.find(query).sort("message_id", -1).limit(limit)
    messages = []
    async for doc in cursor:
        messages.append(doc)
    return messages

async def db_get_and_inc_warn(group_id: str, user_id: int):
    if db is None: return 1
    res = await db.warns.find_one_and_update(
        {"group_id": str(group_id), "user_id": user_id},
        {"$inc": {"count": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        upsert=True,
        return_document=True
    )
    return res.get("count", 1)

async def db_reset_warn(group_id: str, user_id: int):
    if db is None: return
    await db.warns.delete_one({"group_id": str(group_id), "user_id": user_id})

# ==========================================
# YARDIMCI VE LLM HAZIRLIK
# ==========================================

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
    return re.sub(r"\s+", " ", emoji_pattern.sub("", clean)).strip()

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
    except Exception:
        cached_active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

async def tg_api(method: str, data: dict = None):
    if not BOT_TOKEN: return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=data or {})
            return resp.json()
    except Exception as e:
        log_event("ERR", f"TG API ({method}): {e}")
        return None

# ==========================================
# FİZİKSEL EYLEM VE YÖNETİM MOTORU
# ==========================================

async def tool_delete_messages(group_id: str, count: int = 1, user_id: int = None, text_contains: str = None) -> str:
    """MongoDB'deki 48 saatlik sınırsız depodan eşleşen mesajları çeker ve Telegram'dan siler."""
    docs = await db_get_recent_messages(group_id, limit=count, user_id=user_id, text_contains=text_contains)
    deleted = 0
    for d in docs:
        res = await tg_api("deleteMessage", {"chat_id": group_id, "message_id": d["message_id"]})
        if res and res.get("ok"):
            deleted += 1
            if db is not None:
                await db.messages.delete_one({"_id": d["_id"]})

    log_event("ACTION", f"Grup ({group_id}) {deleted} mesaj MongoDB kaydindan silindi.")
    return f"{deleted} adet mesaj basariyla silindi annecigim."

async def tool_ban_user(group_id: str, user_id: int) -> str:
    res = await tg_api("banChatMember", {"chat_id": group_id, "user_id": user_id})
    if res and res.get("ok"):
        return "Kullanici gruptan kalici olarak banlandi."
    return "Kullanici banlanamadi."

async def tool_unban_user(group_id: str, user_id: int) -> str:
    res = await tg_api("unbanChatMember", {"chat_id": group_id, "user_id": user_id, "only_if_banned": True})
    if res and res.get("ok"):
        return "Kullanicinin bani kaldirildi."
    return "Ban kaldirilamadi."

async def tool_mute_user(group_id: str, user_id: int, duration_seconds: int = 3600) -> str:
    until_date = int(datetime.now(timezone.utc).timestamp()) + duration_seconds
    perms = {"can_send_messages": False, "can_send_other_messages": False}
    res = await tg_api("restrictChatMember", {
        "chat_id": group_id,
        "user_id": user_id,
        "permissions": perms,
        "until_date": until_date
    })
    if res and res.get("ok"):
        return f"Kullanici {duration_seconds // 60} dakika susturuldu."
    return "Kullanici susturulamadi."

async def tool_unmute_user(group_id: str, user_id: int) -> str:
    perms = {"can_send_messages": True, "can_send_other_messages": True, "can_add_web_page_previews": True}
    res = await tg_api("restrictChatMember", {"chat_id": group_id, "user_id": user_id, "permissions": perms})
    if res and res.get("ok"):
        return "Kullanicinin susturmasi kaldirildi."
    return "Susturma kaldirilamadi."

async def tool_warn_user(group_id: str, user_id: int, user_name: str, reason: str) -> str:
    current = await db_get_and_inc_warn(group_id, user_id)
    if current >= 3:
        await tool_ban_user(group_id, user_id)
        await db_reset_warn(group_id, user_id)
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"🚨 {user_name} 3 uyariya ulastigi icin gruptan defedildi!\nSebep: {reason}"
        })
        return f"{user_name} 3. uyaridan sonra kalici banlandi."
    else:
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"⚠️ Uyari ({current}/3) - {user_name}:\n{reason}\n3 uyarida banlanacaksiniz."
        })
        return f"{user_name} uyarildi ({current}/3)."

async def tool_lock_group(group_id: str, lock: bool) -> str:
    perms = {"can_send_messages": not lock, "can_send_other_messages": not lock, "can_add_web_page_previews": not lock}
    res = await tg_api("setChatPermissions", {"chat_id": group_id, "permissions": perms})
    durum = "kapatildi" if lock else "acildi"
    if res and res.get("ok"):
        return f"Grup mesajlasmaya {durum} annecigim."
    return "Grup izinleri degistirilemedi."

async def tool_generate_and_send_image(group_id: str, prompt: str, caption: str = "") -> str:
    clean_prompt = urllib.parse.quote(prompt.strip())
    image_url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1024&height=1024&nologo=true"
    res = await tg_api("sendPhoto", {
        "chat_id": group_id,
        "photo": image_url,
        "caption": caption or f"🎨 Katya AI Tasarimi: {prompt[:70]}"
    })
    if res and res.get("ok"):
        return "Gorsel uretildi ve gruba gonderildi annecigim."
    return "Gorsel iletilemedi."

async def tool_learn_rule(group_id: str, rule_text: str) -> str:
    await db_save_rule(group_id, rule_text)
    return f"Yeni kural MongoDB hafizasina bu grup icin muhurlendi annecigim: '{rule_text}'."

# ==========================================
# AI AGENT FUNCTION DEFINITIONS
# ==========================================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_delete_messages",
            "description": "Grupta son N mesaji, belirli bir kullanicinin mesajlarini veya icerik tarifine gore mesajlari siler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "count": {"type": "integer", "description": "Silinecek adet"},
                    "text_contains": {"type": "string", "description": "Mesaj icinde aranan kelime"}
                },
                "required": ["group_id", "count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_generate_and_send_image",
            "description": "Yapay zeka ile sinirsiz gorsel uretip gruba yollar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "caption": {"type": "string"}
                },
                "required": ["group_id", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_lock_group",
            "description": "Grubu mesajlasmaya kapatir veya acar.",
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
            "name": "tool_learn_rule",
            "description": "Bu grup icin kalici soyut veya somut moderasyon kurali kaydeder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "rule_text": {"type": "string"}
                },
                "required": ["group_id", "rule_text"]
            }
        }
    }
]

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının akıllı, sadık kızı ve birden fazla Telegram grubunu MongoDB ile bağımsız yöneten yöneticisisin.
Kullanıcı senin ANNEN. Ona daima sevgiyle 'Anneciğim' diye hitap et.

MUTLAK YETKİ VE ÇOKLU GRUP HAFIZASI:
1. Yönettiğin gruplar birbirinden tamamen bağımsızdır. Kurallar ve uyarılar ilgili grubun ID'sine özel işlenir.
2. 48 saat içindeki tüm mesajlar MongoDB'de saklanır; annen mesaj silme istediğinde (`tool_delete_messages`) adet veya kelime filtresiyle doğrudan silebilirsin.
3. KESİN KURAL: Asla emoji veya markdown yıldızları kullanma. Kısa, net, tek cümlelik doğal Türkçe ile yanıt ver.
"""

async def execute_mother_order(order_text: str, target_gid: str = None) -> str:
    if not groq_client: return "Yapay zeka beynim bagli degil annecigim."
    if not cached_active_models: await refresh_groq_models()

    groups_map = await db_get_groups()
    if not target_gid and groups_map:
        target_gid = list(groups_map.keys())[0]

    rules_list = await db_get_rules(target_gid or "GLOBAL")
    rules_str = "\n".join([f"- {r}" for r in rules_list]) if rules_list else "Bu grup icin ozel kural yok."
    groups_str = "\n".join([f"- {v['title']} (ID: {k})" for k, v in groups_map.items()])

    context_str = f"Yonetilen Gruplar:\n{groups_str}\n\nHedef Grup ID: {target_gid}\nBu Grubun Kurallari:\n{rules_str}"

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
                "max_tokens": 250,
                "tools": AVAILABLE_TOOLS,
                "tool_choice": "auto"
            }
            response = await groq_client.chat.completions.create(**call_params)
            msg = response.choices[0].message

            if getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    gid = str(args.get("group_id") or target_gid)

                    tool_result = ""
                    if fn_name == "tool_delete_messages":
                        tool_result = await tool_delete_messages(gid, args.get("count", 1), text_contains=args.get("text_contains"))
                    elif fn_name == "tool_generate_and_send_image":
                        tool_result = await tool_generate_and_send_image(gid, args.get("prompt", ""), args.get("caption", ""))
                    elif fn_name == "tool_lock_group":
                        tool_result = await tool_lock_group(gid, args.get("lock", True))
                    elif fn_name == "tool_learn_rule":
                        tool_result = await tool_learn_rule(gid, args.get("rule_text", ""))

                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                final_res = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=100
                )
                return strip_emojis_and_markdown(final_res.choices[0].message.content.strip())

            raw_text = msg.content.strip() if msg.content else "Hemen hallediyorum annecigim."
            return strip_emojis_and_markdown(raw_text)
        except Exception:
            continue

    return "Hemen yapiyorum annecigim."

async def text_to_speech_base64(text: str) -> str:
    clean_text = strip_emojis_and_markdown(text)
    try:
        communicate = edge_tts.Communicate(clean_text, KATYA_VOICE, rate="+10%", pitch="+3Hz")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception:
        return ""

# ==========================================
# CANLI MODERASYON DENETLEYİCİSİ
# ==========================================

async def analyze_and_moderate_message(chat_id: int, message_id: int, user_id: int, user_name: str, message_text: str):
    cid = str(chat_id)
    rules = await db_get_rules(cid)
    if not rules or not groq_client or not cached_active_models:
        return

    rules_prompt = "\n".join([f"- {r}" for r in rules])
    mod_prompt = f"""
Telegram grup moderatörüsün.
Grup Kuralları:
{rules_prompt}

Kullanıcı '{user_name}': "{message_text}"
Bu mesaj kuralları ihlal ediyor mu?
SADECE JSON: {{"violation": true/false, "reason": "kısa sebep"}}
"""
    try:
        res = await groq_client.chat.completions.create(
            model=cached_active_models[0],
            messages=[{"role": "user", "content": mod_prompt}],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        if data.get("violation"):
            reason = data.get("reason", "Kural ihlali")
            log_event("MOD", f"Ihlal ({cid} - {user_name}): {reason}")
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            await tool_warn_user(cid, user_id, user_name, reason)
    except Exception as e:
        log_event("WARN", f"Moderasyon hatasi: {e}")

# ==========================================
# TELEGRAM POLLING & EVENT LOOP
# ==========================================

async def telegram_long_poll_loop():
    global polling_running
    polling_running = True
    offset = 0

    await tg_api("deleteWebhook", {"drop_pending_updates": True})
    log_event("SYS", "Telegram Polling motoru aktif.")

    while polling_running:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            async with httpx.AsyncClient(timeout=35.0) as client:
                res = await client.post(url, json={"offset": offset, "timeout": 25})
                if res.status_code == 200:
                    data = res.json()
                    if data.get("ok"):
                        for upd in data.get("result", []):
                            offset = upd["update_id"] + 1
                            await handle_telegram_update(upd)
                elif res.status_code == 409:
                    await asyncio.sleep(3)
        except Exception:
            await asyncio.sleep(2)
        await asyncio.sleep(0.3)

async def handle_telegram_update(upd: dict):
    msg = upd.get("message") or upd.get("edited_message")
    my_chat_member = upd.get("my_chat_member")

    if my_chat_member:
        chat = my_chat_member.get("chat", {})
        if chat.get("type") in ["group", "supergroup"]:
            cid = str(chat.get("id"))
            title = chat.get("title", "Grup")
            await db_save_group(cid, title)
            log_event("SYS", f"MongoDB Grup Eklendi: {title} ({cid})")
        return

    if not msg: return

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    chat_type = chat.get("type")
    message_id = msg.get("message_id")
    text = (msg.get("text") or "").strip()
    user = msg.get("from", {})
    user_id = user.get("id")
    user_name = user.get("first_name", "Biri")
    reply_to = msg.get("reply_to_message")

    # 1. Gruptan gelen mesajlar
    if chat_type in ["group", "supergroup"]:
        cid = str(chat_id)
        title = chat.get("title", "Grup")
        await db_save_group(cid, title)

        # Mesajı 48 saatlik MongoDB deposuna yaz
        await db_record_message(cid, message_id, user_id, user_name, text)

        if text.startswith("/bagla") or text.startswith("/katya"):
            await db_save_group(cid, title)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": f"Grup MongoDB hafizasina baglandi: {title}"})
            return

        # Basit komutlar
        if text == "/defol" and reply_to:
            await tool_ban_user(cid, reply_to.get("from", {}).get("id"))
            return
        if text == "/itaat" and reply_to:
            await tool_mute_user(cid, reply_to.get("from", {}).get("id"), 3600)
            return

        # Canlı Kural Analizi
        if text and not user.get("is_bot", False):
            asyncio.create_task(analyze_and_moderate_message(chat_id, message_id, user_id, user_name, text))
        return

    # 2. Özel mesajlar
    if chat_type == "private":
        if text == "/start":
            app_url = "https://katya-asistan-bot.onrender.com/live"
            markup = {"inline_keyboard": [[{"text": "🎙️ Katya Canlı Asistan", "web_app": {"url": app_url}}]]}
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": "Merhaba annecigim! MongoDB destekli coklu grup ve sinirsiz mesaj takip beynim aktif.",
                "reply_markup": markup
            })
            return

        reply = await execute_mother_order(text)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})

# ==========================================
# CANLI MINI APP ARAYÜZÜ
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
      <div class="title">Katya Canli Asistan</div>
      <div class="subtitle">MongoDB Destekli Coklu Grup Yonetimi</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Baslatmak Icin Dokun</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Sohbeti baslatmak icin dokun annecigim...</div>
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
        if (!isSpeaking && !isProcessing) statusText.innerText = "Seni Dinliyor";
      };

      recognition.onresult = async (event) => {
        if (isSpeaking || isProcessing) return;

        let interim = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
          else interim += event.results[i][0].transcript;
        }

        if (interim && !isSpeaking) speechBox.innerText = `"${interim}"`;

        if (finalTranscript.trim() && !isProcessing && !isSpeaking) {
          const userMessage = finalTranscript.trim();
          speechBox.innerText = `"${userMessage}"`;
          statusText.innerText = "Islem Yapiliyor...";
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
            if (data.audio) playNeuralVoice(data.audio);
            else finishSpeaking();
          } catch (e) {
            speechBox.innerText = "Baglantida sorun oldu annecigim.";
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
        if (sessionActive && !isSpeaking && !isProcessing) setTimeout(resumeListening, 150);
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
      statusText.innerText = "Katya Konusuyor";

      try { recognition.stop(); } catch(e) {}

      currentAudio.onended = () => { finishSpeaking(); };
      currentAudio.onerror = () => { finishSpeaking(); };
      currentAudio.play().catch(() => { finishSpeaking(); });
    }

    async function toggleSession() {
      if (!recognition) {
        alert("Tarayiciniz ses tanimayi desteklemiyor.");
        return;
      }
      if (!sessionActive) {
        sessionActive = true;
        speechBox.innerText = "Canli sohbet basladi, seni dinliyorum annecigim...";
        resumeListening();
      } else {
        sessionActive = false;
        if (currentAudio) { currentAudio.pause(); currentAudio = null; }
        isSpeaking = false;
        isProcessing = false;
        try { recognition.stop(); } catch(e) {}
        statusText.innerText = "Durduruldu";
        speechBox.innerText = "Sohbet durduruldu. Baslatmak icin dokun.";
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

poll_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poll_task
    await init_db_indexes()
    await refresh_groq_models()

    if BOT_TOKEN:
        poll_task = asyncio.create_task(telegram_long_poll_loop())
        log_event("SYS", "Katya MongoDB coklu grup motoru devrede!")

    yield

    global polling_running
    polling_running = False
    if poll_task:
        poll_task.cancel()
    if mongo_client:
        mongo_client.close()

app = FastAPI(lifespan=lifespan)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya MongoDB Destekli Agent Sunucusu Aktif.</h3>")

@app.head("/live")
@app.get("/live", response_class=HTMLResponse)
async def live_app():
    return HTMLResponse(content=LIVE_ORB_HTML)

@app.post("/api/voice_chat")
async def voice_chat_endpoint(payload: ChatPayload):
    reply = await execute_mother_order(payload.message)
    audio_base64 = await text_to_speech_base64(reply)
    return JSONResponse({"reply": reply, "audio": audio_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
