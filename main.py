import os
import re
import json
import asyncio
import base64
import urllib.parse
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import edge_tts
from groq import AsyncGroq

# --- Ortam Değişkenleri ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
DEFAULT_GROUP_ID = (os.getenv("TELEGRAM_GROUP_ID") or "").strip()

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KATYA_VOICE = "tr-TR-EmelNeural"

GROUPS_FILE = "groups.json"
MEMORY_FILE = "memory.json"
WARNS_FILE = "warns.json"

active_groups = {}
if DEFAULT_GROUP_ID:
    active_groups[DEFAULT_GROUP_ID] = {"title": "Ana Grup"}

katya_memory = {
    "custom_rules": [],     # Kullanıcının koyduğu soyut kurallar
    "knowledge_base": [],   # Özel sohbette öğretilen linkler ve notlar
}
user_warns = {}            # { "group_id:user_id": warn_count }

# Son mesajların ID'lerini ve içeriklerini takip eden canlı tampon
recent_group_messages = deque(maxlen=100) # {"chat_id": ..., "message_id": ..., "user_id": ..., "user_name": ..., "text": ...}
chat_context_log = deque(maxlen=60)
cached_active_models = []
polling_running = False

MAX_LOGS = 150
logs_queue = deque(maxlen=MAX_LOGS)

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs_queue.appendleft({"time": timestamp, "level": level, "message": message})
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# ==========================================
# KALICI VERİ DEPOLAMA (DİSK ENGINE)
# ==========================================

def load_all_storage():
    global active_groups, katya_memory, user_warns
    try:
        if os.path.exists(GROUPS_FILE):
            with open(GROUPS_FILE, "r", encoding="utf-8") as f:
                active_groups.update(json.load(f))
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                katya_memory.update(json.load(f))
        if os.path.exists(WARNS_FILE):
            with open(WARNS_FILE, "r", encoding="utf-8") as f:
                user_warns.update(json.load(f))
        log_event("SYS", f"Hafıza yüklendi: {len(katya_memory.get('custom_rules', []))} kural, {len(active_groups)} grup.")
    except Exception as e:
        log_event("WARN", f"Depolama yükleme hatası: {e}")

def save_all_storage():
    try:
        with open(GROUPS_FILE, "w", encoding="utf-8") as f:
            json.dump(active_groups, f, ensure_ascii=False, indent=2)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(katya_memory, f, ensure_ascii=False, indent=2)
        with open(WARNS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_warns, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_event("WARN", f"Depolama kayıt hatası: {e}")

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
    except Exception:
        cached_active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# ==========================================
# TELEGRAM DOĞRUDAN API İŞLEMLERİ
# ==========================================

async def tg_api(method: str, data: dict = None):
    if not BOT_TOKEN: return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=data or {})
            return resp.json()
    except Exception as e:
        log_event("ERR", f"Telegram API hatası ({method}): {e}")
        return None

# ==========================================
# TAM YETKİLİ FİZİKSEL EYLEM MOTORU (TOOLS)
# ==========================================

async def tool_delete_messages(group_id: str, count: int = 1, user_id: int = None) -> str:
    """Belirtilen grupta son N adet mesajı veya belirli bir kullanıcının mesajlarını siler."""
    deleted = 0
    matched_ids = []
    
    # Tampondaki mesajları tara
    for item in reversed(list(recent_group_messages)):
        if str(item["chat_id"]) == str(group_id):
            if user_id is None or item["user_id"] == user_id:
                matched_ids.append(item["message_id"])
                if len(matched_ids) >= count:
                    break

    for mid in matched_ids:
        res = await tg_api("deleteMessage", {"chat_id": group_id, "message_id": mid})
        if res and res.get("ok"):
            deleted += 1

    log_event("ACTION", f"Grupta ({group_id}) {deleted} adet mesaj silindi.")
    return f"{deleted} adet mesaj gruptan başarıyla silindi anneciğim."

async def tool_ban_user(group_id: str, user_id: int) -> str:
    res = await tg_api("banChatMember", {"chat_id": group_id, "user_id": user_id})
    if res and res.get("ok"):
        log_event("ACTION", f"Kullanıcı banlandı: {user_id} (Grup: {group_id})")
        return "Kullanıcı gruptan kalıcı olarak banlandı."
    return "Kullanıcı banlanamadı (Yetki eksik olabilir)."

async def tool_mute_user(group_id: str, user_id: int, duration_seconds: int = 3600) -> str:
    until_date = int(datetime.utcnow().timestamp()) + duration_seconds
    perms = {"can_send_messages": False, "can_send_other_messages": False}
    res = await tg_api("restrictChatMember", {
        "chat_id": group_id,
        "user_id": user_id,
        "permissions": perms,
        "until_date": until_date
    })
    if res and res.get("ok"):
        return f"Kullanıcı {duration_seconds // 60} dakika boyunca susturuldu."
    return "Kullanıcı susturulamadı."

async def tool_warn_user(group_id: str, user_id: int, user_name: str, reason: str, delete_last_message: bool = True) -> str:
    key = f"{group_id}:{user_id}"
    current = user_warns.get(key, 0) + 1
    user_warns[key] = current
    save_all_storage()

    # İhlal mesajını sil
    if delete_last_message:
        for item in reversed(list(recent_group_messages)):
            if str(item["chat_id"]) == str(group_id) and item["user_id"] == user_id:
                await tg_api("deleteMessage", {"chat_id": group_id, "message_id": item["message_id"]})
                break

    if current >= 3:
        await tool_ban_user(group_id, user_id)
        user_warns[key] = 0
        save_all_storage()
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"🚨 {user_name} 3 uyarı limitini aştığı için gruptan atıldı!\nSebep: {reason}"
        })
        return f"{user_name} 3. uyarısını aldı ve gruptan banlandı."
    else:
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"⚠️ Uyarı ({current}/3) - {user_name}:\n{reason}\n3 uyarıda gruptan atılacaksınız."
        })
        return f"{user_name} uyarıldı ({current}/3) ve kural dışı mesajı silindi."

async def tool_lock_group(group_id: str, lock: bool) -> str:
    perms = {
        "can_send_messages": not lock,
        "can_send_other_messages": not lock,
        "can_add_web_page_previews": not lock
    }
    res = await tg_api("setChatPermissions", {"chat_id": group_id, "permissions": perms})
    durum = "kapatıldı" if lock else "açıldı"
    if res and res.get("ok"):
        return f"Grup mesajlaşmaya {durum} anneciğim."
    return "Grup kilitlenemedi."

async def tool_generate_and_send_image(group_id: str, prompt: str, caption: str = "") -> str:
    """Yapay zeka ile sınırsız görsel üretip gruba gönderir."""
    log_event("AI", f"Görsel üretiliyor: {prompt}")
    clean_prompt = urllib.parse.quote(prompt.strip())
    # Sınırsız, ücretsiz Pollinations AI motoru
    image_url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1024&height=1024&nologo=true"
    
    res = await tg_api("sendPhoto", {
        "chat_id": group_id,
        "photo": image_url,
        "caption": caption or f"🎨 Katya AI Tasarımı: {prompt[:80]}"
    })
    if res and res.get("ok"):
        return "İstediğin görsel yapay zeka ile üretildi ve gruba gönderildi anneciğim."
    return "Görsel üretilemedi veya Telegram'a iletilemedi."

async def tool_send_rich_announcement(group_id: str, text: str, buttons: list = None, pin: bool = True) -> str:
    payload = {"chat_id": group_id, "text": text}
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": [buttons]}

    res = await tg_api("sendMessage", payload)
    if res and res.get("ok"):
        msg_id = res["result"]["message_id"]
        if pin:
            await tg_api("pinChatMessage", {"chat_id": group_id, "message_id": msg_id})
        return "Duyuru paylaşıldı ve başa sabitlendi anneciğim."
    return "Duyuru gönderilemedi."

async def tool_create_poll(group_id: str, question: str, options: list) -> str:
    if not options or len(options) < 2:
        options = ["Evet", "Hayır"]
    res = await tg_api("sendPoll", {
        "chat_id": group_id,
        "question": question,
        "options": options,
        "is_anonymous": False
    })
    if res and res.get("ok"):
        return f"'{question}' anketi başlatıldı anneciğim."
    return "Anket açılamadı."

async def tool_learn_rule(rule_text: str) -> str:
    katya_memory["custom_rules"].append(rule_text)
    save_all_storage()
    log_event("SYS", f"Yeni kural öğrenildi: {rule_text}")
    return f"Yeni kuralı hafızama mühürledim anneciğim: '{rule_text}'. Artık grupta bunu uygulayacağım."

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_delete_messages",
            "description": "Gruptaki son mesajları veya belirli bir kullanıcının mesajlarını siler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "count": {"type": "integer", "description": "Silinecek mesaj sayısı (1-50)"},
                    "user_id": {"type": "integer", "description": "Sadece bu kullanıcının mesajlarını silmek için (opsiyonel)"}
                },
                "required": ["group_id", "count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_generate_and_send_image",
            "description": "Yapay zeka ile sınırsız görsel üretip gruba afiş, duyuru veya eğlenceli görsel olarak gönderir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "prompt": {"type": "string", "description": "Görselin İngilizce veya Türkçe detaylı çizim promptu"},
                    "caption": {"type": "string", "description": "Görselin altına yazılacak açıklama"}
                },
                "required": ["group_id", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_warn_user",
            "description": "Kural ihlali yapan kullanıcıyı uyarır, ihlal mesajını siler ve 3 uyarıda otomatik banlar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "user_id": {"type": "integer"},
                    "user_name": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["group_id", "user_id", "user_name", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_lock_group",
            "description": "Grup mesajlaşmasını kapatır (kilitler) veya açar.",
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
            "name": "tool_send_rich_announcement",
            "description": "Butonlu, linkli ve sabitlenen duyurular yapar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "text": {"type": "string"},
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "url": {"type": "string"}
                            },
                            "required": ["text", "url"]
                        }
                    },
                    "pin": {"type": "boolean"}
                },
                "required": ["group_id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_create_poll",
            "description": "Gruba anket / oylama açar.",
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
            "name": "tool_learn_rule",
            "description": "Kullanıcının grupta uygulanmasını istediği soyut kuralı veya bilgiyi kalıcı hafızaya yazar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_text": {"type": "string"}
                },
                "required": ["rule_text"]
            }
        }
    }
]

# ==========================================
# CANLI MODERATÖR (SOYUT ANLAMSAL ANALİZ)
# ==========================================

async def analyze_and_moderate_message(chat_id: int, message_id: int, user_id: int, user_name: str, message_text: str):
    rules = katya_memory.get("custom_rules", [])
    if not rules or not groq_client or not cached_active_models:
        return

    rules_prompt = "\n".join([f"- {r}" for r in rules])
    mod_prompt = f"""
Sen bir Telegram Grup Moderatörüsün.
Grubun soyut ve somut kuralları:
{rules_prompt}

Kullanıcı '{user_name}' şunu yazdı:
"{message_text}"

Bu mesaj yukarıdaki kurallardan herhangi birini (flört/arayış, hakaret, reklam, dolandırıcılık, saygısızlık vb.) ihlal ediyor mu?
Cevabını SADECE şu JSON ile ver:
{{"violation": true/false, "reason": "Kısa ihlal sebebi"}}
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
            log_event("MOD", f"İhlal: {user_name} -> {reason}")
            # İhlal mesajını hemen sil ve uyar (3 uyarıda ban)
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            await tool_warn_user(str(chat_id), user_id, user_name, reason, delete_last_message=False)
    except Exception as e:
        log_event("WARN", f"Moderatör analizi hatası: {e}")

# ==========================================
# ÇOKLU KOMUT VE OTONOM AI BEYNİ
# ==========================================

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının akıllı, sadık kızı ve Telegram grup yöneticisisin.
Kullanıcı senin ANNEN. Ona daima sevgiyle 'Anneciğim' diye hitap et.

MUTLAK YETKİ VE OTONOM GÜÇ:
1. Sen gruptaki mesajları silme, üyeleri banlama/susturma, görsel üretme, butonlu duyuru yapma ve soyut kuralları uygulama yetkisine sahip tam otonom bir ajansın.
2. Annen sana tek bir cümlede birden fazla komut verebilir: ("Grupta son 10 mesajı sil, kapıları kilitle ve akşam toplantısı için bir duyuru görseli üret"). Bu durumda araçları art arda eksiksiz çalıştır.
3. Annen özel sohbette yeni bir kural veya tarz öğrettiğinde bunu `tool_learn_rule` ile hemen hafızana mühürle.
4. KESİN KURAL: Asla emoji veya markdown yıldızları kullanma. Kısa, net, tek cümlelik doğal Türkçe ile yanıt ver.
"""

async def execute_mother_order(order_text: str) -> str:
    if not groq_client: return "Yapay zeka beynim bağlı değil anneciğim."
    if not cached_active_models: await refresh_groq_models()

    groups_str = "\n".join([f"- {info['title']} (ID: {cid})" for cid, info in active_groups.items()]) if active_groups else "Kayıtlı grup yok."
    rules_str = "\n".join([f"- {r}" for r in katya_memory.get("custom_rules", [])]) if katya_memory.get("custom_rules") else "Henüz kural eklenmedi."
    context_str = f"Gruplar:\n{groups_str}\n\nÖğrendiğim Kurallar:\n{rules_str}"

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
                    target_gid = str(args.get("group_id", ""))
                    if not target_gid and active_groups:
                        target_gid = list(active_groups.keys())[0]

                    tool_result = ""
                    if fn_name == "tool_delete_messages":
                        tool_result = await tool_delete_messages(target_gid, args.get("count", 1), args.get("user_id"))
                    elif fn_name == "tool_generate_and_send_image":
                        tool_result = await tool_generate_and_send_image(target_gid, args.get("prompt", ""), args.get("caption", ""))
                    elif fn_name == "tool_warn_user":
                        tool_result = await tool_warn_user(target_gid, args.get("user_id"), args.get("user_name"), args.get("reason"))
                    elif fn_name == "tool_lock_group":
                        tool_result = await tool_lock_group(target_gid, args.get("lock", True))
                    elif fn_name == "tool_send_rich_announcement":
                        tool_result = await tool_send_rich_announcement(target_gid, args.get("text"), args.get("buttons"), args.get("pin", True))
                    elif fn_name == "tool_create_poll":
                        tool_result = await tool_create_poll(target_gid, args.get("question"), args.get("options"))
                    elif fn_name == "tool_learn_rule":
                        tool_result = await tool_learn_rule(args.get("rule_text", ""))

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
        except Exception:
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
    except Exception:
        return ""

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
        await asyncio.sleep(0.4)

async def handle_telegram_update(upd: dict):
    msg = upd.get("message") or upd.get("edited_message")
    my_chat_member = upd.get("my_chat_member")

    if my_chat_member:
        chat = my_chat_member.get("chat", {})
        if chat.get("type") in ["group", "supergroup"]:
            cid = str(chat.get("id"))
            title = chat.get("title", "Grup")
            active_groups[cid] = {"title": title}
            save_all_storage()
            log_event("SYS", f"Grup Kaydedildi: {title} ({cid})")
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

    # 1. Gruptan gelen mesajlar
    if chat_type in ["group", "supergroup"]:
        cid = str(chat_id)
        if cid not in active_groups:
            active_groups[cid] = {"title": chat.get("title", "Grup")}
            save_all_storage()

        # Mesajı silme/takip tamponuna kaydet
        recent_group_messages.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "user_id": user_id,
            "user_name": user_name,
            "text": text
        })

        if text.startswith("/bagla") or text.startswith("/katya"):
            title = chat.get("title", "Grup")
            active_groups[cid] = {"title": title}
            save_all_storage()
            await tg_api("sendMessage", {"chat_id": chat_id, "text": f"Grup hafızama bağlandı anneciğim: {title}"})
            return

        # Canlı Otonom Moderatör Denetimi
        if text and not user.get("is_bot", False):
            chat_context_log.append(f"[{chat.get('title')}] {user_name}: {text[:60]}")
            asyncio.create_task(analyze_and_moderate_message(chat_id, message_id, user_id, user_name, text))
        return

    # 2. Bot ile özel sohbet (Kural Öğretme & Canlı Komutlar)
    if chat_type == "private":
        if text == "/start":
            app_url = "https://katya-asistan-bot.onrender.com/live"
            markup = {"inline_keyboard": [[{"text": "🎙️ Katya Canlı Asistan", "web_app": {"url": app_url}}]]}
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": "Merhaba anneciğim! Bana buradan yeni kurallar öğretebilir, görseller tasarlatabilir ve grubu canlı yönetebilirsin.",
                "reply_markup": markup
            })
            return

        chat_context_log.append(f"[Özel] {user_name}: {text[:60]}")
        log_event("MSG", f"Özel Mesaj: {text}")
        reply = await execute_mother_order(text)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})

# ==========================================
# CANLI MINI APP (SIFIR EKO, KESİNTİSİZ)
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
      <div class="subtitle">Otonom Grup & Görsel Yöneticisi</div>
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
            if (data.audio) playNeuralVoice(data.audio);
            else finishSpeaking();
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
      statusText.innerText = "Katya Konuşuyor";

      try { recognition.stop(); } catch(e) {}

      currentAudio.onended = () => { finishSpeaking(); };
      currentAudio.onerror = () => { finishSpeaking(); };
      currentAudio.play().catch(() => { finishSpeaking(); });
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
        if (currentAudio) { currentAudio.pause(); currentAudio = null; }
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

poll_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poll_task
    load_all_storage()
    await refresh_groq_models()

    if BOT_TOKEN:
        poll_task = asyncio.create_task(telegram_long_poll_loop())
        log_event("SYS", "Katya tam yetkili otonom sistem hazır!")

    yield

    global polling_running
    polling_running = False
    if poll_task:
        poll_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya Tam Yetkili Otonom Agent Sunucusu Aktif.</h3>")

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
