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
CONFIG_FILE = "config.json"

active_groups = {}
if DEFAULT_GROUP_ID:
    active_groups[DEFAULT_GROUP_ID] = {"title": "Ana Grup"}

katya_memory = {
    "custom_rules": [],
    "knowledge_base": [],
}
user_warns = {}            # { "group_id:user_id": warn_count }
group_configs = {}         # { "group_id": { "welcome_msg": "...", "btn_text": "...", "btn_url": "..." } }

# Zamanlanmış görevler havuzu: { "task_id": asyncio.Task }
scheduled_tasks = {}
recent_group_messages = deque(maxlen=200)
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
# KALICI VERİ DEPOLAMA
# ==========================================

def load_all_storage():
    global active_groups, katya_memory, user_warns, group_configs
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
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                group_configs.update(json.load(f))
        log_event("SYS", f"Depolama yuklendi. Gruplar: {list(active_groups.keys())}")
    except Exception as e:
        log_event("WARN", f"Depolama yukleme hatasi: {e}")

def save_all_storage():
    try:
        with open(GROUPS_FILE, "w", encoding="utf-8") as f:
            json.dump(active_groups, f, ensure_ascii=False, indent=2)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(katya_memory, f, ensure_ascii=False, indent=2)
        with open(WARNS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_warns, f, ensure_ascii=False, indent=2)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(group_configs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_event("WARN", f"Depolama kayit hatasi: {e}")

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
# TELEGRAM API MOTORU
# ==========================================

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
# FİZİKSEL EYLEM VE YÖNETİM FONKSİYONLARI
# ==========================================

async def tool_delete_messages(group_id: str, count: int = 1, user_id: int = None, text_contains: str = None) -> str:
    deleted = 0
    matched_ids = []
    
    for item in reversed(list(recent_group_messages)):
        if str(item["chat_id"]) == str(group_id):
            u_ok = (user_id is None) or (item["user_id"] == user_id)
            t_ok = (text_contains is None) or (text_contains.lower() in item.get("text", "").lower())
            if u_ok and t_ok:
                matched_ids.append(item["message_id"])
                if len(matched_ids) >= count:
                    break

    for mid in matched_ids:
        res = await tg_api("deleteMessage", {"chat_id": group_id, "message_id": mid})
        if res and res.get("ok"):
            deleted += 1

    log_event("ACTION", f"Grupta ({group_id}) {deleted} mesaj silindi.")
    return f"{deleted} adet mesaj basariyla silindi annecigim."

async def tool_ban_user(group_id: str, user_id: int) -> str:
    res = await tg_api("banChatMember", {"chat_id": group_id, "user_id": user_id})
    if res and res.get("ok"):
        log_event("ACTION", f"Ban: {user_id} (Grup: {group_id})")
        return "Kullanici gruptan kalici olarak banlandi."
    return "Kullanici banlanamadi."

async def tool_unban_user(group_id: str, user_id: int) -> str:
    res = await tg_api("unbanChatMember", {"chat_id": group_id, "user_id": user_id, "only_if_banned": True})
    if res and res.get("ok"):
        return "Kullanicinin bani kaldirildi."
    return "Ban kaldirilamadi."

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
        return f"Kullanici {duration_seconds // 60} dakika susturuldu."
    return "Kullanici susturulamadi."

async def tool_unmute_user(group_id: str, user_id: int) -> str:
    perms = {
        "can_send_messages": True,
        "can_send_other_messages": True,
        "can_add_web_page_previews": True
    }
    res = await tg_api("restrictChatMember", {
        "chat_id": group_id,
        "user_id": user_id,
        "permissions": perms
    })
    if res and res.get("ok"):
        return "Kullanicinin susturmasi kaldirildi."
    return "Susturma kaldirilamadi."

async def tool_warn_user(group_id: str, user_id: int, user_name: str, reason: str) -> str:
    key = f"{group_id}:{user_id}"
    current = user_warns.get(key, 0) + 1
    user_warns[key] = current
    save_all_storage()

    if current >= 3:
        await tool_ban_user(group_id, user_id)
        user_warns[key] = 0
        save_all_storage()
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"🚨 {user_name} 3 uyariya ulastigi icin gruptan defedildi!\nSebep: {reason}"
        })
        return f"{user_name} 3. uyaridan sonra banlandi."
    else:
        await tg_api("sendMessage", {
            "chat_id": group_id,
            "text": f"⚠️ Uyari ({current}/3) - {user_name}:\n{reason}\n3 uyarida banlanacaksiniz."
        })
        return f"{user_name} uyarildi ({current}/3)."

async def tool_unwarn_user(group_id: str, user_id: int, user_name: str) -> str:
    key = f"{group_id}:{user_id}"
    user_warns[key] = 0
    save_all_storage()
    await tg_api("sendMessage", {
        "chat_id": group_id,
        "text": f"✅ {user_name} kullanicisinin tum uyarilari sifirlandi."
    })
    return f"{user_name} uyarilari sifirlandi."

async def tool_pin_message(group_id: str, message_id: int = None, text: str = None) -> str:
    if message_id is None and text:
        res = await tg_api("sendMessage", {"chat_id": group_id, "text": text})
        if res and res.get("ok"):
            message_id = res["result"]["message_id"]

    if message_id:
        res = await tg_api("pinChatMessage", {"chat_id": group_id, "message_id": message_id})
        if res and res.get("ok"):
            return "Mesaj basa sabitlendi annecigim."
    return "Mesaj sabitlenemedi."

async def tool_unpin_message(group_id: str, unpin_all: bool = False) -> str:
    if unpin_all:
        res = await tg_api("unpinAllChatMessages", {"chat_id": group_id})
    else:
        res = await tg_api("unpinChatMessage", {"chat_id": group_id})
    if res and res.get("ok"):
        return "Sabitlenen mesaj basariyla kaldirildi."
    return "Sabit kaldirilamadi."

# ==========================================
# ZAMANLANMIŞ PERİYODİK DUYURU MOTORU
# ==========================================

async def periodic_announcement_worker(task_id: str, group_id: str, interval_min: int, message: str, btn_text: str = None, btn_url: str = None):
    log_event("SCHED", f"Zamanlayici baslatildi: {task_id} ({interval_min} dk)")
    payload = {"chat_id": group_id, "text": message}
    if btn_text and btn_url:
        payload["reply_markup"] = {"inline_keyboard": [[{"text": btn_text, "url": btn_url}]]}

    try:
        while True:
            await asyncio.sleep(interval_min * 60)
            await tg_api("sendMessage", payload)
            log_event("SCHED", f"Zamanli duyuru iletildi: {task_id}")
    except asyncio.CancelledError:
        log_event("SCHED", f"Zamanlayici durduruldu: {task_id}")

def tool_schedule_announcement(group_id: str, interval_min: int, message: str, btn_text: str = None, btn_url: str = None) -> str:
    task_id = f"{group_id}_{interval_min}_{message[:15]}"
    if task_id in scheduled_tasks:
        scheduled_tasks[task_id].cancel()

    task = asyncio.create_task(
        periodic_announcement_worker(task_id, group_id, interval_min, message, btn_text, btn_url)
    )
    scheduled_tasks[task_id] = task
    return f"Her {interval_min} dakikada bir paylasilacak duyuru baslatildi annecigim."

def tool_stop_scheduled_announcement(group_id: str) -> str:
    stopped = 0
    for tid, t in list(scheduled_tasks.items()):
        if tid.startswith(str(group_id)):
            t.cancel()
            del scheduled_tasks[tid]
            stopped += 1
    return f"{stopped} adet zamanlanmis duyuru gorevi durduruldu."

# ==========================================
# AI FUNCTION CALLING ARAÇLARI
# ==========================================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_delete_messages",
            "description": "Grupta son N mesajı, belirli kullanıcının mesajlarını veya belirli kelimeleri içeren mesajları siler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "count": {"type": "integer", "description": "Silinecek adet (1-50)"},
                    "text_contains": {"type": "string", "description": "İçinde geçen kelime veya tarif (opsiyonel)"}
                },
                "required": ["group_id", "count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_schedule_announcement",
            "description": "Grupta periyodik (örneğin her 15 dakikada bir) tekrarlanan duyuru başlatır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "interval_min": {"type": "integer", "description": "Kaç dakikada bir atılacak"},
                    "message": {"type": "string"},
                    "btn_text": {"type": "string"},
                    "btn_url": {"type": "string"}
                },
                "required": ["group_id", "interval_min", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_stop_scheduled_announcement",
            "description": "Gruptaki aktif periyodik zamanlanmış duyuruları durdurur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"}
                },
                "required": ["group_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_pin_message",
            "description": "Yeni bir duyuru mesajı yazıp başa sabitler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["group_id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_unpin_message",
            "description": "Gruptaki sabitlenmiş mesajı veya tüm sabitleri kaldırır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "unpin_all": {"type": "boolean"}
                },
                "required": ["group_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_warn_user",
            "description": "Kullanıcıyı uyarır, 3 uyarıda banlar.",
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
    }
]

KATYA_SYSTEM_PROMPT = """
Sen Katya'sın. Kullanıcının akıllı, sadık kızı ve Telegram grup yöneticisisin.
Kullanıcı senin ANNEN. Ona daima sevgiyle 'Anneciğim' diye hitap et.

YETKİLERİN VE GOLDEN GUARDIAN ÖZELLİKLERİ:
1. Zamanlı periyodik duyurular başlatabilir (`tool_schedule_announcement`) ve durdurabilirsin (`tool_stop_scheduled_announcement`).
2. Mesajları sabitleyebilir, sabitleri kaldırabilir, tarif edilen ya da istenen son mesajları silebilirsin (`tool_delete_messages`).
3. Annen komut tarzında (/zamanla, /duyuru vb.) veya doğrudan doğal dille konuşsa da ne istediğini anla ve araçları çağır.
4. KESİN KURAL: Asla emoji veya markdown yıldızları kullanma. Kısa, net, tek cümlelik doğal Türkçe ile yanıt ver.
"""

async def execute_mother_order(order_text: str) -> str:
    if not groq_client: return "Yapay zeka beynim bagli degil annecigim."
    if not cached_active_models: await refresh_groq_models()

    groups_str = "\n".join([f"- {info['title']} (ID: {cid})" for cid, info in active_groups.items()]) if active_groups else "Kayitli grup yok."
    context_str = f"Gruplar:\n{groups_str}"

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
                        tool_result = await tool_delete_messages(target_gid, args.get("count", 1), text_contains=args.get("text_contains"))
                    elif fn_name == "tool_schedule_announcement":
                        tool_result = tool_schedule_announcement(target_gid, args.get("interval_min", 10), args.get("message", ""), args.get("btn_text"), args.get("btn_url"))
                    elif fn_name == "tool_stop_scheduled_announcement":
                        tool_result = tool_stop_scheduled_announcement(target_gid)
                    elif fn_name == "tool_pin_message":
                        tool_result = await tool_pin_message(target_gid, text=args.get("text"))
                    elif fn_name == "tool_unpin_message":
                        tool_result = await tool_unpin_message(target_gid, args.get("unpin_all", False))
                    elif fn_name == "tool_warn_user":
                        tool_result = await tool_warn_user(target_gid, args.get("user_id"), args.get("user_name"), args.get("reason"))

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
# GOLDEN GUARDIAN KOMUT PARSER VE DİNLEYİCİ
# ==========================================

async def handle_golden_guardian_commands(chat_id: int, text: str, reply_to_msg: dict) -> bool:
    cid = str(chat_id)
    cmd = text.split()[0].lower() if text else ""

    # /welcome <mesaj>
    if cmd == "/welcome":
        val = text[len("/welcome"):].strip()
        if cid not in group_configs: group_configs[cid] = {}
        group_configs[cid]["welcome_msg"] = val
        save_all_storage()
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Hos geldin mesaji guncellendi annecigim."})
        return True

    # /butonmetni <yazı>
    if cmd == "/butonmetni":
        val = text[len("/butonmetni"):].strip()
        if cid not in group_configs: group_configs[cid] = {}
        group_configs[cid]["btn_text"] = val
        save_all_storage()
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Buton metni ayarlandi."})
        return True

    # /butonlinki <url>
    if cmd == "/butonlinki":
        val = text[len("/butonlinki"):].strip()
        if cid not in group_configs: group_configs[cid] = {}
        group_configs[cid]["btn_url"] = val
        save_all_storage()
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Buton linki ayarlandi."})
        return True

    # /duyuru <mesaj> veya /duyuru <mesaj> | <metin> | <link> (Otomatik Sabitlenir)
    if cmd == "/duyuru":
        content = text[len("/duyuru"):].strip()
        parts = [p.strip() for p in content.split("|")]
        payload = {"chat_id": chat_id, "text": f"📢 Katya Duyuru:\n\n{parts[0]}"}
        if len(parts) >= 3:
            payload["reply_markup"] = {"inline_keyboard": [[{"text": parts[1], "url": parts[2]}]]}
        res = await tg_api("sendMessage", payload)
        if res and res.get("ok"):
            mid = res["result"]["message_id"]
            await tg_api("pinChatMessage", {"chat_id": chat_id, "message_id": mid})
        return True

    # /zamanla <dakika> | <mesaj> | <buton> | <link>
    if cmd == "/zamanla":
        content = text[len("/zamanla"):].strip()
        parts = [p.strip() for p in content.split("|")]
        if parts and parts[0].isdigit():
            m_min = int(parts[0])
            msg_txt = parts[1] if len(parts) > 1 else "Duyuru"
            b_txt = parts[2] if len(parts) > 2 else None
            b_url = parts[3] if len(parts) > 3 else None
            res_str = tool_schedule_announcement(cid, m_min, msg_txt, b_txt, b_url)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": res_str})
        return True

    # /zamanladur
    if cmd in ["/zamanladur", "/zamanla_dur"]:
        res_str = tool_stop_scheduled_announcement(cid)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": res_str})
        return True

    # /defol (Ban)
    if cmd == "/defol" and reply_to_msg:
        target_uid = reply_to_msg.get("from", {}).get("id")
        await tool_ban_user(cid, target_uid)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Kullanici gruptan defedildi."})
        return True

    # /undefol (Unban)
    if cmd == "/undefol" and reply_to_msg:
        target_uid = reply_to_msg.get("from", {}).get("id")
        await tool_unban_user(cid, target_uid)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Kullanicinin bani kaldirildi."})
        return True

    # /itaat (Mute)
    if cmd == "/itaat" and reply_to_msg:
        target_uid = reply_to_msg.get("from", {}).get("id")
        await tool_mute_user(cid, target_uid, 3600)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Kullanici 1 saat susturuldu."})
        return True

    # /unitaat (Unmute)
    if cmd == "/unitaat" and reply_to_msg:
        target_uid = reply_to_msg.get("from", {}).get("id")
        await tool_unmute_user(cid, target_uid)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": "Susturma kaldirildi."})
        return True

    # /warn
    if cmd == "/warn" and reply_to_msg:
        target_user = reply_to_msg.get("from", {})
        await tool_warn_user(cid, target_user.get("id"), target_user.get("first_name", "Uye"), "Yonetici uyarisi")
        return True

    # /unwarn
    if cmd == "/unwarn" and reply_to_msg:
        target_user = reply_to_msg.get("from", {})
        await tool_unwarn_user(cid, target_user.get("id"), target_user.get("first_name", "Uye"))
        return True

    return False

# ==========================================
# TELEGRAM POLLING DÖNGÜSÜ
# ==========================================

async def telegram_long_poll_loop():
    global polling_running
    polling_running = True
    offset = 0

    await tg_api("deleteWebhook", {"drop_pending_updates": True})
    log_event("SYS", "Telegram Polling dinleyicisi baslatildi.")

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
            active_groups[cid] = {"title": title}
            save_all_storage()
            log_event("SYS", f"Grup Katilimi: {title} ({cid})")
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
    new_members = msg.get("new_chat_members")

    # Yeni üye katıldığında otomatik hoş geldin mesajı
    if new_members and chat_type in ["group", "supergroup"]:
        cfg = group_configs.get(str(chat_id), {})
        w_msg = cfg.get("welcome_msg", "Grubumuza hos geldiniz!")
        b_txt = cfg.get("btn_text")
        b_url = cfg.get("btn_url")
        payload = {"chat_id": chat_id, "text": f"Merhaba {new_members[0].get('first_name')}! {w_msg}"}
        if b_txt and b_url:
            payload["reply_markup"] = {"inline_keyboard": [[{"text": b_txt, "url": b_url}]]}
        await tg_api("sendMessage", payload)
        return

    # 1. Gruptan gelen komutlar ve mesajlar
    if chat_type in ["group", "supergroup"]:
        cid = str(chat_id)
        if cid not in active_groups:
            active_groups[cid] = {"title": chat.get("title", "Grup")}
            save_all_storage()

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
            await tg_api("sendMessage", {"chat_id": chat_id, "text": f"Grup hafizama baglandi annecigim: {title}"})
            return

        # Golden Guardian komutlarını yakala
        handled = await handle_golden_guardian_commands(chat_id, text, reply_to)
        if handled:
            return

        if text:
            chat_context_log.append(f"[{chat.get('title')}] {user_name}: {text[:60]}")
        return

    # 2. Özel mesajlar (Bot Chat)
    if chat_type == "private":
        if text == "/start":
            app_url = "https://katya-asistan-bot.onrender.com/live"
            markup = {"inline_keyboard": [[{"text": "🎙️ Katya Canlı Asistan", "web_app": {"url": app_url}}]]}
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": "Merhaba annecigim! Golden Guardian komutlari ve sesli yapay zeka beynim hazir.",
                "reply_markup": markup
            })
            return

        chat_context_log.append(f"[Ozel] {user_name}: {text[:60]}")
        log_event("MSG", f"Ozel Mesaj: {text}")
        reply = await execute_mother_order(text)
        await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})

# ==========================================
# CANLI MINI APP ARAYÜZÜ (SIFIR EKO)
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
      <div class="subtitle">Golden Guardian & Otonom Grup Yonetimi</div>
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
    load_all_storage()
    await refresh_groq_models()

    if BOT_TOKEN:
        poll_task = asyncio.create_task(telegram_long_poll_loop())
        log_event("SYS", "Katya Golden Guardian motoru devrede!")

    yield

    global polling_running
    polling_running = False
    if poll_task:
        poll_task.cancel()
    for t in scheduled_tasks.values():
        t.cancel()

app = FastAPI(lifespan=lifespan)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("<h3>Katya Otonom Agent Sunucusu Aktif.</h3>")

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
