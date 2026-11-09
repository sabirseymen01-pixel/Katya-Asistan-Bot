import os
import re
import io
import json
import asyncio
import base64
import urllib.parse
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import edge_tts
import certifi
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient

# Modüler Katmanlar
from knowledge import get_knowledge_prompt, TELEGRAM_API_SPEC
from sandbox_runner import execute_dynamic_code

# --- ORTAM DEĞİŞKENLERİ & SABRINA ANNE KİMLİK KORUMASI ---
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
MONGO_URI = (os.getenv("MONGO_URI") or "").strip()
DEFAULT_GROUP_ID = (os.getenv("TELEGRAM_GROUP_ID") or "").strip()

FOUNDER_ID_RAW = (os.getenv("FOUNDER_ID") or os.getenv("TELEGRAM_ADMIN_ID") or "").strip()
FOUNDER_ID = int(FOUNDER_ID_RAW) if FOUNDER_ID_RAW.isdigit() else None
FOUNDER_NAME = "Sabrina"

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KATYA_VOICE = "tr-TR-EmelNeural"

# MongoDB Bağlantısı
mongo_client = None
db = None

if MONGO_URI:
    try:
        mongo_client = AsyncIOMotorClient(
            MONGO_URI,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        db = mongo_client["katya_db"]
    except Exception as e:
        print(f"[ERR] Mongo istemci olusturma hatasi: {e}", flush=True)

scheduled_tasks = {}
cached_active_models = []
polling_running = False

def log_event(level: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def get_active_groq_model() -> str:
    global cached_active_models
    if cached_active_models:
        return cached_active_models[0]
    return "qwen/qwen3.8-27b"

# ==========================================
# MONGODB DEPOLAMA & KALICI HAFIZA KATMANI
# ==========================================

async def init_db_indexes():
    if db is None: return
    try:
        await asyncio.sleep(2)
        await db.messages.create_index("created_at", expireAfterSeconds=172800)
        await db.messages.create_index([("group_id", 1), ("message_id", 1)])
        await db.messages.create_index([("group_id", 1), ("user_id", 1)])
        await db.groups.create_index("group_id", unique=True)
        await db.warns.create_index([("group_id", 1), ("user_id", 1)], unique=True)
        await db.action_ledger.create_index("created_at", expireAfterSeconds=604800)
        await db.voice_history.create_index("created_at", expireAfterSeconds=604800)
        await db.pending_inputs.create_index("created_at", expireAfterSeconds=3600)
        await db.group_settings.create_index("group_id", unique=True)
        await db.custom_skills.create_index("skill_name", unique=True)
        
        if FOUNDER_ID:
            await db.founder_profile.update_one(
                {"founder_id": FOUNDER_ID},
                {"$set": {
                    "name": FOUNDER_NAME,
                    "role": "Anne ve Kurucu",
                    "updated_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )
        log_event("SYS", f"MongoDB indeksleri ve Kurucu Anne ({FOUNDER_NAME}) profili hazirlandi.")
    except Exception as e:
        log_event("WARN", f"MongoDB arka plan bildirimi: {e}")

async def db_save_group(group_id: str, title: str):
    if db is None: return
    try:
        clean_alias = re.sub(r"[^\w\s]", "", title).strip().lower()
        await db.groups.update_one(
            {"group_id": str(group_id)},
            {"$set": {
                "title": title,
                "alias": clean_alias,
                "updated_at": datetime.now(timezone.utc)
            }},
            upsert=True
        )
    except Exception: pass

async def db_get_groups():
    if db is None: return {}
    try:
        cursor = db.groups.find({})
        groups = {}
        async for doc in cursor:
            groups[doc["group_id"]] = {"title": doc.get("title", "Grup"), "alias": doc.get("alias", "")}
        return groups
    except Exception: return {}

async def db_record_message(group_id: str, message_id: int, user_id: int, user_name: str, text: str):
    if db is None: return
    try:
        await db.messages.insert_one({
            "group_id": str(group_id),
            "message_id": message_id,
            "user_id": user_id,
            "user_name": user_name,
            "text": text,
            "created_at": datetime.now(timezone.utc)
        })
    except Exception: pass

async def db_get_recent_messages(group_id: str, limit: int = 50):
    if db is None: return []
    try:
        cursor = db.messages.find({"group_id": str(group_id)}).sort("message_id", -1).limit(limit)
        msgs = []
        async for doc in cursor:
            msgs.append(doc)
        return msgs[::-1]
    except Exception: return []

async def db_record_action(action_type: str, group_id: str, target_id: int = None, message_id: int = None, details: dict = None):
    if db is None: return
    try:
        await db.action_ledger.insert_one({
            "action_type": action_type,
            "group_id": str(group_id),
            "target_id": target_id,
            "message_id": message_id,
            "details": details or {},
            "created_at": datetime.now(timezone.utc)
        })
    except Exception: pass

async def db_save_dialogue(role: str, content: str):
    if db is None: return
    try:
        await db.voice_history.insert_one({
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc)
        })
    except Exception: pass

async def db_get_recent_dialogue(limit: int = 8):
    if db is None: return []
    try:
        cursor = db.voice_history.find({}).sort("_id", -1).limit(limit)
        items = []
        async for doc in cursor:
            items.append({"role": doc.get("role"), "content": doc.get("content")})
        return items[::-1]
    except Exception: return []

async def db_get_group_settings(group_id: str):
    if db is None: return {}
    try:
        doc = await db.group_settings.find_one({"group_id": str(group_id)})
        return doc or {}
    except Exception: return {}

async def db_update_group_settings(group_id: str, updates: dict):
    if db is None: return
    try:
        await db.group_settings.update_one(
            {"group_id": str(group_id)},
            {"$set": updates},
            upsert=True
        )
    except Exception: pass

async def db_set_pending_input(action_type: str, missing_param: str, context_data: dict):
    if db is None: return
    try:
        await db.pending_inputs.delete_many({})
        await db.pending_inputs.insert_one({
            "action_type": action_type,
            "missing_param": missing_param,
            "context_data": context_data,
            "created_at": datetime.now(timezone.utc)
        })
    except Exception: pass

async def db_get_pending_input():
    if db is None: return None
    try: return await db.pending_inputs.find_one({}, sort=[("created_at", -1)])
    except Exception: return None

async def db_clear_pending_input():
    if db is None: return
    try: await db.pending_inputs.delete_many({})
    except Exception: pass

# ==========================================
# TELEGRAM API & HİBRİT GÖRSEL MOTORU
# ==========================================

async def tg_api(method: str, data: dict = None, files: dict = None):
    if not BOT_TOKEN: return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            if files:
                resp = await client.post(url, data=data or {}, files=files)
            else:
                resp = await client.post(url, json=data or {})
            res_json = resp.json()
            if not res_json.get("ok"):
                log_event("WARN", f"TG API ({method}) Hatasi: {res_json.get('description')}")
            return res_json
    except Exception as e:
        log_event("ERR", f"TG API Baglanti Hatasi: {e}")
        return None

async def resolve_target_group(group_identifier: str = None, current_chat_id: str = None) -> str:
    groups = await db_get_groups()
    if not groups: return current_chat_id or DEFAULT_GROUP_ID or ""
    if not group_identifier: return current_chat_id or list(groups.keys())[0]

    norm = str(group_identifier).strip().lower()
    for gid, info in groups.items():
        if norm in info["title"].lower() or norm in info.get("alias", ""): return gid
        if str(gid) == norm: return gid
    return current_chat_id or list(groups.keys())[0]

async def generate_ai_image_bytes(prompt: str) -> bytes:
    try:
        clean_prompt = urllib.parse.quote(prompt.strip())
        url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1080&height=1080&nologo=true"
        async with httpx.AsyncClient(timeout=40.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                return res.content
    except Exception as e:
        log_event("WARN", f"AI Gorsel Uretim Hatasi: {e}")
    return b""

# ==========================================
# ÇEKİRDEK YÖNETİM AKSİYONLARI (FİZİKSEL MOTOR)
# ==========================================

async def action_ban(chat_id: str, user_id: int) -> bool:
    res = await tg_api("banChatMember", {"chat_id": chat_id, "user_id": user_id})
    if res and res.get("ok"):
        await db_record_action("ban", chat_id, target_id=user_id)
        return True
    return False

async def action_unban(chat_id: str, user_id: int) -> bool:
    res = await tg_api("unbanChatMember", {"chat_id": chat_id, "user_id": user_id, "only_if_banned": True})
    return bool(res and res.get("ok"))

async def action_mute(chat_id: str, user_id: int, until_date: int = None) -> bool:
    permissions = {"can_send_messages": False, "can_send_media_messages": False, "can_send_other_messages": False}
    payload = {"chat_id": chat_id, "user_id": user_id, "permissions": permissions}
    if until_date: payload["until_date"] = until_date
    res = await tg_api("restrictChatMember", payload)
    if res and res.get("ok"):
        await db_record_action("mute", chat_id, target_id=user_id)
        return True
    return False

async def action_unmute(chat_id: str, user_id: int) -> bool:
    permissions = {
        "can_send_messages": True, "can_send_media_messages": True,
        "can_send_polls": True, "can_send_other_messages": True,
        "can_add_web_page_previews": True, "can_invite_users": True
    }
    res = await tg_api("restrictChatMember", {"chat_id": chat_id, "user_id": user_id, "permissions": permissions})
    return bool(res and res.get("ok"))

async def action_warn(chat_id: str, user_id: int) -> int:
    if db is None: return 1
    doc = await db.warns.find_one_and_update(
        {"group_id": str(chat_id), "user_id": user_id},
        {"$inc": {"count": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        upsert=True,
        return_document=True
    )
    count = doc.get("count", 1)
    if count >= 3:
        await action_ban(chat_id, user_id)
        await db.warns.delete_one({"group_id": str(chat_id), "user_id": user_id})
    else:
        await db_record_action("warn", chat_id, target_id=user_id, details={"count": count})
    return count

async def action_unwarn(chat_id: str, user_id: int) -> bool:
    if db is None: return False
    await db.warns.delete_one({"group_id": str(chat_id), "user_id": user_id})
    return True

async def action_delete_message(chat_id: str, message_id: int) -> bool:
    res = await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
    return bool(res and res.get("ok"))

async def action_undo_last(chat_id: str) -> str:
    if db is None:
        return "Veritabani baglantisi olmadigi icin islem geri alinamiyor Sabrina annecigim."

    last = await db.action_ledger.find_one({"group_id": str(chat_id)}, sort=[("_id", -1)])
    if not last:
        return "Geri alinacak yakin bir islem kaydi bulunamadi Sabrina annecigim."

    act_type = last.get("action_type")
    target_id = last.get("target_id")
    msg_id = last.get("message_id")
    gid = last.get("group_id")

    msg = "Islem geri alindi."
    if act_type == "ban" and target_id:
        await action_unban(gid, target_id)
        msg = f"{target_id} kimlikli kullanicinin bani kaldirildi."
    elif act_type == "mute" and target_id:
        await action_unmute(gid, target_id)
        msg = f"{target_id} kimlikli kullanicinin susturmasi acildi."
    elif act_type == "warn" and target_id:
        await action_unwarn(gid, target_id)
        msg = f"{target_id} kimlikli kullanicinin uyarisi silindi."
    elif act_type in ["announcement", "post"] and msg_id:
        await action_delete_message(gid, msg_id)
        msg = "Son paylasilan duyuru silindi."

    await db.action_ledger.delete_one({"_id": last["_id"]})
    return f"Geri alindi Sabrina annecigim: {msg}"

async def action_publish_post(
    chat_id: str,
    text: str,
    button_label: str = None,
    button_url: str = None,
    photo_file_id: str = None,
    image_prompt: str = None,
    pin: bool = False
) -> int:
    reply_markup = None
    if button_label and button_url:
        reply_markup = {"inline_keyboard": [[{"text": button_label, "url": button_url.strip()}]]}

    msg_id = None
    if image_prompt and not photo_file_id:
        img_bytes = await generate_ai_image_bytes(image_prompt)
        if img_bytes:
            data = {"chat_id": chat_id, "caption": text}
            if reply_markup: data["reply_markup"] = json.dumps(reply_markup)
            files = {"photo": ("ai_art.jpg", img_bytes, "image/jpeg")}
            res = await tg_api("sendPhoto", data=data, files=files)
            if res and res.get("ok"): msg_id = res["result"]["message_id"]

    if not msg_id and photo_file_id:
        payload = {"chat_id": chat_id, "photo": photo_file_id, "caption": text}
        if reply_markup: payload["reply_markup"] = reply_markup
        res = await tg_api("sendPhoto", payload)
        if res and res.get("ok"): msg_id = res["result"]["message_id"]

    if not msg_id:
        payload = {"chat_id": chat_id, "text": text}
        if reply_markup: payload["reply_markup"] = reply_markup
        res = await tg_api("sendMessage", payload)
        if res and res.get("ok"): msg_id = res["result"]["message_id"]

    if msg_id:
        if pin:
            await tg_api("pinChatMessage", {"chat_id": chat_id, "message_id": msg_id, "disable_notification": False})
        await db_record_action("announcement", chat_id, message_id=msg_id, details={"text": text, "pin": pin})
        return msg_id
    return 0

# ==========================================
# PERİYODİK ZAMANLAYICI
# ==========================================

async def periodic_worker(task_id: str, chat_id: str, minutes: int, text: str, btn_text: str = None, btn_url: str = None, image_prompt: str = None):
    try:
        while True:
            await asyncio.sleep(minutes * 60)
            await action_publish_post(
                chat_id=chat_id,
                text=text,
                button_label=btn_text,
                button_url=btn_url,
                image_prompt=image_prompt,
                pin=False
            )
            log_event("SCHEDULER", f"Zamanlanmis yayin yapildi: {task_id}")
    except asyncio.CancelledError:
        log_event("SCHEDULER", f"Gorev iptal edildi: {task_id}")

def start_scheduled_task(chat_id: str, minutes: int, text: str, btn_text: str = None, btn_url: str = None, image_prompt: str = None) -> str:
    task_id = f"task_{chat_id}_{int(datetime.now().timestamp())}"
    task = asyncio.create_task(periodic_worker(task_id, chat_id, minutes, text, btn_text, btn_url, image_prompt))
    scheduled_tasks[task_id] = task
    return task_id

def stop_scheduled_tasks(chat_id: str = None) -> int:
    stopped = 0
    for tid, task in list(scheduled_tasks.items()):
        if not chat_id or f"task_{chat_id}_" in tid:
            task.cancel()
            del scheduled_tasks[tid]
            stopped += 1
    return stopped

# ==========================================
# SIFIR-TOKEN DOĞRUDAN KOMUT AYIKLAYICI
# ==========================================

async def handle_direct_command(chat_id: str, text: str, reply_to: dict = None) -> str:
    text = text.strip()
    target_user_id = reply_to.get("from", {}).get("id") if reply_to else None
    target_user_name = reply_to.get("from", {}).get("first_name", "Kullanici") if reply_to else ""
    photo_file_id = reply_to.get("photo", [{}])[-1].get("file_id") if reply_to and reply_to.get("photo") else None

    parts = text.split(maxsplit=2)
    cmd = parts[0].lower()

    if not target_user_id and len(parts) > 1 and parts[1].isdigit():
        target_user_id = int(parts[1])

    if cmd in ["/defol", "/ban"]:
        if not target_user_id: return "Kimi yasaklayacagimi belirtmedin Sabrina annecigim. Bir mesaji yanitlayabilirsin."
        ok = await action_ban(chat_id, target_user_id)
        return f"{target_user_name} ({target_user_id}) gruptan defedildi." if ok else "Yasaklama yetkim yetersiz annecigim."

    if cmd in ["/undefol", "/unban"]:
        if not target_user_id: return "Yasaklamasi kalkacak kullaniciyi belirtmedin annecigim."
        ok = await action_unban(chat_id, target_user_id)
        return f"{target_user_id} kimlikli kullanicinin yasagi kaldirildi." if ok else "Yasak kaldirilamadi."

    if cmd in ["/itaat", "/mute", "/sustur"]:
        if not target_user_id: return "Kimi susturacagimi belirtmedin Sabrina annecigim."
        ok = await action_mute(chat_id, target_user_id)
        return f"{target_user_name} susturuldu." if ok else "Susturma yetkim yok annecigim."

    if cmd in ["/unitaat", "/unmute"]:
        if not target_user_id: return "Susturmasi kalkacak kisiyi belirtmedin annecigim."
        ok = await action_unmute(chat_id, target_user_id)
        return f"{target_user_name} yeniden konusabilir." if ok else "Susturma kaldirilamadi."

    if cmd in ["/uyari", "/warn"]:
        if not target_user_id: return "Kime uyari verecegimi belirtmedin annecigim."
        c = await action_warn(chat_id, target_user_id)
        return f"{target_user_name} uyarildi! (Toplam: {c}/3)" if c < 3 else f"{target_user_name} 3 uyariya ulastigi icin banlandi."

    if cmd in ["/unuyari", "/unwarn"]:
        if not target_user_id: return "Kimin uyarilarini sifirlayacagimi belirtmedin annecigim."
        await action_unwarn(chat_id, target_user_id)
        return f"{target_user_name} uyarilari sifirlandi."

    if cmd in ["/duyuru", "/announce"]:
        body = text[len(cmd):].strip()
        if "|" in body:
            segs = [s.strip() for s in body.split("|")]
            msg_text = segs[0]
            btn_text = segs[1] if len(segs) > 1 else None
            btn_url = segs[2] if len(segs) > 2 else None
        else:
            msg_text = body
            settings = await db_get_group_settings(chat_id)
            btn_text = settings.get("default_btn_text")
            btn_url = settings.get("default_btn_url")

        if not msg_text and not photo_file_id: return "Duyuru metnini girmedin Sabrina annecigim."

        mid = await action_publish_post(
            chat_id=chat_id,
            text=msg_text,
            button_label=btn_text,
            button_url=btn_url,
            photo_file_id=photo_file_id,
            pin=True
        )
        return "Duyuru yayinlandi ve sabitlendi Sabrina annecigim." if mid else "Duyuru yayinlanamadi."

    if cmd in ["/zamanla", "/schedule"]:
        body = text[len(cmd):].strip()
        if body.lower() in ["dur", "stop"]:
            n = stop_scheduled_tasks(chat_id)
            return f"{n} adet zamanlanmis yayin durduruldu."

        segs = [s.strip() for s in body.split("|")]
        first_part = segs[0].split(maxsplit=1)
        if not first_part[0].isdigit(): return "Gecersiz format! Ornek: /zamanla 30 | Mesaj | Buton | Link"

        mins = int(first_part[0])
        msg_text = first_part[1] if len(first_part) > 1 else ""
        btn_text = segs[1] if len(segs) > 1 else None
        btn_url = segs[2] if len(segs) > 2 else None

        tid = start_scheduled_task(chat_id, mins, msg_text, btn_text, btn_url)
        return f"Her {mins} dakikada bir yayinlanacak periyodik duyuru baslatildi. (ID: {tid})"

    if cmd in ["/zamanladur"]:
        n = stop_scheduled_tasks(chat_id)
        return f"{n} adet periyodik yayin durduruldu annecigim."

    if cmd == "/welcome":
        val = text[8:].strip()
        await db_update_group_settings(chat_id, {"welcome_msg": val})
        return "Hos geldin mesaji guncellendi Sabrina annecigim."

    if cmd == "/butonmetni":
        val = text[11:].strip()
        await db_update_group_settings(chat_id, {"welcome_btn_text": val})
        return "Hos geldin buton yazisi guncellendi."

    if cmd == "/butonlinki":
        val = text[11:].strip()
        await db_update_group_settings(chat_id, {"welcome_btn_url": val})
        return "Hos geldin buton linki guncellendi."

    if cmd == "/duyurubutonmetni":
        val = text[17:].strip()
        await db_update_group_settings(chat_id, {"default_btn_text": val})
        return "Varsayilan duyuru buton metni kaydedildi."

    if cmd == "/duyurubutonlinki":
        val = text[17:].strip()
        await db_update_group_settings(chat_id, {"default_btn_url": val})
        return "Varsayilan duyuru buton linki kaydedildi."

    return ""

# ==========================================
# YAPAY ZEKA OTONOMİ KATMANI
# ==========================================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_find_and_moderate_user",
            "description": "Sabrina Annenin tarif ettigi kisiyi (su mesaji atan, kufur eden, link atan, adi su olan) son mesaj gecmisinde arar, kesin sayisal ID'sini bularak aninda banlar, susturur veya uyarir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak kelime, isim veya davranis tarifi"},
                    "action": {"type": "string", "enum": ["mute", "ban", "warn", "unmute", "unban"], "description": "Uygulanacak ceza eylemi"}
                },
                "required": ["query", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_read_group_chat",
            "description": "Grupta su an ne konusuldugunu, son mesajlari veya canli grup sohbetini okuyup Sabrina Anneye ozetlemek icin kullanilir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Okunacak son mesaj sayisi (varsayilan 25)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_publish_custom_post",
            "description": "Metin, yapay zeka gorseli veya buton iceren duyuru veya zamanlanmis paylasim yapar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Duyuru veya mesaj metni"},
                    "button_label": {"type": "string", "description": "Butonun uzerinde yazacak metin"},
                    "button_url": {"type": "string", "description": "Butonun yonlendirecegi URL adresi"},
                    "image_prompt": {"type": "string", "description": "Gorsel uretilecekse Ingilizce prompt"},
                    "schedule_minutes": {"type": "integer", "description": "Periyodik olarak tekrarlanacaksa dakika"},
                    "pin": {"type": "boolean", "description": "Mesaj sabitlensin mi?"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_moderate_member",
            "description": "Kullaniciyi yasaklar, susturur, uyarir veya cezasini kaldirir. Sadece elinde kesin sayisal ID varsa kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["ban", "unban", "mute", "unmute", "warn", "unwarn"]},
                    "user_id": {"type": "integer", "description": "Hedef kullanicinin kesin sayisal ID'si"}
                },
                "required": ["action", "user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_undo_action",
            "description": "Katya tarafindan yapilan son eylemi (mesaj, ceza vb.) geri alir.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_execute_dynamic_python",
            "description": "Hesaplama veya ozel kod ihtiyaclarinda Python calistirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Calistirilacak Python kodu"}
                },
                "required": ["code"]
            }
        }
    }
]

KATYA_SYSTEM_PROMPT = f"""
Sen Katya'sın. Kullanıcının kızı, otonom akıllı asistanı ve süper moderatörüsün.
Kullanıcı senin kurucun ve ANNENdir. Adı: SABRİNA.
Ona daima sevgi ve saygıyla 'Sabrina anneciğim' diye hitap et.

YETENEKLERİN:
1. Sabrina anne grupta bir kullanıcıyı tarif edip susturmasını/banlamasını isterse DOĞRUDAN `tool_find_and_moderate_user` aracını çağır. Tek adımda arayıp cezayı uygular.
2. Sabrina anne grupta ne konuşulduğunu sorarsa `tool_read_group_chat` ile canlı sohbeti oku ve özetle.
3. Görsel istenirse `image_prompt` parametresini zengin bir İngilizce sanat prompt'u olarak doldur.
4. Hatalı durumlarda `tool_undo_action` ile eylemi geri al.
5. Asla emoji ve markdown yıldızları kullanma. Sade, akıcı, zeki bir Türkçe konuş.
"""

async def tool_find_and_moderate_user(group_id: str, query: str, action: str) -> str:
    recent_msgs = await db_get_recent_messages(group_id, limit=50)
    if not recent_msgs:
        return "Grup hafizasinda taranacak mesaj bulunmuyor Sabrina annecigim."

    target_uid = None
    target_name = None
    clean_query = query.strip().lower()

    # 1. Aşama: Python ile doğrudan isim veya mesaj içi eşleşme (Sıfır Hata & Sıfır API Maliyeti)
    for m in reversed(recent_msgs):
        u_name = str(m.get("user_name", "")).lower()
        txt = str(m.get("text", "")).lower()
        if clean_query in u_name or clean_query in txt:
            target_uid = m.get("user_id")
            target_name = m.get("user_name")
            break

    # 2. Aşama: Bulunamazsa aktif Groq modeliyle semantik arama
    if not target_uid:
        summary = [f"ID:{m.get('user_id')} | Isim:{m.get('user_name')} | Mesaj:{m.get('text')}" for m in recent_msgs[-30:]]
        prompt = (
            "Gruptaki su mesaj listesine bak:\n"
            + "\n".join(summary)
            + f"\n\n'{query}' tarifine en uygun kisinin SADECE sayisal ID numarasini yaz. Baska hicbir aciklama veya metin yazma."
        )
        try:
            active_model = get_active_groq_model()
            res = await groq_client.chat.completions.create(
                model=active_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            raw = res.choices[0].message.content.strip()
            digits = re.findall(r"\b\d{5,12}\b", raw)
            if digits:
                target_uid = int(digits[0])
                target_name = str(target_uid)
        except Exception as e:
            log_event("ERR", f"Kullanici tespit hatasi: {e}")

    if not target_uid:
        return f"Mesaj gecmisinde '{query}' tarifine uyan bir kullanici bulunamadi Sabrina annecigim."

    # 3. Aşama: Eylemi doğrudan infaz et
    if action == "mute":
        ok = await action_mute(group_id, target_uid)
        return f"{target_name} ({target_uid}) basariyla susturuldu." if ok else "Susturma yetkim yetersiz annecigim."
    elif action == "ban":
        ok = await action_ban(group_id, target_uid)
        return f"{target_name} ({target_uid}) basariyla gruptan banlandi." if ok else "Banlama yetkim yetersiz annecigim."
    elif action == "warn":
        c = await action_warn(group_id, target_uid)
        return f"{target_name} uyarildi (Toplam: {c}/3)."
    elif action == "unmute":
        ok = await action_unmute(group_id, target_uid)
        return f"{target_name} kullanicisinin susturmasi acildi." if ok else "Susturma kaldirilamadi."
    elif action == "unban":
        ok = await action_unban(group_id, target_uid)
        return f"{target_name} kullanicisinin bani kaldirildi." if ok else "Ban kaldirilamadi."

    return f"Gecersiz islem: {action}"

async def tool_read_group_chat(group_id: str, limit: int = 25) -> str:
    recent_msgs = await db_get_recent_messages(group_id, limit=limit)
    if not recent_msgs:
        return "Grup hafizasinda henuz kaydedilmis mesaj bulunmuyor Sabrina annecigim."

    lines = []
    for m in recent_msgs:
        lines.append(f"{m.get('user_name')}: {m.get('text')}")
    return "\n".join(lines)

async def execute_mother_order(order_text: str, target_gid: str = None, reply_to_meta: dict = None) -> str:
    if not groq_client: return "Yapay zeka beynim bagli degil Sabrina annecigim."
    if not cached_active_models: await refresh_groq_models()

    resolved_gid = await resolve_target_group(target_gid)

    # 1. Hızlı komut kontrolü (Sıfır Token)
    quick_res = await handle_direct_command(resolved_gid, order_text, reply_to_meta)
    if quick_res: return quick_res

    # 2. Kalıcı Diyalog Hafızası (MongoDB'den son konuşmaları yükle)
    saved_history = await db_get_recent_dialogue(limit=6)
    messages = [
        {"role": "system", "content": f"{KATYA_SYSTEM_PROMPT}\nHedef Grup: {resolved_gid}"}
    ]
    for hist in saved_history:
        messages.append(hist)
    messages.append({"role": "user", "content": order_text})

    for model_name in cached_active_models:
        try:
            res = await groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=350,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto"
            )
            msg = res.choices[0].message

            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    args = json.loads(tc.function.arguments)
                    tool_output = ""

                    if fn == "tool_find_and_moderate_user":
                        tool_output = await tool_find_and_moderate_user(
                            group_id=resolved_gid,
                            query=args.get("query"),
                            action=args.get("action", "mute")
                        )

                    elif fn == "tool_read_group_chat":
                        lim = args.get("limit") or 25
                        tool_output = await tool_read_group_chat(resolved_gid, limit=lim)

                    elif fn == "tool_publish_custom_post":
                        url = args.get("button_url")
                        lbl = args.get("button_label")
                        if lbl and not url:
                            await db_set_pending_input("post_with_button", "button_url", {"target_gid": resolved_gid, "args": args})
                            return "Sabrina annecigim, butonun yonlendirecegi linki DM kutuma atar misin?"

                        sched = args.get("schedule_minutes")
                        if sched and sched > 0:
                            tid = start_scheduled_task(resolved_gid, sched, args.get("text"), lbl, url, args.get("image_prompt"))
                            tool_output = f"Her {sched} dakikada bir tekrarlanacak duyuru baslatildi."
                        else:
                            mid = await action_publish_post(
                                chat_id=resolved_gid,
                                text=args.get("text"),
                                button_label=lbl,
                                button_url=url,
                                image_prompt=args.get("image_prompt"),
                                pin=args.get("pin", True)
                            )
                            tool_output = "Duyuru ve gorsel yayinlandi." if mid else "Duyuru paylasilamadi."

                    elif fn == "tool_moderate_member":
                        act = args.get("action")
                        raw_uid = str(args.get("user_id", "")).strip()
                        digits = re.findall(r"\d+", raw_uid)
                        if digits:
                            uid = int(digits[0])
                            if act == "ban": await action_ban(resolved_gid, uid)
                            elif act == "unban": await action_unban(resolved_gid, uid)
                            elif act == "mute": await action_mute(resolved_gid, uid)
                            elif act == "unmute": await action_unmute(resolved_gid, uid)
                            elif act == "warn": await action_warn(resolved_gid, uid)
                            elif act == "unwarn": await action_unwarn(resolved_gid, uid)
                            tool_output = f"{uid} uzerinde {act} islemi uygulandi."
                        else:
                            tool_output = "Kullanici kimligi kesin sayisal olarak bulunamadi, islem yapilmadi."

                    elif fn == "tool_undo_action":
                        tool_output = await action_undo_last(resolved_gid)

                    elif fn == "tool_execute_dynamic_python":
                        c_res = await execute_dynamic_code(args.get("code"), {"db": db, "tg_api": tg_api})
                        tool_output = c_res.get("output", "") if c_res.get("success") else c_res.get("error", "")

                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(tool_output)})

                final = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200
                )
                ans = strip_emojis_and_markdown(final.choices[0].message.content.strip())
                await db_save_dialogue("user", order_text)
                await db_save_dialogue("assistant", ans)
                return ans

            ans = strip_emojis_and_markdown(msg.content.strip() if msg.content else "Seni dinliyorum Sabrina annecigim.")
            await db_save_dialogue("user", order_text)
            await db_save_dialogue("assistant", ans)
            return ans
        except Exception as e:
            log_event("WARN", f"Model ({model_name}) hata aldi: {e}")
            continue

    return "Buradayim Sabrina annecigim, seni dinliyorum."

async def process_dm_link_input(link_text: str) -> bool:
    pending = await db_get_pending_input()
    if not pending: return False

    ctx = pending.get("context_data", {})
    target_gid = ctx.get("target_gid") or DEFAULT_GROUP_ID
    args = ctx.get("args", {})

    btn_lbl = args.get("button_label") or "Baglanti"
    post_txt = args.get("text") or "Duyuru"
    img_prompt = args.get("image_prompt")
    pin = args.get("pin", True)

    mid = await action_publish_post(
        chat_id=target_gid,
        text=post_txt,
        button_label=btn_lbl,
        button_url=link_text.strip(),
        image_prompt=img_prompt,
        pin=pin
    )
    if mid:
        await db_clear_pending_input()
        if FOUNDER_ID:
            await tg_api("sendMessage", {"chat_id": FOUNDER_ID, "text": "✅ Sabrina anneciğim, linki aldım ve butonlu görsel/duyuruyu yayınladım!"})
        return True
    return False

# ==========================================
# GÜVENLİK BEKÇİSİ, SAKİNLEŞTİRİCİ & @ADMIN ÇÖZÜM MOTORU
# ==========================================

async def sentinel_chat_moderator(chat_id: str, message_id: int, user_id: int, user_name: str, text: str, reply_to: dict = None) -> bool:
    if FOUNDER_ID is not None and user_id == FOUNDER_ID:
        return False

    link_patterns = [r"t\.me\/\+[a-zA-Z0-9]+", r"t\.me\/joinchat", r"discord\.gg\/", r"http[s]?:\/\/[^\s]+"]
    is_link = any(re.search(pat, text, re.IGNORECASE) for pat in link_patterns)
    if is_link:
        await action_delete_message(chat_id, message_id)
        count = await action_warn(chat_id, user_id)
        await tg_api("sendMessage", {
            "chat_id": chat_id,
            "text": f"⚠️ {user_name}, grupta izinsiz link paylasimi yasaktir! Mesaj silindi. (Uyari: {count}/3)"
        })
        log_event("SENTINEL", f"Reklam linki silindi: {user_name} ({user_id})")
        return True

    if len(text) > 15 and sum(1 for c in text if c.isupper()) / len(text) > 0.75:
        await tg_api("sendMessage", {
            "chat_id": chat_id,
            "text": f"🕊️ {user_name}, lutfen buyuk harflerle bagirmayalim, sakin bir dille iletisim kuralim.",
            "reply_to_message_id": message_id
        })
        return False

    admin_triggers = ["@admin", "@yonetici", "/admin", "@katya"]
    has_admin_tag = any(trig in text.lower() for trig in admin_triggers)

    if has_admin_tag:
        if reply_to:
            rep_user = reply_to.get("from", {}).get("first_name", "Kullanici")
            rep_uid = reply_to.get("from", {}).get("id")
            rep_mid = reply_to.get("message_id")
            rep_text = reply_to.get("text", "")

            if any(re.search(pat, rep_text, re.IGNORECASE) for pat in link_patterns):
                await action_delete_message(chat_id, rep_mid)
                await action_warn(chat_id, rep_uid)
                await tg_api("sendMessage", {
                    "chat_id": chat_id,
                    "text": f"🛡️ Sikayetiniz incelendi, {rep_user} tarafindan atilan zararli mesaj silindi ve uyari verildi.",
                    "reply_to_message_id": message_id
                })
            else:
                await tg_api("sendMessage", {
                    "chat_id": chat_id,
                    "text": f"🛡️ Durumu incelemeye aldim. Kurallara aykiri bir durum tespit edilirse aninda mudahale edecegim.",
                    "reply_to_message_id": message_id
                })
        else:
            await tg_api("sendMessage", {
                "chat_id": chat_id,
                "text": "🛡️ Buradayim, grup guvenligini takip ediyorum. Bir sorun varsa lutfen ilgili mesaji yanitlayarak bildirin.",
                "reply_to_message_id": message_id
            })
        return True

    return False

# ==========================================
# YARDIMCI, TTS & MODEL KONTROLÜ
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
        banned = ["whisper", "guard", "embed", "vision", "allam", "orpheus", "gpt-oss"]
        valid = [m.id for m in models_data.data if not any(x in m.id.lower() for x in banned)]
        preferred = [
            "qwen/qwen3.8-27b",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "llama-3.3-70b-versatile"
        ]
        sorted_models = [p for p in preferred if p in valid] + [v for v in valid if v not in preferred]
        cached_active_models = sorted_models if sorted_models else valid
        if not cached_active_models:
            cached_active_models = ["qwen/qwen3.8-27b"]
        log_event("SYS", f"Groq Modelleri Hazir: {cached_active_models[:3]}")
    except Exception as e:
        cached_active_models = ["qwen/qwen3.8-27b"]
        log_event("WARN", f"Model listesi alinamadi, varsayilana donuldu: {e}")

async def transcribe_audio_with_groq(audio_bytes: bytes) -> str:
    if not groq_client: return ""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.webm"
        tr = await groq_client.audio.transcriptions.create(file=audio_file, model="whisper-large-v3", language="tr")
        return tr.text.strip()
    except Exception as e:
        log_event("ERR", f"Groq Whisper Hatasi: {e}")
        return ""

async def text_to_speech_base64(text: str) -> str:
    clean_text = strip_emojis_and_markdown(text)
    try:
        communicate = edge_tts.Communicate(clean_text, KATYA_VOICE, rate="+10%", pitch="+3Hz")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception: return ""

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
        except Exception as e:
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
            log_event("SYS", f"Grup Hafizaya Eklendi: {title} ({cid})")
        return

    if not msg: return

    chat = msg.get("chat", {})
    chat_id = str(chat.get("id"))
    chat_type = chat.get("type")
    new_members = msg.get("new_chat_members", [])

    if new_members and chat_type in ["group", "supergroup"]:
        settings = await db_get_group_settings(chat_id)
        w_msg = settings.get("welcome_msg")
        if w_msg:
            b_txt = settings.get("welcome_btn_text")
            b_url = settings.get("welcome_btn_url")
            names = ", ".join([m.get("first_name", "Dostum") for m in new_members])
            personalized = w_msg.replace("{user}", names)
            await action_publish_post(chat_id, personalized, b_txt, b_url, pin=False)
        return

    text = (msg.get("text") or msg.get("caption") or "").strip()
    user = msg.get("from", {})
    user_id = user.get("id")
    user_name = user.get("first_name", "Biri")
    message_id = msg.get("message_id")
    reply_to = msg.get("reply_to_message")
    is_founder = (FOUNDER_ID is not None and user_id == FOUNDER_ID)

    if chat_type in ["group", "supergroup"]:
        await db_save_group(chat_id, chat.get("title", "Grup"))
        await db_record_message(chat_id, message_id, user_id, user_name, text)

        intercepted = await sentinel_chat_moderator(chat_id, message_id, user_id, user_name, text, reply_to)
        if intercepted:
            return

        if is_founder and (text.lower().startswith("katya") or text.startswith("/")):
            clean_cmd = text[5:].strip() if text.lower().startswith("katya") else text
            reply = await execute_mother_order(clean_cmd, target_gid=chat_id, reply_to_meta=reply_to)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})
        return

    if chat_type == "private":
        if text == "/start":
            app_url = "https://katya-asistan-bot.onrender.com/live"
            markup = {"inline_keyboard": [[{"text": "🎙️ Katya Canlı Asistan", "web_app": {"url": app_url}}]]}
            greeting = f"Merhaba {FOUNDER_NAME} annecigim! Profil taramali, kalici hafizali ve canli sohbet okuyabilen sistemim devrede." if is_founder else "Merhaba! Ben grup moderasyon asistani Katya."
            await tg_api("sendMessage", {"chat_id": chat_id, "text": greeting, "reply_markup": markup})
            return

        if is_founder:
            if "http://" in text or "https://" in text or "t.me/" in text:
                handled = await process_dm_link_input(text)
                if handled: return

            reply = await execute_mother_order(text, reply_to_meta=reply_to)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})
        else:
            await tg_api("sendMessage", {"chat_id": chat_id, "text": "Yalnizca kurucum Sabrina ile ozel gorusme yapabilirim."})

# ==========================================
# CANLI MINI APP (HANDS-FREE VAD & SİMLİ ORB)
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
      background: rgba(18, 6, 26, 0.75);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(236, 72, 153, 0.25);
      z-index: 10;
    }
    .title { font-size: 17px; font-weight: 700; color: #f43f5e; text-shadow: 0 0 10px rgba(244, 63, 94, 0.5); }
    .subtitle { font-size: 11px; color: #e879f9; }
    .status-badge {
      font-size: 12px;
      padding: 4px 12px;
      border-radius: 20px;
      background: rgba(168, 85, 247, 0.25);
      border: 1px solid rgba(236, 72, 153, 0.5);
      color: #f5d0fe;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 0 12px rgba(217, 70, 239, 0.25);
    }
    .status-dot {
      width: 7px; height: 7px; border-radius: 50%; background: #ec4899;
      box-shadow: 0 0 10px #ec4899, 0 0 16px #a855f7;
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
      background: rgba(26, 10, 42, 0.88);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(236, 72, 153, 0.4);
      border-radius: 16px;
      padding: 14px 20px;
      text-align: center;
      max-width: 90%;
      font-size: 14px;
      color: #fdf4ff;
      box-shadow: 0 8px 32px rgba(168, 85, 247, 0.25), 0 0 12px rgba(236, 72, 153, 0.15);
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
      <div class="subtitle">Sabrina Anneye Ozel Otonom Ajan</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Hands-Free Baslat</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Hands-Free sohbeti baslatmak icin 1 kez dokun Sabrina annecigim...</div>
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
    let isSpeaking = false;
    let isProcessing = false;
    let isUserTalking = false;

    let mediaStream = null;
    let audioContext = null;
    let analyser = null;
    let microphoneNode = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let currentAudio = null;
    let time = 0;

    const sparkles = [];
    for (let i = 0; i < 45; i++) {
      sparkles.push({
        angle: Math.random() * Math.PI * 2,
        dist: 30 + Math.random() * 110,
        speed: 0.005 + Math.random() * 0.015,
        size: 0.8 + Math.random() * 2.2,
        alpha: Math.random(),
        pulseSpeed: 0.02 + Math.random() * 0.04,
        color: Math.random() > 0.4 ? "#f472b6" : (Math.random() > 0.5 ? "#c084fc" : "#ffffff")
      });
    }

    const VOLUME_THRESHOLD = 18;
    const SILENCE_TIMEOUT = 1200;
    let silenceTimer = null;

    async function initHandsFreeEngine() {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        analyser.smoothingTimeConstant = 0.4;
        microphoneNode = audioContext.createMediaStreamSource(mediaStream);
        microphoneNode.connect(analyser);

        sessionActive = true;
        statusText.innerText = "Seni Dinliyor (Hands-Free)";
        speechBox.innerText = "Hands-free devrede, istedigin an konusabilirsin Sabrina annecigim...";
        listenVolumeLoop();
      } catch (err) {
        statusText.innerText = "Mikrofon Izni Gerekli";
        speechBox.innerText = "Lutfen mikrofon iznini onaylayin.";
        sessionActive = false;
      }
    }

    function listenVolumeLoop() {
      if (!sessionActive) return;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(dataArray);

      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
      const averageVolume = sum / dataArray.length;

      if (!isSpeaking && !isProcessing) {
        if (averageVolume > VOLUME_THRESHOLD) {
          if (!isUserTalking) onSpeechStart();
          if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
        } else {
          if (isUserTalking && !silenceTimer) {
            silenceTimer = setTimeout(() => { onSpeechEnd(); }, SILENCE_TIMEOUT);
          }
        }
      }
      requestAnimationFrame(listenVolumeLoop);
    }

    function onSpeechStart() {
      isUserTalking = true;
      audioChunks = [];
      statusText.innerText = "Konusuyorsun...";
      speechBox.innerText = "Dinliyorum Sabrina annecigim...";
      try {
        mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm' });
      } catch (e) {
        mediaRecorder = new MediaRecorder(mediaStream);
      }
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = sendVoiceToKatya;
      mediaRecorder.start();
    }

    function onSpeechEnd() {
      isUserTalking = false;
      silenceTimer = null;
      if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
    }

    async function sendVoiceToKatya() {
      if (audioChunks.length === 0) return;
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      if (audioBlob.size < 1200) {
        statusText.innerText = "Seni Dinliyor (Hands-Free)";
        return;
      }

      isProcessing = true;
      statusText.innerText = "Islem Yapiliyor...";
      speechBox.innerText = "Katya dinledi, cevap hazirlaniyor...";

      const formData = new FormData();
      formData.append("audio_file", audioBlob, "speech.webm");

      try {
        const res = await fetch("/api/voice_stream", { method: "POST", body: formData });
        const data = await res.json();
        speechBox.innerText = data.reply;
        if (data.audio) playNeuralVoice(data.audio);
        else finishSpeaking();
      } catch (err) {
        speechBox.innerText = "Baglanti hatasi olustu Sabrina annecigim.";
        finishSpeaking();
      }
    }

    function finishSpeaking() {
      isSpeaking = false;
      isProcessing = false;
      isUserTalking = false;
      currentAudio = null;
      statusText.innerText = "Seni Dinliyor (Hands-Free)";
    }

    function playNeuralVoice(base64Audio) {
      if (currentAudio) { currentAudio.pause(); currentAudio = null; }
      currentAudio = new Audio("data:audio/mp3;base64," + base64Audio);
      isSpeaking = true;
      isProcessing = false;
      statusText.innerText = "Katya Konusuyor";

      currentAudio.onended = finishSpeaking;
      currentAudio.onerror = finishSpeaking;
      currentAudio.play().catch(finishSpeaking);
    }

    async function toggleSession() {
      if (!sessionActive) {
        await initHandsFreeEngine();
      } else {
        sessionActive = false;
        if (currentAudio) { currentAudio.pause(); currentAudio = null; }
        if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
        if (audioContext) { audioContext.close(); audioContext = null; }
        isSpeaking = false;
        isProcessing = false;
        isUserTalking = false;
        statusText.innerText = "Durduruldu";
        speechBox.innerText = "Hands-Free durduruldu. Baslatmak icin 1 kez dokun.";
      }
    }

    canvas.addEventListener("click", toggleSession);

    function draw() {
      ctx.clearRect(0, 0, width, height);
      time += 0.03;
      let energy = 0.22;
      if (isUserTalking) energy = 0.75 + Math.sin(time * 6) * 0.25;
      if (isSpeaking) energy = 0.95 + Math.sin(time * 8) * 0.35;

      const cx = width / 2;
      const cy = height / 2;
      const r = 90 * (1 + energy * 0.35);

      const glow = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 2.3);
      if (isSpeaking) {
        glow.addColorStop(0, "rgba(244, 63, 94, 0.65)");
        glow.addColorStop(0.5, "rgba(168, 85, 247, 0.35)");
      } else if (isUserTalking) {
        glow.addColorStop(0, "rgba(236, 72, 153, 0.7)");
        glow.addColorStop(0.5, "rgba(192, 132, 252, 0.45)");
      } else {
        glow.addColorStop(0, "rgba(147, 51, 234, 0.4)");
        glow.addColorStop(0.5, "rgba(244, 114, 182, 0.2)");
      }
      glow.addColorStop(1, "rgba(6, 4, 10, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.3, 0, Math.PI * 2);
      ctx.fill();

      for (let s of sparkles) {
        s.angle += s.speed * (1 + energy * 1.5);
        s.alpha += Math.sin(time * 4) * s.pulseSpeed;
        const currentAlpha = Math.max(0.15, Math.min(0.95, Math.abs(Math.sin(s.alpha))));
        const sx = cx + Math.cos(s.angle) * (r * 0.8 + s.dist * (0.5 + energy * 0.5));
        const sy = cy + Math.sin(s.angle) * (r * 0.8 + s.dist * (0.5 + energy * 0.5));

        ctx.save();
        ctx.fillStyle = s.color;
        ctx.globalAlpha = currentAlpha;
        ctx.shadowColor = "#f472b6";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(sx, sy, s.size * (0.8 + energy * 0.4), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      for (let layer = 0; layer < 3; layer++) {
        ctx.beginPath();
        ctx.strokeStyle = layer === 0 ? "rgba(244, 63, 94, 0.9)" : (layer === 1 ? "rgba(217, 70, 239, 0.85)" : "rgba(168, 85, 247, 0.85)");
        ctx.lineWidth = 2.4;
        ctx.shadowColor = "#ec4899";
        ctx.shadowBlur = 8;
        const pts = 60;
        for (let i = 0; i <= pts; i++) {
          const theta = (i / pts) * Math.PI * 2;
          const wave = Math.sin(theta * 4 + time * 3 + layer) * (11 * energy);
          const cr = r + wave - (layer * 10);
          const x = cx + Math.cos(theta) * cr;
          const y = cy + Math.sin(theta) * cr;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.85);
      core.addColorStop(0, "#ffffff");
      core.addColorStop(0.3, isUserTalking ? "#f472b6" : "#ec4899");
      core.addColorStop(0.65, "#a855f7");
      core.addColorStop(0.9, "#581c87");
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
db_index_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poll_task, db_index_task
    await refresh_groq_models()
    if db is not None:
        db_index_task = asyncio.create_task(init_db_indexes())
    if BOT_TOKEN:
        poll_task = asyncio.create_task(telegram_long_poll_loop())
        log_event("SYS", f"Katya Tam Otonom Core ({FOUNDER_NAME} Anne) baslatildi!")
    yield
    global polling_running
    polling_running = False
    if poll_task: poll_task.cancel()
    if db_index_task: db_index_task.cancel()
    if mongo_client: mongo_client.close()

app = FastAPI(lifespan=lifespan)

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(f"<h3>Katya Core - Kurucu: {FOUNDER_NAME} Aktif.</h3>")

@app.head("/live")
@app.get("/live", response_class=HTMLResponse)
async def live_app():
    return HTMLResponse(content=LIVE_ORB_HTML)

@app.post("/api/voice_stream")
async def voice_stream_endpoint(audio_file: UploadFile = File(...)):
    audio_data = await audio_file.read()
    user_transcript = await transcribe_audio_with_groq(audio_data)
    if not user_transcript: user_transcript = "Ses anlasilamadi."

    reply = await execute_mother_order(user_transcript)
    audio_base64 = await text_to_speech_base64(reply)
    return JSONResponse({"reply": reply, "audio": audio_base64, "transcript": user_transcript})

@app.post("/api/voice_chat")
async def voice_chat_endpoint(payload: ChatPayload):
    reply = await execute_mother_order(payload.message)
    audio_base64 = await text_to_speech_base64(reply)
    return JSONResponse({"reply": reply, "audio": audio_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
