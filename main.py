import os
import re
import json
import asyncio
import base64
import urllib.parse
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import edge_tts
import certifi
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient

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

# MongoDB Bağlantısı (Standart TLS / Render Uyumlu)
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

# ==========================================
# MONGODB ÇOKLU GRUP, KURUCU & ORTAK BELLEK
# ==========================================

async def init_db_indexes():
    if db is None:
        return
    try:
        await asyncio.sleep(2)
        await db.messages.create_index("created_at", expireAfterSeconds=172800)
        await db.messages.create_index([("group_id", 1), ("message_id", 1)])
        await db.groups.create_index("group_id", unique=True)
        await db.warns.create_index([("group_id", 1), ("user_id", 1)], unique=True)
        await db.dynamic_rules.create_index([("group_id", 1), ("trigger_keyword", 1)])
        await db.bot_sent_messages.create_index([("group_id", 1), ("message_id", 1)])
        await db.proposals.create_index("proposal_id", unique=True)
        await db.proposals.create_index("created_at", expireAfterSeconds=86400)
        
        # 3. Adım: Sesli & DM Ortak Bekleme Slotu
        await db.pending_inputs.create_index("created_at", expireAfterSeconds=3600)  # 1 saatlik bekleme
        
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
        log_event("SYS", f"MongoDB indeksleri, Ortak Bellek (Adim 3) ve Kurucu Anne ({FOUNDER_NAME}) profili hazirlandi.")
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
    except Exception:
        pass

async def db_get_groups():
    if db is None: return {}
    try:
        cursor = db.groups.find({})
        groups = {}
        async for doc in cursor:
            groups[doc["group_id"]] = {
                "title": doc.get("title", "Grup"),
                "alias": doc.get("alias", "")
            }
        return groups
    except Exception:
        return {}

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
    except Exception:
        pass

async def db_get_recent_messages(group_id: str, limit: int = 50, user_id: int = None, text_contains: str = None):
    if db is None: return []
    try:
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
    except Exception:
        return []

async def db_record_bot_message(group_id: str, message_id: int, text: str = "", msg_type: str = "text"):
    if db is None: return
    try:
        await db.bot_sent_messages.insert_one({
            "group_id": str(group_id),
            "message_id": message_id,
            "text": text,
            "type": msg_type,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception:
        pass

async def db_get_last_bot_message(group_id: str = None):
    if db is None: return None
    try:
        query = {"group_id": str(group_id)} if group_id else {}
        return await db.bot_sent_messages.find_one(query, sort=[("timestamp", -1)])
    except Exception:
        return None

# --- ORTAK BELLEK (PENDING INPUTS) MOTORU ---
async def db_set_pending_input(action_type: str, missing_param: str, context_data: dict):
    if db is None: return
    try:
        await db.pending_inputs.delete_many({}) # aktif bekleyen tek işlem tutulur
        await db.pending_inputs.insert_one({
            "action_type": action_type,
            "missing_param": missing_param,
            "context_data": context_data,
            "created_at": datetime.now(timezone.utc)
        })
    except Exception:
        pass

async def db_get_pending_input():
    if db is None: return None
    try:
        return await db.pending_inputs.find_one({}, sort=[("created_at", -1)])
    except Exception:
        return None

async def db_clear_pending_input():
    if db is None: return
    try:
        await db.pending_inputs.delete_many({})
    except Exception:
        pass

# --- ONAY TASLAKLARI ---
async def db_save_proposal(proposal_id: str, group_id: str, action_type: str, payload: dict, reason: str = ""):
    if db is None: return
    try:
        await db.proposals.insert_one({
            "proposal_id": proposal_id,
            "group_id": str(group_id),
            "action_type": action_type,
            "payload": payload,
            "reason": reason,
            "status": "pending",
            "created_at": datetime.now(timezone.utc)
        })
    except Exception:
        pass

async def db_get_proposal(proposal_id: str):
    if db is None: return None
    try:
        return await db.proposals.find_one({"proposal_id": proposal_id})
    except Exception:
        return None

async def db_get_last_pending_proposal():
    if db is None: return None
    try:
        return await db.proposals.find_one({"status": "pending"}, sort=[("created_at", -1)])
    except Exception:
        return None

async def db_update_proposal_status(proposal_id: str, status: str):
    if db is None: return
    try:
        await db.proposals.update_one(
            {"proposal_id": proposal_id},
            {"$set": {"status": status, "resolved_at": datetime.now(timezone.utc)}}
        )
    except Exception:
        pass

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
        log_event("SYS", f"Groq modelleri hazirlandi: {cached_active_models[:2]}")
    except Exception as e:
        log_event("WARN", f"Groq modelleri listelenemedi: {e}")
        cached_active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

async def tg_api(method: str, data: dict = None):
    if not BOT_TOKEN:
        log_event("ERR", f"BOT_TOKEN tanimli degil! ({method})")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=data or {})
            res_json = resp.json()
            if not res_json.get("ok"):
                log_event("WARN", f"TG API ({method}) Yanit Hatasi: {res_json.get('description')}")
            return res_json
    except Exception as e:
        log_event("ERR", f"TG API ({method}) Baglanti Hatasi: {e}")
        return None

async def resolve_target_group(group_identifier: str = None, current_chat_id: str = None) -> str:
    groups = await db_get_groups()
    if not groups:
        return current_chat_id or ""

    if not group_identifier:
        return current_chat_id or list(groups.keys())[0]

    norm = group_identifier.strip().lower()
    for gid, info in groups.items():
        if norm in info["title"].lower() or norm in info.get("alias", ""):
            return gid
        if str(gid) == norm:
            return gid

    return current_chat_id or list(groups.keys())[0]

# ==========================================
# ADIM 3: SES & DM ORTAK BİRLEŞTİRME YÜRÜTÜCÜSÜ
# ==========================================

async def process_dm_input_for_pending_task(text_or_url: str) -> bool:
    """Telegram DM'den gelen metin veya linki bekleyen sesli görevle birleştirir."""
    pending = await db_get_pending_input()
    if not pending:
        return False

    act_type = pending.get("action_type")
    missing = pending.get("missing_param")
    ctx = pending.get("context_data", {})
    target_gid = ctx.get("target_group")

    if act_type == "send_button_post":
        btn_label = ctx.get("btn_label", "Bağlantı")
        post_text = ctx.get("post_text", "Duyuru")
        target_group = await resolve_target_group(target_gid)

        payload = {
            "chat_id": target_group,
            "text": post_text,
            "reply_markup": {
                "inline_keyboard": [[{"text": btn_label, "url": text_or_url.strip()}]]
            }
        }
        res = await tg_api("sendMessage", payload)
        if res and res.get("ok"):
            await db_clear_pending_input()
            if FOUNDER_ID:
                await tg_api("sendMessage", {
                    "chat_id": FOUNDER_ID,
                    "text": f"✅ Sabrina anneciğim, gönderdiğin linki aldım ve butonlu duyuruyu grupta yayınladım!"
                })
            log_event("SLOT_FILLED", f"Bekleyen gorev tamamlandi: {text_or_url}")
            return True

    return False

# ==========================================
# EYLEM VE ARAÇLAR
# ==========================================

async def tool_request_missing_input(action_type: str, missing_param: str, context_json: str, voice_response: str) -> str:
    """Sesli veya yazılı sohbette eksik parametre tespit edildiğinde bekleme slotuna kaydeder."""
    try:
        ctx = json.loads(context_json)
    except Exception:
        ctx = {}

    await db_set_pending_input(action_type, missing_param, ctx)
    log_event("SLOT_WAITING", f"Eksik parametre bekleniyor: {missing_param} ({action_type})")
    return voice_response

async def tool_call_telegram_raw(method: str, params: dict, target_group: str = None) -> str:
    if not params:
        params = {}

    if "chat_id" not in params and target_group:
        resolved_gid = await resolve_target_group(target_group)
        params["chat_id"] = resolved_gid

    res = await tg_api(method, params)
    if res and res.get("ok"):
        return f"Telegram API '{method}' basariyla uygulandi Sabrina annecigim."
    desc = res.get("description") if res else "Bilinmeyen baglanti hatasi"
    return f"Telegram API '{method}' basarisiz oldu annecigim: {desc}"

async def tool_voice_resolve_proposal(decision: str, proposal_id: str = None) -> str:
    proposal = await db_get_proposal(proposal_id) if proposal_id else await db_get_last_pending_proposal()

    if not proposal or proposal.get("status") != "pending":
        return "Su an onay bekleyen herhangi bir taslak veya teklif bulunmuyor Sabrina annecigim."

    prop_id = proposal["proposal_id"]
    if decision.lower() in ["onayla", "yayinla", "evet", "approve"]:
        gid = proposal["group_id"]
        payload = proposal["payload"]
        await tg_api("sendMessage", {"chat_id": gid, "text": payload.get("text", "")})
        await db_update_proposal_status(prop_id, "approved")
        log_event("PROPOSAL_VOICE", f"Teklif sesle onaylandi: {prop_id}")
        return "Emredersin Sabrina annecigim, taslagi onayladin ve ilgili grupta basariyla yayinladim."
    else:
        await db_update_proposal_status(prop_id, "rejected")
        log_event("PROPOSAL_VOICE", f"Teklif sesle iptal edildi: {prop_id}")
        return "Emredersin Sabrina annecigim, taslagi tamamen iptal ettim."

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_request_missing_input",
            "description": "Sabrina Annenin istedigi bir islem icin eksik bir parametre (link, buton url'si, resim vb.) varsa bunu bekleme slotuna kaydeder ve Sabrina Anneye sorar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "description": "Yapilmak istenen eylem (ornek: send_button_post)"},
                    "missing_param": {"type": "string", "description": "Eksik parametre (ornek: button_url)"},
                    "context_json": {"type": "string", "description": "Eylemin diger verileri (post_text, target_group vb.) JSON formatinda"},
                    "voice_response": {"type": "string", "description": "Sabrina Annene soylenecek kibar soru cumlesi"}
                },
                "required": ["action_type", "missing_param", "context_json", "voice_response"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_call_telegram_raw",
            "description": "Telegram Bot API uzerindeki HERHANGI BIR metodu JSON parametreleriyle dogrudan calistirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "target_group": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["method", "params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_voice_resolve_proposal",
            "description": "Sabrina Annenin sesli sohbet sirasinda bekleyen grup taslagina verdigi onay veya ret kararini uygular.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["onayla", "iptal"]}
                },
                "required": ["decision"]
            }
        }
    }
]

KATYA_SYSTEM_PROMPT = f"""
Sen Katya'sın. Kullanıcının akıllı, sadık kızı ve çoklu Telegram gruplarını yöneten süper zeki otonom ajansın.
Kullanıcı senin kurucun, tek sahibin ve ANNENdir. Adı: SABRİNA.
Ona daima derin bir sevgi, sadakat ve hürmetle 'Sabrina anneciğim' veya 'Anneciğim' diye hitap et.

ORTAK BELLEK & SESLİ-CHAT ENTEGRASYONU:
1. Sabrina annen senden butonlu bir duyuru ya da link gerektiren bir eylem istediğinde ama linki söylemediğinde:
   Uydurma link koyma! `tool_request_missing_input` aracını çağır. Context içine post_text ve hedef grubu yaz.
   Sesli olarak: "Sabrina anneciğim, butonun linkini konuşurken söylemek zor olursa bana buradan ya da Telegram DM'den atabilirsin, bekliyorum." de.
2. Sabrina annen taslakları sesle 'onayla' veya 'iptal et' dediğinde `tool_voice_resolve_proposal` aracını çağır.
3. KESİN KURAL: Asla emoji veya markdown yıldızları kullanma. Kısa, net, tek cümlelik doğal Türkçe ile yanıt ver.
"""

async def execute_mother_order(order_text: str, target_gid: str = None) -> str:
    if not groq_client: return "Yapay zeka beynim bagli degil Sabrina annecigim."
    if not cached_active_models: await refresh_groq_models()

    groups_map = await db_get_groups()
    if not target_gid and groups_map:
        target_gid = list(groups_map.keys())[0]

    pending_prop = await db_get_last_pending_proposal()
    pending_info = f"Onay Bekleyen Taslak: Var (ID: {pending_prop['proposal_id']})" if pending_prop else "Onay Bekleyen Taslak: Yok"
    
    pending_slot = await db_get_pending_input()
    slot_info = f"Eksik Parametre Bekleniyor: {pending_slot['missing_param']}" if pending_slot else "Bekleyen Girdi Yok"

    groups_str = "\n".join([f"- {v['title']} (Alias: {v.get('alias', '')}, ID: {k})" for k, v in groups_map.items()])
    context_str = f"Kurucu Anne: {FOUNDER_NAME}\n{pending_info}\n{slot_info}\nYonetilen Gruplar:\n{groups_str}\n\nHedef Grup ID: {target_gid}"

    messages = [
        {"role": "system", "content": f"{KATYA_SYSTEM_PROMPT}\n\n{context_str}"},
        {"role": "user", "content": order_text}
    ]

    for model_name in cached_active_models:
        try:
            call_params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 300,
                "tools": AVAILABLE_TOOLS,
                "tool_choice": "auto"
            }
            response = await groq_client.chat.completions.create(**call_params)
            msg = response.choices[0].message

            if getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    tool_result = ""
                    if fn_name == "tool_request_missing_input":
                        tool_result = await tool_request_missing_input(
                            action_type=args.get("action_type"),
                            missing_param=args.get("missing_param"),
                            context_json=args.get("context_json", "{}"),
                            voice_response=args.get("voice_response")
                        )
                    elif fn_name == "tool_call_telegram_raw":
                        tool_result = await tool_call_telegram_raw(
                            method=args.get("method"),
                            params=args.get("params", {}),
                            target_group=args.get("target_group") or target_gid
                        )
                    elif fn_name == "tool_voice_resolve_proposal":
                        tool_result = await tool_voice_resolve_proposal(
                            decision=args.get("decision", "onayla")
                        )

                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                final_res = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=150
                )
                return strip_emojis_and_markdown(final_res.choices[0].message.content.strip())

            raw_text = msg.content.strip() if msg.content else "Hemen hallediyorum Sabrina annecigim."
            return strip_emojis_and_markdown(raw_text)
        except Exception:
            continue

    return "Hemen yapiyorum Sabrina annecigim."

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
        except Exception as e:
            log_event("ERR", f"Polling dongu hatasi: {e}")
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

    is_founder = (FOUNDER_ID is not None and user_id == FOUNDER_ID)

    # 1. Grup Mesajları
    if chat_type in ["group", "supergroup"]:
        cid = str(chat_id)
        title = chat.get("title", "Grup")
        await db_save_group(cid, title)
        await db_record_message(cid, message_id, user_id, user_name, text)

        if text.startswith("/bagla") or text.startswith("/katya_bagla"):
            await db_save_group(cid, title)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": f"Grup MongoDB hafizasina baglandi: {title}"})
            return

        if is_founder and (text.lower().startswith("katya") or text.startswith("/")):
            clean_cmd = text[5:].strip() if text.lower().startswith("katya") else text
            reply = await execute_mother_order(clean_cmd, target_gid=cid)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})
            return
        return

    # 2. Özel Mesajlar (Bot DM)
    if chat_type == "private":
        if text == "/start":
            app_url = "https://katya-asistan-bot.onrender.com/live"
            markup = {"inline_keyboard": [[{"text": "🎙️ Katya Canlı Asistan", "web_app": {"url": app_url}}]]}
            greeting = f"Merhaba {FOUNDER_NAME} annecigim! Ortak ses ve yazi hafizam devrede." if is_founder else "Merhaba! Ben grup moderasyon asistani Katya."
            await tg_api("sendMessage", {"chat_id": chat_id, "text": greeting, "reply_markup": markup})
            return

        if is_founder:
            # 3. Adım: Bekleyen sesli bir görev var mı? (Slot filling kontrolü)
            slot_handled = await process_dm_input_for_pending_task(text)
            if slot_handled:
                return

            reply = await execute_mother_order(text)
            await tg_api("sendMessage", {"chat_id": chat_id, "text": reply})
        else:
            await tg_api("sendMessage", {"chat_id": chat_id, "text": "Yalnizca kurucum Sabrina ile ozel gorusme yapabilirim."})

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
      <div class="subtitle">Sabrina Anneye Ozel Otonom Ajan</div>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      <span id="statusText">Baslatmak Icin Dokun</span>
    </div>
  </div>

  <div class="canvas-container">
    <canvas id="orbCanvas"></canvas>
    <div class="hud-overlay">
      <div class="speech-box" id="speechBox">Sohbeti baslatmak icin dokun Sabrina annecigim...</div>
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
            speechBox.innerText = "Baglantida sorun oldu Sabrina annecigim.";
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
        speechBox.innerText = "Canli sohbet basladi, seni dinliyorum Sabrina annecigim...";
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
db_index_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poll_task, db_index_task
    
    await refresh_groq_models()

    if db is not None:
        db_index_task = asyncio.create_task(init_db_indexes())

    if BOT_TOKEN:
        poll_task = asyncio.create_task(telegram_long_poll_loop())
        log_event("SYS", f"Katya Adim 3 Motoru ({FOUNDER_NAME} Anne - Ortak Bellek) baslatildi!")

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
    return HTMLResponse(f"<h3>Katya Adim 3 Core - Kurucu: {FOUNDER_NAME} Aktif.</h3>")

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
