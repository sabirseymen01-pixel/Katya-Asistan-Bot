"""
Katya Knowledge Base: Telegram Bot API & WebApp SDK Referans Şeması.
Katya kod yazmadan önce veya dinamik API çağrısı yaparken parametre kurallarını buradan doğrular.
"""

TELEGRAM_API_SPEC = {
    "sendPoll": {
        "description": "Grupta anket başlatır.",
        "required": ["chat_id", "question", "options"],
        "schema": {"chat_id": "str", "question": "str", "options": "list[str]", "is_anonymous": "bool"}
    },
    "setMessageReaction": {
        "description": "Belirli bir mesaja emoji reaksiyonu koyar.",
        "required": ["chat_id", "message_id", "reaction"],
        "schema": {"chat_id": "str", "message_id": "int", "reaction": "list[dict]"}
    },
    "createChatInviteLink": {
        "description": "Süreli veya limitli özel davet linki üretir.",
        "required": ["chat_id"],
        "schema": {"chat_id": "str", "name": "str", "expire_date": "int", "member_limit": "int"}
    },
    "sendMessage": {
        "description": "Düz veya inline butonlu mesaj gönderir.",
        "required": ["chat_id", "text"],
        "schema": {"chat_id": "str", "text": "str", "reply_markup": "dict"}
    },
    "pinChatMessage": {
        "description": "Bir mesajı başa sabitler.",
        "required": ["chat_id", "message_id"],
        "schema": {"chat_id": "str", "message_id": "int", "disable_notification": "bool"}
    },
    "unpinChatMessage": {
        "description": "Sabitlenen mesajı kaldırır.",
        "required": ["chat_id"],
        "schema": {"chat_id": "str", "message_id": "int"}
    },
    "setChatPermissions": {
        "description": "Grubun genel mesajlaşma ve medya izinlerini günceller.",
        "required": ["chat_id", "permissions"],
        "schema": {"chat_id": "str", "permissions": "dict"}
    },
    "restrictChatMember": {
        "description": "Bir kullanıcıyı belirli bir süre susturur veya kısıtlar.",
        "required": ["chat_id", "user_id", "permissions"],
        "schema": {"chat_id": "str", "user_id": "int", "permissions": "dict", "until_date": "int"}
    },
    "banChatMember": {
        "description": "Kullanıcıyı gruptan kalıcı olarak atar (banlar).",
        "required": ["chat_id", "user_id"],
        "schema": {"chat_id": "str", "user_id": "int", "revoke_messages": "bool"}
    },
    "unbanChatMember": {
        "description": "Kullanıcının banını kaldırır.",
        "required": ["chat_id", "user_id"],
        "schema": {"chat_id": "str", "user_id": "int", "only_if_banned": "bool"}
    },
    "editMessageText": {
        "description": "Botun attığı bir mesajın metnini veya butonunu günceller.",
        "required": ["chat_id", "message_id", "text"],
        "schema": {"chat_id": "str", "message_id": "int", "text": "str", "reply_markup": "dict"}
    },
    "deleteMessage": {
        "description": "Belirli bir mesajı siler.",
        "required": ["chat_id", "message_id"],
        "schema": {"chat_id": "str", "message_id": "int"}
    }
}

def get_knowledge_prompt() -> str:
    lines = ["TELEGRAM BOT API KÜTÜPHANESİ REFERANS REHBERİ:"]
    for method, info in TELEGRAM_API_SPEC.items():
        lines.append(f"- {method}: {info['description']} (Gerekenler: {', '.join(info['required'])}) | Şema: {info['schema']}")
    return "\n".join(lines)
