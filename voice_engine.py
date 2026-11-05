import os
import asyncio
import edge_tts
from pyrogram import Client
from pytgcalls import PyTgCalls
from pytgcalls.types import AudioPiped

API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH", "")
SESSION_STRING = os.getenv("SESSION_STRING", "")

# UserBot İstemcisi
user_client = Client(
    "katya_assistant_user",
    api_id=API_ID,
    api_hash=API_HASH,
    session_string=SESSION_STRING
)

call_client = PyTgCalls(user_client)

async def start_voice_client():
    await user_client.start()
    await call_client.start()

async def join_and_speak(chat_id: int, text: str):
    """Metni doğal Türkçe sesine dönüştürür ve sesli sohbete canlı yayınlar."""
    audio_path = "katya_live_voice.raw"
    temp_mp3 = "temp_katya.mp3"
    
    # 1. Edge-TTS ile sesi üret
    communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
    await communicate.save(temp_mp3)

    # 2. PyTgCalls için uygun ham ses formatına çevirme (ffmpeg)
    os.system(f"ffmpeg -y -i {temp_mp3} -f s16le -ac 1 -ar 48000 {audio_path}")

    # 3. Ses kanalına bağlan ve yayını başlat
    try:
        await call_client.join_group_call(
            chat_id,
            AudioPiped(audio_path)
        )
    except Exception:
        # Zaten odadaysa sesi değiştirir/akışı günceller
        await call_client.change_stream(
            chat_id,
            AudioPiped(audio_path)
        )

async def leave_voice(chat_id: int):
    """Sesli sohbetten ayrılır."""
    await call_client.leave_group_call(chat_id)
