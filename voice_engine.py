import os
import asyncio
import edge_tts
from pyrogram import Client

API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH", "")
SESSION_STRING = os.getenv("SESSION_STRING", "")

user_client = None
call_client = None
AudioPipedClass = None

if SESSION_STRING and API_ID and API_HASH:
    try:
        from pytgcalls import PyTgCalls
        try:
            from pytgcalls.types.input_stream import AudioPiped
            AudioPipedClass = AudioPiped
        except Exception:
            from pytgcalls.types import AudioPiped
            AudioPipedClass = AudioPiped

        user_client = Client(
            "katya_assistant_user",
            api_id=API_ID,
            api_hash=API_HASH,
            session_string=SESSION_STRING,
            in_memory=True
        )
        call_client = PyTgCalls(user_client)
    except Exception as e:
        print(f"[VOICE] PyTgCalls başlatma uyarısı: {e}")
        call_client = None

async def start_voice_client():
    """UserBot ve ses motorunu başlatır."""
    if user_client and call_client:
        try:
            await user_client.start()
            await call_client.start()
            print("[VOICE] UserBot ve PyTgCalls aktif.")
        except Exception as e:
            print(f"[VOICE] Başlatma hatası: {e}")

async def stop_voice_client():
    """Kapanışta istemcileri durdurur."""
    if call_client:
        try:
            await call_client.stop()
        except Exception:
            pass

async def join_and_speak(chat_id: int, text: str):
    """Metni Türkçe kadın sesine çevirir ve sesli sohbete basar."""
    if not call_client or not AudioPipedClass:
        return "Ses istemcisi aktif değil veya kütüphane uyumsuz."

    file_id = f"{abs(chat_id)}_{int(asyncio.get_event_loop().time())}"
    temp_mp3 = f"voice_{file_id}.mp3"

    try:
        communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
        await communicate.save(temp_mp3)

        audio_stream = AudioPipedClass(temp_mp3)

        try:
            await call_client.join_group_call(chat_id, audio_stream)
        except Exception:
            await call_client.change_stream(chat_id, audio_stream)

        return None
    except Exception as e:
        print(f"[VOICE] Ses yayını hatası: {e}")
        return str(e)
    finally:
        await asyncio.sleep(5)
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except Exception:
                pass

async def leave_voice(chat_id: int):
    """Sesli sohbetten ayrılır."""
    if call_client:
        try:
            await call_client.leave_group_call(chat_id)
        except Exception as e:
            print(f"[VOICE] Ayrılma hatası: {e}")
