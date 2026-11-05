import os
import asyncio
import edge_tts
from pyrogram import Client

API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH", "")
SESSION_STRING = os.getenv("SESSION_STRING", "")

user_client = None
call_client = None
StreamClass = None

if SESSION_STRING and API_ID and API_HASH:
    try:
        from pytgcalls import PyTgCalls

        # py-tgcalls 2.x için MediaStream, eski sürümler için AudioPiped uyumluluğu
        try:
            from pytgcalls.types import MediaStream
            StreamClass = MediaStream
        except Exception:
            try:
                from pytgcalls.types.input_stream import AudioPiped
                StreamClass = AudioPiped
            except Exception:
                from pytgcalls.types import AudioPiped
                StreamClass = AudioPiped

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
    if not call_client or not StreamClass:
        return "Ses istemcisi aktif değil veya kütüphane uyumsuz."

    file_id = f"{abs(chat_id)}_{int(asyncio.get_event_loop().time())}"
    temp_mp3 = f"voice_{file_id}.mp3"

    try:
        communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
        await communicate.save(temp_mp3)

        audio_stream = StreamClass(temp_mp3)

        try:
            # py-tgcalls 2.x API desteği: play veya join_group_call
            if hasattr(call_client, "play"):
                await call_client.play(chat_id, audio_stream)
            else:
                await call_client.join_group_call(chat_id, audio_stream)
        except Exception:
            if hasattr(call_client, "change_stream"):
                await call_client.change_stream(chat_id, audio_stream)
            elif hasattr(call_client, "play"):
                await call_client.play(chat_id, audio_stream)

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
            if hasattr(call_client, "leave_call"):
                await call_client.leave_call(chat_id)
            else:
                await call_client.leave_group_call(chat_id)
        except Exception as e:
            print(f"[VOICE] Ayrılma hatası: {e}")
