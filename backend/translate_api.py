from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO

router = APIRouter()


# -----------------------------
# TRANSLATION (Google Translate)
# -----------------------------
@router.post("/translate")
async def translate(payload: dict):
    text = payload.get("text", "")
    target = payload.get("target", "hi")

    if not text:
        raise HTTPException(status_code=400, detail="text required")

    try:
        result = GoogleTranslator(source='auto', target=target).translate(text)
        return JSONResponse(content={
            "translatedText": result,
            "source": "googletrans"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# -----------------------------
# TEXT TO SPEECH
# -----------------------------
GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "or": "hi"  # Odia fallback
}

@router.post("/tts")
async def tts(payload: dict):
    text = payload.get("text", "")
    lang = payload.get("lang", "en")

    if not text:
        raise HTTPException(status_code=400, detail="text required")

    try:
        tts = gTTS(text=text, lang=GTTS_LANG_MAP.get(lang, "en"))
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail="TTS failed")
