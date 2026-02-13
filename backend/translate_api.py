# backend/translate_api.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from gtts import gTTS
from io import BytesIO
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Try a prioritized list of public translation endpoints
TRANSLATION_ENDPOINTS = [
    "https://translate.argosopentech.com/translate",
    "https://libretranslate.de/translate"
]

@router.post("/translate")
async def translate(payload: dict):
    """
    payload: { "text": "...", "target": "hi" }
    Returns the translation JSON { translatedText: "..." } on success.
    """
    text = payload.get("text", "")
    target = payload.get("target", "hi")
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    for endpoint in TRANSLATION_ENDPOINTS:
        try:
            resp = requests.post(
                endpoint,
                json={"q": text, "source": "en", "target": target, "format": "text"},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                # standardize structure: some instances return 'translatedText' directly or as string
                if isinstance(data, dict) and "translatedText" in data:
                    return JSONResponse(content={"translatedText": data["translatedText"], "source": endpoint})
                # libretranslate returns {"translatedText": "..."} usually; argos also
                # fallback: if string returned
                if isinstance(data, str):
                    return JSONResponse(content={"translatedText": data, "source": endpoint})
            else:
                logger.warning("Translate failed %s -> %s", endpoint, resp.status_code)
        except Exception as e:
            logger.warning("Translate endpoint error %s -> %s", endpoint, str(e))

    # If all endpoints failed:
    raise HTTPException(status_code=503, detail="Translation service unavailable")

# TTS endpoint using gTTS (quick demo-grade solution)
GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "or": "or"  # gTTS may not support Odia; fallback to Hindi/English may be necessary
}

@router.post("/tts")
async def tts(payload: dict):
    """
    payload: { "text":"...", "lang":"hi" }
    Returns audio/mpeg stream of speech.
    """
    text = payload.get("text", "")
    lang = payload.get("lang", "en")
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    # map to gTTS supported code (basic)
    gtts_lang = GTTS_LANG_MAP.get(lang.split('-')[0], "en")

    try:
        tts = gTTS(text=text, lang=gtts_lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail="TTS generation failed")
