from fastapi import FastAPI, HTTPException, Query
from services import transcription_service
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

DEFAULT_SPEECH_URL = "https://pronunciationstudio.com/wp-content/uploads/2016/02/Audio-Introduction-0.1.mp3"

@app.get("/transcript")
async def get_transcript(url: str = Query(default=DEFAULT_SPEECH_URL)):
    try:
        text = await transcription_service.transcribe_from_url(url)
        return {
            "status": "success",
            "transcript": text,
            "url": url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))