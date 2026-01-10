import os
import tempfile
import httpx
from fastapi import FastAPI, HTTPException, Query
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Direct link to a short English speech sample (Introductory clip)
DEFAULT_SPEECH_URL = "https://pronunciationstudio.com/wp-content/uploads/2016/02/Audio-Introduction-0.1.mp3"

@app.get("/transcript")
async def get_transcript(url: str = Query(default=DEFAULT_SPEECH_URL)):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing in .env")
        
    try:
        # 1. Download the audio
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            response = await http_client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Download failed: {response.status_code}")
            audio_content = response.content

        # 2. Use a temp file for Groq processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_content)
            tmp_path = tmp.name

        # 3. Call Groq Whisper
        try:
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_path, file.read()),
                    model="whisper-large-v3",
                    temperature=0,
                    response_format="json",
                )
            result_text = transcription.text
        finally:
            # Always clean up the file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "status": "success",
            "transcript": result_text,
            "url": url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))