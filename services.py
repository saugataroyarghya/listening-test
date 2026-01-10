import os
import tempfile
import httpx
from groq import Groq

class TranscriptionService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def transcribe_from_url(self, url: str) -> str:
        # 1. Download
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            response = await http_client.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to download audio: {response.status_code}")
            audio_content = response.content

        # 2. Process via Temp File
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_content)
            tmp_path = tmp.name

        try:
            # 3. Transcribe
            with open(tmp_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(tmp_path, file.read()),
                    model="whisper-large-v3",
                    temperature=0,
                    response_format="json",
                )
            return transcription.text
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Create a singleton instance to be used across the app
transcription_service = TranscriptionService()