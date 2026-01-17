import os
import json
import httpx
import tempfile
from groq import Groq
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

class WhisperService:
    def __init__(self):
        # M4 Mac Optimization - uses CoreML automatically
        print("Loading Whisper model...")
        self.model = WhisperModel(
            "small.en",
            device="cpu",
            compute_type="int8",  # Optimized for M4
        )
        print("✓ Whisper model loaded")
        
    async def transcribe_from_url(self, url: str):
        temp_path = None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(response.content)
                    temp_path = temp_audio.name

            full_text, annotated = self.transcribe_with_confidence(temp_path)
            
            return {
                "text": full_text,
                "annotated": annotated
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def transcribe_with_confidence(self, audio_path: str):
        # Transcribe with word-level timestamps
        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,  # Enable word-level data
            vad_filter=False,  # Don't filter filler words
            initial_prompt="Um, uh, like, you know, hmm, ah",  # Preserve fillers
        )
        
        full_text = []
        annotated_transcript = []
        
        # Extract words with confidence
        for segment in segments:
            for word in segment.words:
                word_text = word.word.strip()
                if not word_text:
                    continue
                
                # Get confidence (probability)
                confidence = word.probability
                
                import math
                if math.isnan(confidence):
                    confidence = 0.5
                
                confidence = round(confidence, 2)
                
                full_text.append(word_text)
                annotated_transcript.append(f"{word_text}({confidence})")
        
        return " ".join(full_text), " ".join(annotated_transcript)


class GroqService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️  Warning: GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_id = "llama-3.3-70b-versatile"

    def get_analysis(self, transcript: str, annotated: str, system_message: str):
        if not self.client:
            return {"error": "Groq API key not configured"}
        
        prompt = f"""
TRANSCRIPT: {transcript}

TRANSCRIPT WITH CONFIDENCE SCORES: {annotated}

The confidence scores (0.0 to 1.0) indicate how sure the STT model was. 
Low scores (< 0.7) might mean the speaker mumbled, mispronounced, or there was background noise.
Filler words like 'um', 'uh', 'like', 'you know' are preserved - please count them in your analysis.

Please analyze the speech quality and provide insights.
"""
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000
        )
        
        return json.loads(response.choices[0].message.content)


transcription_service = WhisperService()
groq_service = GroqService()