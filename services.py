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

            full_text, annotated, words_data = self.transcribe_with_confidence(temp_path)

            return {
                "text": full_text,
                "annotated": annotated,
                "words": words_data
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
        words_data = []

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
                words_data.append({
                    "word": word_text,
                    "confidence": confidence,
                    "start": round(word.start, 2),
                    "end": round(word.end, 2)
                })

        return " ".join(full_text), " ".join(annotated_transcript), words_data


class GroqService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️  Warning: GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_id = "llama-3.3-70b-versatile"

    def get_ielts_analysis(self, transcript: str, annotated: str):
        """Analyze speech for IELTS scoring - Lexical Resource and Grammar only"""
        if not self.client:
            return {"error": "Groq API key not configured"}

        system_message = """You are an expert IELTS speaking examiner. Analyze the transcript and provide detailed feedback.

You must respond with a valid JSON object with this EXACT structure:
{
    "lexical_resource": {
        "vocabulary_range": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "accuracy": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "idiomatic_language": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "vocabulary_mistakes": [
            {
                "original": "<word used>",
                "suggestion": "<better word>",
                "explanation": "<why>"
            }
        ],
        "cefr_level": "<A1|A2|B1|B2|C1|C2>"
    },
    "grammar": {
        "range_of_structures": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "grammar_accuracy": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "tense_accuracy": {
            "score": <number 1-9>,
            "feedback": "<string>"
        },
        "errors": [
            {
                "original": "<incorrect phrase>",
                "correction": "<corrected phrase>",
                "type": "<error type: grammar|tense|word_choice|syntax>",
                "explanation": "<explanation>"
            }
        ]
    },
    "filler_words": {
        "count": <number>,
        "words": ["<list of filler words found>"]
    },
    "overall_lexical_score": <number 1-9>,
    "overall_grammar_score": <number 1-9>
}

IELTS Band Score Guidelines:
- Band 9: Expert user - full operational command, appropriate, accurate, complete understanding
- Band 8: Very good user - occasional inaccuracies, handles complex detailed argumentation well
- Band 7: Good user - occasional inaccuracies, handles complex language well, detailed reasoning
- Band 6: Competent user - generally effective command despite inaccuracies
- Band 5: Modest user - partial command, handles basic communication
- Band 4: Limited user - basic competence, frequent problems
- Band 3: Extremely limited user - conveys and understands only general meaning
- Band 2: Intermittent user - great difficulty understanding
- Band 1: Non-user - no ability except isolated words

Be thorough but fair in your assessment. Identify specific examples from the transcript."""

        prompt = f"""
TRANSCRIPT: {transcript}

TRANSCRIPT WITH CONFIDENCE SCORES: {annotated}

The confidence scores (0.0 to 1.0) indicate speech-to-text model certainty.
Low scores (< 0.7) may indicate mumbling, mispronunciation, or unclear speech.
Filler words like 'um', 'uh', 'like', 'you know' are preserved - count them.

Analyze this IELTS speaking sample for:
1. Lexical Resource (vocabulary range, accuracy, idiomatic language, CEFR level)
2. Grammar (range of structures, grammar accuracy, tense accuracy, specific errors)

Provide detailed, constructive feedback with specific examples from the transcript.
"""

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000
        )

        return json.loads(response.choices[0].message.content)

    def get_analysis(self, transcript: str, annotated: str, system_message: str):
        """Legacy analysis method for custom prompts"""
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