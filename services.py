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

    def calculate_speaking_metrics(self, words_data: list) -> dict:
        """Calculate speaking speed and pause analysis from word timestamps"""
        if not words_data or len(words_data) < 2:
            return {
                "speaking_speed": {
                    "words_per_minute": 0,
                    "total_words": len(words_data) if words_data else 0,
                    "total_duration_seconds": 0,
                    "assessment": "Insufficient data"
                },
                "pauses": {
                    "good_pauses": [],
                    "bad_pauses": [],
                    "good_pause_count": 0,
                    "bad_pause_count": 0,
                    "total_pause_time": 0,
                    "feedback": "Insufficient data for pause analysis"
                }
            }

        # Calculate total duration
        first_word_start = words_data[0]["start"]
        last_word_end = words_data[-1]["end"]
        total_duration = last_word_end - first_word_start

        # Words per minute calculation
        total_words = len(words_data)
        wpm = round((total_words / total_duration) * 60) if total_duration > 0 else 0

        # Assess speaking speed
        if wpm < 100:
            speed_assessment = "Too slow - may indicate hesitation or difficulty"
        elif wpm < 120:
            speed_assessment = "Slightly slow - natural but could be more fluent"
        elif wpm <= 150:
            speed_assessment = "Good pace - natural conversational speed"
        elif wpm <= 170:
            speed_assessment = "Slightly fast - still clear but approaching rushed"
        else:
            speed_assessment = "Too fast - may affect clarity and comprehension"

        # Pause analysis
        good_pauses = []  # Natural pauses (0.3s - 1.5s)
        bad_pauses = []   # Hesitation pauses (> 1.5s) or too many short pauses

        total_pause_time = 0

        for i in range(1, len(words_data)):
            prev_word = words_data[i - 1]
            curr_word = words_data[i]

            gap = curr_word["start"] - prev_word["end"]

            if gap > 0.3:  # Only consider gaps > 300ms as pauses
                pause_info = {
                    "after_word": prev_word["word"],
                    "before_word": curr_word["word"],
                    "duration": round(gap, 2),
                    "position": round(prev_word["end"], 2)
                }

                total_pause_time += gap

                if 0.3 <= gap <= 1.5:
                    # Natural pause - good for thought organization
                    good_pauses.append(pause_info)
                else:
                    # Long pause - likely hesitation
                    bad_pauses.append(pause_info)

        # Generate pause feedback
        if len(bad_pauses) == 0 and len(good_pauses) <= 5:
            pause_feedback = "Excellent fluency with natural pausing patterns"
        elif len(bad_pauses) == 0:
            pause_feedback = "Good fluency with appropriate pauses for thought organization"
        elif len(bad_pauses) <= 2:
            pause_feedback = "Generally fluent with occasional hesitations"
        elif len(bad_pauses) <= 4:
            pause_feedback = "Some noticeable hesitations affecting fluency"
        else:
            pause_feedback = "Frequent hesitations significantly impacting fluency"

        return {
            "speaking_speed": {
                "words_per_minute": wpm,
                "total_words": total_words,
                "total_duration_seconds": round(total_duration, 2),
                "assessment": speed_assessment
            },
            "pauses": {
                "good_pauses": good_pauses,
                "bad_pauses": bad_pauses,
                "good_pause_count": len(good_pauses),
                "bad_pause_count": len(bad_pauses),
                "total_pause_time": round(total_pause_time, 2),
                "feedback": pause_feedback
            }
        }

    def calculate_pronunciation_clarity(self, words_data: list) -> dict:
        """
        Calculate pronunciation clarity based on Whisper confidence scores.

        Logic:
        - High confidence (>0.85) = Whisper easily recognized = clear pronunciation
        - Medium confidence (0.7-0.85) = Some difficulty = acceptable pronunciation
        - Low confidence (<0.7) = Whisper struggled = unclear/mispronounced
        """
        if not words_data:
            return {
                "clarity": {
                    "score": 0,
                    "average_confidence": 0,
                    "feedback": "Insufficient data"
                },
                "clear_words": [],
                "unclear_words": [],
                "clarity_percentage": 0
            }

        # Categorize words by confidence
        clear_words = []      # confidence >= 0.85
        acceptable_words = [] # 0.7 <= confidence < 0.85
        unclear_words = []    # confidence < 0.7

        # Common filler words to exclude from pronunciation assessment
        filler_words = {'um', 'uh', 'hmm', 'ah', 'er', 'like', 'you know', 'basically', 'actually', 'so', 'well'}

        for word_info in words_data:
            word = word_info["word"].lower().strip()
            confidence = word_info["confidence"]

            # Skip filler words - they're not pronunciation issues
            if word in filler_words:
                continue

            word_entry = {
                "word": word_info["word"],
                "confidence": confidence,
                "start": word_info["start"]
            }

            if confidence >= 0.85:
                clear_words.append(word_entry)
            elif confidence >= 0.7:
                acceptable_words.append(word_entry)
            else:
                unclear_words.append(word_entry)

        # Calculate metrics
        total_assessed = len(clear_words) + len(acceptable_words) + len(unclear_words)

        if total_assessed == 0:
            return {
                "clarity": {
                    "score": 0,
                    "average_confidence": 0,
                    "feedback": "No assessable words found"
                },
                "clear_words": [],
                "unclear_words": [],
                "clarity_percentage": 0
            }

        # Calculate average confidence (excluding fillers)
        all_confidences = [w["confidence"] for w in clear_words + acceptable_words + unclear_words]
        avg_confidence = round(sum(all_confidences) / len(all_confidences), 2)

        # Calculate clarity percentage (clear + acceptable words)
        clarity_percentage = round(((len(clear_words) + len(acceptable_words)) / total_assessed) * 100, 1)

        # Map to IELTS-style 1-9 score based on clarity percentage
        if clarity_percentage >= 95:
            score = 9
            feedback = "Excellent pronunciation clarity - speech is easily understood with no strain on the listener"
        elif clarity_percentage >= 90:
            score = 8
            feedback = "Very good clarity - occasional unclear words but overall highly intelligible"
        elif clarity_percentage >= 85:
            score = 7
            feedback = "Good clarity - generally clear with some words requiring listener effort"
        elif clarity_percentage >= 80:
            score = 6
            feedback = "Acceptable clarity - mostly understandable but some pronunciation issues affect comprehension"
        elif clarity_percentage >= 70:
            score = 5
            feedback = "Moderate clarity - noticeable pronunciation issues that may cause misunderstanding"
        elif clarity_percentage >= 60:
            score = 4
            feedback = "Limited clarity - frequent unclear pronunciation affecting communication"
        elif clarity_percentage >= 50:
            score = 3
            feedback = "Poor clarity - significant pronunciation issues making speech difficult to understand"
        else:
            score = 2
            feedback = "Very poor clarity - pronunciation issues severely impact intelligibility"

        # Sort unclear words by confidence (worst first) and limit to top 10
        unclear_words_sorted = sorted(unclear_words, key=lambda x: x["confidence"])[:10]

        return {
            "clarity": {
                "score": score,
                "average_confidence": avg_confidence,
                "clarity_percentage": clarity_percentage,
                "feedback": feedback
            },
            "word_counts": {
                "clear": len(clear_words),
                "acceptable": len(acceptable_words),
                "unclear": len(unclear_words),
                "total_assessed": total_assessed
            },
            "unclear_words": unclear_words_sorted
        }

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

    def get_ielts_analysis(self, transcript: str, annotated: str, speaking_metrics: dict = None):
        """Analyze speech for IELTS scoring - Fluency, Lexical Resource, and Grammar"""
        if not self.client:
            return {"error": "Groq API key not configured"}

        # Build speaking metrics context for the LLM
        metrics_context = ""
        if speaking_metrics:
            speed = speaking_metrics.get("speaking_speed", {})
            pauses = speaking_metrics.get("pauses", {})
            metrics_context = f"""
SPEAKING METRICS (calculated from audio timestamps):
- Words per minute: {speed.get('words_per_minute', 0)}
- Total words: {speed.get('total_words', 0)}
- Total duration: {speed.get('total_duration_seconds', 0)} seconds
- Speed assessment: {speed.get('assessment', 'N/A')}
- Good pauses (natural): {pauses.get('good_pause_count', 0)}
- Bad pauses (hesitations >1.5s): {pauses.get('bad_pause_count', 0)}
- Total pause time: {pauses.get('total_pause_time', 0)} seconds
- Pause feedback: {pauses.get('feedback', 'N/A')}
"""

        system_message = """You are an expert IELTS speaking examiner. Analyze the transcript and provide detailed feedback.

You must respond with a valid JSON object with this EXACT structure:
{
    "fluency_and_coherence": {
        "fluency": {
            "score": <number 1-9>,
            "feedback": "<string - assess flow, hesitation, self-correction, false starts>"
        },
        "topic_development": {
            "score": <number 1-9>,
            "feedback": "<string - assess how well ideas are extended, examples given, relevance maintained>"
        },
        "cohesive_devices": {
            "score": <number 1-9>,
            "feedback": "<string - assess use of linking words, transitions, discourse markers>",
            "devices_used": ["<list of cohesive devices found in transcript>"]
        }
    },
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
        "words": ["<list of filler words found: um, uh, like, you know, basically, actually, etc.>"],
        "impact": "<string - assess impact on fluency>"
    },
    "overall_fluency_coherence_score": <number 1-9>,
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

FLUENCY & COHERENCE Scoring Guide:
- Band 9: Speaks fluently with only rare repetition or self-correction; any hesitation is content-related; speech develops topics fully and appropriately
- Band 8: Speaks fluently with only occasional repetition or self-correction; hesitation is usually content-related; develops topics coherently
- Band 7: Speaks at length without noticeable effort; may demonstrate language-related hesitation or repetition; uses a range of connectives and discourse markers
- Band 6: Willing to speak at length but may lose coherence; uses connectives but not always appropriately
- Band 5: Usually maintains flow but uses repetition and self-correction; may speak slowly with pauses; over-uses certain connectives
- Band 4: Cannot respond without noticeable pauses; may speak slowly with frequent repetition

Be thorough but fair in your assessment. Identify specific examples from the transcript."""

        prompt = f"""
TRANSCRIPT: {transcript}

TRANSCRIPT WITH CONFIDENCE SCORES: {annotated}
{metrics_context}
The confidence scores (0.0 to 1.0) indicate speech-to-text model certainty.
Low scores (< 0.7) may indicate mumbling, mispronunciation, or unclear speech.
Filler words like 'um', 'uh', 'like', 'you know' are preserved - count them and assess their impact.

Analyze this IELTS speaking sample for:
1. Fluency and Coherence (fluency, topic development, cohesive devices usage)
2. Lexical Resource (vocabulary range, accuracy, idiomatic language, CEFR level)
3. Grammar (range of structures, grammar accuracy, tense accuracy, specific errors)

Use the speaking metrics provided to inform your fluency assessment.
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