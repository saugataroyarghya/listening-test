from fastapi import FastAPI, HTTPException, Query
from services import transcription_service, groq_service
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(
    title="Speech Analysis API",
    description="Transcribe audio with confidence scores and analyze speech quality",
    version="1.0.0"
)

DEFAULT_SPEECH_URL = "https://pronunciationstudio.com/wp-content/uploads/2016/02/Audio-Introduction-0.1.mp3"

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Speech Analysis API",
        "endpoints": {
            "/analyzeSpeech": "Transcribe and analyze speech from audio URL",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Check if services are ready"""
    return {
        "status": "healthy",
        "whisper_loaded": transcription_service.model is not None,
        "groq_configured": groq_service.client is not None
    }

@app.get("/analyzeSpeech")
async def analyze_speech(
    url: str = Query(
        default=DEFAULT_SPEECH_URL,
        description="URL of the audio file to transcribe and analyze"
    )
):
    """
    Transcribe audio from URL and analyze speech quality for IELTS.

    Returns IELTS-formatted analysis with:
    - transcript: Plain text transcription
    - words: Word-level details with confidence and timestamps
    - analysis: Comprehensive IELTS analysis including:
        - Fluency and Coherence (In Development)
        - Lexical Resource (vocabulary, accuracy, idiomatic language, CEFR level)
        - Grammar (structures, accuracy, tense, errors)
        - Pronunciation (In Development)
        - Overall band scores (1-9 scale)
    """
    try:
        # 1. Transcribe with Whisper
        result = await transcription_service.transcribe_from_url(url)

        # 2. Get IELTS analysis from Groq
        llm_analysis = groq_service.get_ielts_analysis(
            transcript=result["text"],
            annotated=result["annotated"]
        )

        # 3. Calculate confidence metrics from words
        words_data = result["words"]
        confidences = [w["confidence"] for w in words_data]
        avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
        low_confidence_words = [
            {"word": w["word"], "confidence": w["confidence"]}
            for w in words_data if w["confidence"] < 0.7
        ]

        # 4. Build the full IELTS response structure
        response = {
            "transcript": result["text"],
            "words": words_data,
            "analysis": {
                "fluency_and_coherence": {
                    "fluency": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "topic_development": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "cohesive_devices": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "pauses": {
                        "good_pauses": 0,
                        "bad_pauses": 0,
                        "feedback": "In Development"
                    },
                    "speaking_speed": {
                        "words_per_minute": 0,
                        "assessment": "In Development"
                    }
                },
                "lexical_resource": llm_analysis.get("lexical_resource", {
                    "vocabulary_range": {"score": 0, "feedback": "Analysis failed"},
                    "accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "idiomatic_language": {"score": 0, "feedback": "Analysis failed"},
                    "vocabulary_mistakes": [],
                    "cefr_level": "Unknown"
                }),
                "grammar": llm_analysis.get("grammar", {
                    "range_of_structures": {"score": 0, "feedback": "Analysis failed"},
                    "grammar_accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "tense_accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "errors": []
                }),
                "pronunciation": {
                    "clarity": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "intonation_and_stress": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "chunking_and_rhythm": {
                        "score": 0,
                        "feedback": "In Development"
                    },
                    "mistakes": []
                },
                "overall": {
                    "fluency_and_coherence_score": 0,
                    "lexical_resource_score": llm_analysis.get("overall_lexical_score", 0),
                    "grammar_score": llm_analysis.get("overall_grammar_score", 0),
                    "pronunciation_score": 0,
                    "band_score": 0,
                    "summary": "Partial analysis - Fluency/Coherence and Pronunciation are in development."
                },
                "filler_words": llm_analysis.get("filler_words", {"count": 0, "words": []}),
                "low_confidence_words": low_confidence_words,
                "average_confidence": avg_confidence
            }
        }

        return response

    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )