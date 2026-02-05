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
        words_data = result["words"]

        # 2. Calculate speaking metrics from timestamps
        speaking_metrics = transcription_service.calculate_speaking_metrics(words_data)

        # 3. Calculate pronunciation clarity from Whisper confidence scores
        pronunciation_clarity = transcription_service.calculate_pronunciation_clarity(words_data)

        # 4. Get IELTS analysis from Groq (now includes fluency analysis)
        llm_analysis = groq_service.get_ielts_analysis(
            transcript=result["text"],
            annotated=result["annotated"],
            speaking_metrics=speaking_metrics
        )

        # 5. Calculate confidence metrics from words
        confidences = [w["confidence"] for w in words_data]
        avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
        low_confidence_words = [
            {"word": w["word"], "confidence": w["confidence"]}
            for w in words_data if w["confidence"] < 0.7
        ]

        # 6. Extract LLM fluency analysis with fallbacks
        llm_fluency = llm_analysis.get("fluency_and_coherence", {})

        # Get cohesive devices from LLM
        cohesive_devices_data = llm_fluency.get("cohesive_devices", {
            "score": 0,
            "feedback": "Analysis failed",
            "devices_used": []
        })

        # 7. Calculate overall scores
        fluency_score = llm_analysis.get("overall_fluency_coherence_score", 0)
        lexical_score = llm_analysis.get("overall_lexical_score", 0)
        grammar_score = llm_analysis.get("overall_grammar_score", 0)
        pronunciation_score = pronunciation_clarity["clarity"]["score"]

        # Calculate band score (now includes pronunciation clarity!)
        available_scores = [s for s in [fluency_score, lexical_score, grammar_score, pronunciation_score] if s > 0]
        band_score = round(sum(available_scores) / len(available_scores), 1) if available_scores else 0

        # 7. Build the full IELTS response structure
        response = {
            "transcript": result["text"],
            "words": words_data,
            "analysis": {
                "fluency_and_coherence": {
                    "fluency": llm_fluency.get("fluency", {
                        "score": 0,
                        "feedback": "Analysis failed"
                    }),
                    "topic_development": llm_fluency.get("topic_development", {
                        "score": 0,
                        "feedback": "Analysis failed"
                    }),
                    "cohesive_devices": cohesive_devices_data,
                    "pauses": speaking_metrics["pauses"],
                    "speaking_speed": speaking_metrics["speaking_speed"]
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
                    "clarity": pronunciation_clarity["clarity"],
                    "word_counts": pronunciation_clarity["word_counts"],
                    "unclear_words": pronunciation_clarity["unclear_words"],
                    "intonation_and_stress": {
                        "score": 0,
                        "feedback": "Requires pitch/F0 extraction - In Development"
                    },
                    "chunking_and_rhythm": {
                        "score": 0,
                        "feedback": "Requires prosodic analysis - In Development"
                    }
                },
                "overall": {
                    "fluency_and_coherence_score": fluency_score,
                    "lexical_resource_score": lexical_score,
                    "grammar_score": grammar_score,
                    "pronunciation_score": pronunciation_score,
                    "band_score": band_score,
                    "summary": f"Full analysis complete. Intonation/stress and chunking/rhythm require additional audio models."
                },
                "filler_words": llm_analysis.get("filler_words", {"count": 0, "words": [], "impact": "Not analyzed"}),
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