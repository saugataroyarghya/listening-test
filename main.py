from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
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


def combine_scores(llm_score: float, local_score: float, llm_weight: float = 0.6) -> float:
    if llm_score > 0 and local_score > 0:
        return round((llm_score * llm_weight) + (local_score * (1 - llm_weight)), 1)
    if llm_score > 0:
        return round(llm_score, 1)
    if local_score > 0:
        return round(local_score, 1)
    return 0.0


def build_analysis_response(result: dict, question: Optional[str] = None) -> dict:
    words_data = result["words"]

    speaking_metrics = transcription_service.calculate_speaking_metrics(words_data)
    pronunciation_clarity = transcription_service.calculate_pronunciation_clarity(words_data)
    prosody_features = transcription_service.calculate_prosody_features(words_data)
    local_lexical = transcription_service.calculate_lexical_resource(result["text"], words_data)
    local_grammar = transcription_service.calculate_grammar_analysis(result["text"])

    llm_analysis = groq_service.get_ielts_analysis(
        transcript=result["text"],
        annotated=result["annotated"],
        question=question,
        speaking_metrics=speaking_metrics,
        feature_context={
            "fluency_features": speaking_metrics.get("fluency_features", {}),
            "prosody_features": prosody_features,
            "lexical_metrics": local_lexical.get("local_metrics", {}),
            "grammar_metrics": local_grammar.get("local_metrics", {})
        }
    )

    confidences = [w["confidence"] for w in words_data]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0
    low_confidence_words = [
        {"word": w["word"], "confidence": w["confidence"]}
        for w in words_data if w["confidence"] < 0.7
    ]

    llm_fluency = llm_analysis.get("fluency_and_coherence", {})
    cohesive_devices_data = llm_fluency.get("cohesive_devices", {
        "score": 0,
        "feedback": "Analysis failed",
        "devices_used": []
    })

    local_fluency_score = speaking_metrics.get("fluency_features", {}).get("fluency_score", 0)
    llm_fluency_score = llm_analysis.get("overall_fluency_coherence_score", 0)
    fluency_score = combine_scores(llm_fluency_score, local_fluency_score)

    llm_lexical_score = llm_analysis.get("overall_lexical_score", 0)
    local_lexical_score = local_lexical.get("overall_lexical_score", 0)
    lexical_confidence = local_lexical.get("deterministic_confidence", 0.0)
    lexical_llm_weight = 0.7 if lexical_confidence < 0.5 else 0.5
    lexical_score = combine_scores(llm_lexical_score, local_lexical_score, llm_weight=lexical_llm_weight)

    llm_grammar_score = llm_analysis.get("overall_grammar_score", 0)
    local_grammar_score = local_grammar.get("overall_grammar_score", 0)
    grammar_confidence = local_grammar.get("deterministic_confidence", 0.0)
    grammar_llm_weight = 0.65 if grammar_confidence < 0.7 else 0.45
    grammar_score = combine_scores(llm_grammar_score, local_grammar_score, llm_weight=grammar_llm_weight)

    pronunciation_score = round(
        (
            pronunciation_clarity["clarity"]["score"] +
            prosody_features["intonation_and_stress"]["score"] +
            prosody_features["chunking_and_rhythm"]["score"]
        ) / 3,
        1
    )

    available_scores = [s for s in [fluency_score, lexical_score, grammar_score, pronunciation_score] if s > 0]
    band_score = round(sum(available_scores) / len(available_scores), 1) if available_scores else 0

    return {
        "transcript": result["text"],
        "words": words_data,
        "alignment": result.get("alignment", {}),
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
                "speaking_speed": speaking_metrics["speaking_speed"],
                "fluency_features": speaking_metrics.get("fluency_features", {})
            },
            "lexical_resource": {
                "llm": llm_analysis.get("lexical_resource", {
                    "vocabulary_range": {"score": 0, "feedback": "Analysis failed"},
                    "accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "idiomatic_language": {"score": 0, "feedback": "Analysis failed"},
                    "vocabulary_mistakes": [],
                    "cefr_level": "Unknown"
                }),
                "local": local_lexical
            },
            "grammar": {
                "llm": llm_analysis.get("grammar", {
                    "range_of_structures": {"score": 0, "feedback": "Analysis failed"},
                    "grammar_accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "tense_accuracy": {"score": 0, "feedback": "Analysis failed"},
                    "errors": []
                }),
                "local": local_grammar
            },
            "pronunciation": {
                "clarity": pronunciation_clarity["clarity"],
                "word_counts": pronunciation_clarity["word_counts"],
                "unclear_words": pronunciation_clarity["unclear_words"],
                "intonation_and_stress": prosody_features["intonation_and_stress"],
                "chunking_and_rhythm": prosody_features["chunking_and_rhythm"]
            },
            "overall": {
                "fluency_and_coherence_score": fluency_score,
                "lexical_resource_score": lexical_score,
                "grammar_score": grammar_score,
                "pronunciation_score": pronunciation_score,
                "band_score": band_score,
                "summary": "Hybrid scoring complete using deterministic features + LLM rubric analysis."
            },
            "filler_words": llm_analysis.get("filler_words", {"count": 0, "words": [], "impact": "Not analyzed"}),
            "low_confidence_words": low_confidence_words,
            "average_confidence": avg_confidence
        }
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Speech Analysis API",
        "endpoints": {
            "/analyzeSpeech": "Transcribe and analyze speech from audio URL",
            "/analyzeSpeechFile": "Transcribe and analyze speech from uploaded file",
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
    ),
    question: Optional[str] = Query(
        default=None,
        description="IELTS speaking prompt/question the speaker is responding to (optional)."
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
        result = await transcription_service.transcribe_from_url(url)
        return await run_in_threadpool(build_analysis_response, result, question)

    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/analyzeSpeechFile")
async def analyze_speech_file(
    file: UploadFile = File(...),
    question: Optional[str] = Form(
        default=None,
        description="IELTS speaking prompt/question the speaker is responding to (optional).",
    ),
):
    """
    Transcribe and analyze speech from an uploaded MP3/audio file.
    """
    try:
        suffix = ".mp3"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1].lower()

        result = await transcription_service.transcribe_from_upload_file(file, suffix=suffix)
        return await run_in_threadpool(build_analysis_response, result, question)

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
