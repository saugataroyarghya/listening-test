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
    ),
    custom_system_message: str = Query(
        default=None,
        description="Custom system message for GPT analysis (optional)"
    )
):
    """
    Transcribe audio from URL and analyze speech quality.
    
    Returns:
    - transcript: Plain text transcription with filler words
    - annotated_transcript: Text with confidence scores for each word
    - analysis: GPT analysis of speech quality
    """
    try:
        # 1. Transcribe with whisper.cpp
        result = await transcription_service.transcribe_from_url(url)
        
        # 2. Analyze with Groq
        # Use custom system message if provided, otherwise use default
        system_msg = custom_system_message or """You are a speech quality analyst. 

Analyze the transcription and provide a JSON response with:
- fluency_score: A score from 0-100 indicating overall speech fluency
- filler_count: Total number of filler words (um, uh, like, you know, etc.)
- filler_words: List of filler words found
- low_confidence_words: List of words with confidence < 0.7
- average_confidence: Average confidence score across all words
- insights: Brief analysis of speech quality and areas for improvement

Be constructive and specific in your analysis."""

        analysis = groq_service.get_analysis(
            transcript=result["text"],
            annotated=result["annotated"],
            system_message=system_msg
        )

        return {
            "status": "success",
            "url": url,
            "transcript": result["text"],
            "annotated_transcript": result["annotated"],
            "analysis": analysis
        }
    except Exception as e:
        # Better error handling
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