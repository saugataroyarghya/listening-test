from fastapi.testclient import TestClient
from main import app
from services import WhisperService
import pytest

client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint returns correct structure"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "whisper_loaded" in data
    assert "groq_configured" in data


def test_root_endpoint():
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_analyze_speech_endpoint_structure():
    """Test analyzeSpeech endpoint - may return 200 or 500 depending on API key"""
    response = client.get("/analyzeSpeech")
    # If no API key is set, this might return 500
    assert response.status_code in [200, 500]


def test_analyze_speech_file_requires_file():
    response = client.post("/analyzeSpeechFile")
    assert response.status_code == 422


def test_analyze_speech_file_simplified_requires_file():
    response = client.post("/analyzeSpeechFileSimplified")
    assert response.status_code == 422


def test_analyze_speech_file_endpoint_structure(monkeypatch):
    async def mock_transcribe_from_upload_file(file, suffix=".mp3"):
        return {
            "text": "I think this is a good example for testing.",
            "annotated": "I(0.99) think(0.98) this(0.97) is(0.97) a(0.96) good(0.96) example(0.95) for(0.95) testing(0.94)",
            "words": [
                {"word": "I", "confidence": 0.99, "start": 0.0, "end": 0.1},
                {"word": "think", "confidence": 0.98, "start": 0.12, "end": 0.4},
                {"word": "this", "confidence": 0.97, "start": 0.43, "end": 0.55},
                {"word": "is", "confidence": 0.97, "start": 0.58, "end": 0.67},
                {"word": "a", "confidence": 0.96, "start": 0.7, "end": 0.76},
                {"word": "good", "confidence": 0.96, "start": 0.79, "end": 0.95},
                {"word": "example", "confidence": 0.95, "start": 0.99, "end": 1.32},
                {"word": "for", "confidence": 0.95, "start": 1.36, "end": 1.48},
                {"word": "testing", "confidence": 0.94, "start": 1.52, "end": 1.9},
            ],
            "alignment": {"adjusted_words": 0, "overlap_fixes": 0, "duration_fixes": 0, "micro_gap_fixes": 0}
        }

    captured = {"question": None}

    def mock_get_ielts_analysis(transcript, annotated, question=None, speaking_metrics=None, feature_context=None):
        captured["question"] = question
        return {
            "fluency_and_coherence": {
                "fluency": {"score": 7, "feedback": "Good flow."},
                "topic_development": {"score": 7, "feedback": "Relevant points."},
                "cohesive_devices": {"score": 7, "feedback": "Adequate linking.", "devices_used": ["for example"]}
            },
            "lexical_resource": {
                "vocabulary_range": {"score": 7, "feedback": "Decent range."},
                "accuracy": {"score": 7, "feedback": "Mostly accurate."},
                "idiomatic_language": {"score": 6, "feedback": "Limited idioms."},
                "vocabulary_mistakes": [],
                "cefr_level": "B2"
            },
            "grammar": {
                "range_of_structures": {"score": 7, "feedback": "Some complexity."},
                "grammar_accuracy": {"score": 7, "feedback": "Mostly correct."},
                "tense_accuracy": {"score": 7, "feedback": "Consistent tense."},
                "errors": []
            },
            "filler_words": {"count": 0, "words": [], "impact": "Minimal"},
            "overall_fluency_coherence_score": 7,
            "overall_lexical_score": 7,
            "overall_grammar_score": 7
        }

    monkeypatch.setattr("main.transcription_service.transcribe_from_upload_file", mock_transcribe_from_upload_file)
    monkeypatch.setattr("main.groq_service.get_ielts_analysis", mock_get_ielts_analysis)

    question = "Describe a time you learned something important."
    response = client.post(
        "/analyzeSpeechFile",
        files={"file": ("sample.mp3", b"fake-mp3-bytes", "audio/mpeg")},
        data={"question": question},
    )
    assert response.status_code == 200
    assert captured["question"] == question
    data = response.json()
    assert "transcript" in data
    assert "words" in data
    assert "analysis" in data
    assert "overall" in data["analysis"]


def test_analyze_speech_simplified_endpoint(monkeypatch):
    async def mock_transcribe_from_url(url):
        return {
            "text": "Sample transcript from URL.",
            "annotated": "Sample(0.99) transcript(0.98) from(0.97) URL(0.96)",
            "words": [
                {"word": "Sample", "confidence": 0.99, "start": 0.0, "end": 0.2},
                {"word": "transcript", "confidence": 0.98, "start": 0.22, "end": 0.55},
                {"word": "from", "confidence": 0.97, "start": 0.58, "end": 0.7},
                {"word": "URL", "confidence": 0.96, "start": 0.72, "end": 0.9},
            ],
            "alignment": {"adjusted_words": 0, "overlap_fixes": 0, "duration_fixes": 0, "micro_gap_fixes": 0}
        }

    def mock_build_analysis_response(result, question=None):
        return {
            "transcript": result["text"],
            "words": result["words"],
            "alignment": result["alignment"],
            "analysis": {"overall": {"band_score": 7.26}}
        }

    monkeypatch.setattr("main.transcription_service.transcribe_from_url", mock_transcribe_from_url)
    monkeypatch.setattr("main.build_analysis_response", mock_build_analysis_response)

    response = client.get("/analyzeSpeechSimpliied")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"score", "answer_text"}
    assert data["score"] == 7.5
    assert data["answer_text"] == "Sample transcript from URL."


def test_analyze_speech_file_simplified_endpoint(monkeypatch):
    async def mock_transcribe_from_upload_file(file, suffix=".mp3"):
        return {
            "text": "Uploaded speech transcript.",
            "annotated": "Uploaded(0.99) speech(0.98) transcript(0.97)",
            "words": [
                {"word": "Uploaded", "confidence": 0.99, "start": 0.0, "end": 0.28},
                {"word": "speech", "confidence": 0.98, "start": 0.3, "end": 0.5},
                {"word": "transcript", "confidence": 0.97, "start": 0.52, "end": 0.85},
            ],
            "alignment": {"adjusted_words": 0, "overlap_fixes": 0, "duration_fixes": 0, "micro_gap_fixes": 0}
        }

    def mock_build_analysis_response(result, question=None):
        return {
            "transcript": result["text"],
            "words": result["words"],
            "alignment": result["alignment"],
            "analysis": {"overall": {"band_score": 6.0}}
        }

    monkeypatch.setattr("main.transcription_service.transcribe_from_upload_file", mock_transcribe_from_upload_file)
    monkeypatch.setattr("main.build_analysis_response", mock_build_analysis_response)

    response = client.post(
        "/analyzeSpeechFileSimplified",
        files={"file": ("sample.mp3", b"fake-mp3-bytes", "audio/mpeg")},
        data={"question": "Any question"},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"score", "answer_text"}
    assert data["score"] == 6.0
    assert data["answer_text"] == "Uploaded speech transcript."


def test_speaking_metrics_calculation():
    """Test the speaking metrics calculation from word timestamps"""
    ws = WhisperService(load_model=False)

    # Test with sample word data
    test_words = [
        {'word': 'Hello', 'confidence': 0.95, 'start': 0.0, 'end': 0.5},
        {'word': 'my', 'confidence': 0.92, 'start': 0.55, 'end': 0.7},
        {'word': 'name', 'confidence': 0.98, 'start': 0.75, 'end': 1.1},
        {'word': 'is', 'confidence': 0.91, 'start': 1.15, 'end': 1.3},
        {'word': 'John', 'confidence': 0.88, 'start': 3.0, 'end': 3.5},  # 1.7s pause
        {'word': 'and', 'confidence': 0.95, 'start': 3.55, 'end': 3.7},
        {'word': 'I', 'confidence': 0.99, 'start': 3.75, 'end': 3.85},
        {'word': 'am', 'confidence': 0.94, 'start': 3.9, 'end': 4.1},
        {'word': 'a', 'confidence': 0.92, 'start': 4.15, 'end': 4.25},
        {'word': 'developer', 'confidence': 0.87, 'start': 4.3, 'end': 5.0},
    ]

    metrics = ws.calculate_speaking_metrics(test_words)

    # Check speaking speed
    assert "speaking_speed" in metrics
    assert metrics["speaking_speed"]["total_words"] == 10
    assert metrics["speaking_speed"]["words_per_minute"] == 120
    assert metrics["speaking_speed"]["total_duration_seconds"] == 5.0

    # Check pause detection
    assert "pauses" in metrics
    assert metrics["pauses"]["bad_pause_count"] == 1
    assert metrics["pauses"]["bad_pauses"][0]["after_word"] == "is"
    assert metrics["pauses"]["bad_pauses"][0]["duration"] == 1.7
    assert "fluency_features" in metrics
    assert metrics["fluency_features"]["mean_length_of_run_words"] > 0


def test_speaking_metrics_empty_input():
    """Test speaking metrics with empty or minimal input"""
    ws = WhisperService(load_model=False)

    # Empty list
    metrics = ws.calculate_speaking_metrics([])
    assert metrics["speaking_speed"]["words_per_minute"] == 0
    assert metrics["speaking_speed"]["assessment"] == "Insufficient data"

    # Single word
    metrics = ws.calculate_speaking_metrics([
        {'word': 'Hello', 'confidence': 0.95, 'start': 0.0, 'end': 0.5}
    ])
    assert metrics["speaking_speed"]["assessment"] == "Insufficient data"


def test_pronunciation_clarity_calculation():
    """Test pronunciation clarity based on Whisper confidence scores"""
    ws = WhisperService(load_model=False)

    # Test with mixed confidence scores
    test_words = [
        {'word': 'Hello', 'confidence': 0.95, 'start': 0.0, 'end': 0.5},      # clear
        {'word': 'my', 'confidence': 0.92, 'start': 0.55, 'end': 0.7},        # clear
        {'word': 'name', 'confidence': 0.75, 'start': 0.75, 'end': 1.1},      # acceptable
        {'word': 'is', 'confidence': 0.88, 'start': 1.15, 'end': 1.3},        # clear
        {'word': 'pronunciation', 'confidence': 0.55, 'start': 1.35, 'end': 2.0},  # unclear
        {'word': 'um', 'confidence': 0.30, 'start': 2.1, 'end': 2.3},         # filler - excluded
        {'word': 'test', 'confidence': 0.90, 'start': 2.4, 'end': 2.7},       # clear
    ]

    result = ws.calculate_pronunciation_clarity(test_words)

    # Check structure
    assert "clarity" in result
    assert "word_counts" in result
    assert "unclear_words" in result

    # Check counts (excluding 'um' filler word)
    assert result["word_counts"]["total_assessed"] == 6
    assert result["word_counts"]["clear"] == 4  # Hello, my, is, test
    assert result["word_counts"]["acceptable"] == 1  # name
    assert result["word_counts"]["unclear"] == 1  # pronunciation

    # Check unclear words list
    assert len(result["unclear_words"]) == 1
    assert result["unclear_words"][0]["word"] == "pronunciation"

    # Check score is reasonable (5 out of 6 clear/acceptable = ~83%)
    assert result["clarity"]["score"] >= 5
    assert result["clarity"]["clarity_percentage"] > 80


def test_pronunciation_clarity_empty_input():
    """Test pronunciation clarity with empty input"""
    ws = WhisperService(load_model=False)

    result = ws.calculate_pronunciation_clarity([])
    assert result["clarity"]["score"] == 0
    assert result["clarity"]["feedback"] == "Insufficient data"


def test_alignment_refinement_fixes_overlaps():
    ws = WhisperService(load_model=False)
    raw_words = [
        {"word": "I", "confidence": 0.9, "start": 0.0, "end": 0.2},
        {"word": "think", "confidence": 0.9, "start": 0.18, "end": 0.18},
        {"word": "so", "confidence": 0.9, "start": 0.21, "end": 0.26},
    ]
    refined = ws.refine_word_alignment(raw_words)
    refined_words = refined["words"]
    assert refined["alignment"]["adjusted_words"] > 0
    assert refined_words[1]["start"] >= refined_words[0]["end"]
    assert refined_words[1]["end"] > refined_words[1]["start"]


def test_prosody_lexical_grammar_features_exist():
    ws = WhisperService(load_model=False)
    sample_words = [
        {"word": "I", "confidence": 0.95, "start": 0.0, "end": 0.12},
        {"word": "believe", "confidence": 0.93, "start": 0.14, "end": 0.46},
        {"word": "that", "confidence": 0.91, "start": 0.5, "end": 0.62},
        {"word": "technology", "confidence": 0.9, "start": 0.65, "end": 1.05},
        {"word": "improves", "confidence": 0.88, "start": 1.4, "end": 1.72},
        {"word": "education", "confidence": 0.9, "start": 1.76, "end": 2.15},
    ]
    transcript = "I believe that technology improves education because students can learn faster."

    prosody = ws.calculate_prosody_features(sample_words)
    lexical = ws.calculate_lexical_resource(transcript, sample_words)
    grammar = ws.calculate_grammar_analysis(transcript)

    assert "intonation_and_stress" in prosody
    assert prosody["intonation_and_stress"]["score"] >= 0
    assert "overall_lexical_score" in lexical
    assert lexical["cefr_level"] in {"A1", "A2", "B1", "B2", "C1", "C2", "Unknown"}
    assert "overall_grammar_score" in grammar
    assert "errors" in grammar


def test_lexical_deterministic_cefr_metadata():
    ws = WhisperService(load_model=False)
    transcript = (
        "In my opinion, technology and education are important because they improve communication "
        "and create opportunities for society."
    )
    lexical = ws.calculate_lexical_resource(transcript)
    assert "deterministic_confidence" in lexical
    assert 0 <= lexical["deterministic_confidence"] <= 1
    assert lexical["local_metrics"]["engine"] == "cefr_lexicon"
    assert "cefr_coverage" in lexical["local_metrics"]


def test_grammar_deterministic_metadata():
    ws = WhisperService(load_model=False)
    grammar = ws.calculate_grammar_analysis("He are late and I is tired.")
    assert "deterministic_confidence" in grammar
    assert 0 <= grammar["deterministic_confidence"] <= 1
    assert "engine" in grammar["local_metrics"]
    assert any(err["type"] == "grammar" for err in grammar["errors"])
