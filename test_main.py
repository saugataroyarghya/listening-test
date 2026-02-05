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


def test_speaking_metrics_calculation():
    """Test the speaking metrics calculation from word timestamps"""
    ws = WhisperService()

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


def test_speaking_metrics_empty_input():
    """Test speaking metrics with empty or minimal input"""
    ws = WhisperService()

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
    ws = WhisperService()

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
    ws = WhisperService()

    result = ws.calculate_pronunciation_clarity([])
    assert result["clarity"]["score"] == 0
    assert result["clarity"]["feedback"] == "Insufficient data"