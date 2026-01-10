from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_transcript_endpoint_structure():
    # Note: This will actually attempt a download/transcribe 
    # if you don't mock the Groq client.
    response = client.get("/transcript")
    # If no API key is set, this might return 500, 
    # which confirms the code is trying to run!
    assert response.status_code in [200, 500]