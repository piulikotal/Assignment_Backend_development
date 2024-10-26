from app import get_answer  # Import get_answer from app.py

def test_get_answer():
    question = "What is the content about?"
    assert get_answer(question) is not None  # Check if answer exists
