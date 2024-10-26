from fastapi import FastAPI
from app import get_answer  # Import get_answer from app.py

app = FastAPI()

@app.post("/query/")
async def query_api(question: str):
    answer = get_answer(question)
    return {"answer": answer}
