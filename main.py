from fastapi import FastAPI
from huggingface_hub import InferenceClient
import os

app = FastAPI()

client = InferenceClient(
    model="VoltIC/Automated-Text-Summarizer", 
    token=os.getenv("HF_TOKEN")
)

@app.get("/")
def home():
    return {"message": "Summarizer API is running!"}

@app.post("/summarize")
def summarize(text: str):
    try:
        summary = client.summarization(text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}
