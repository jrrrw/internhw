from main import analyze
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/search")
def search(q: str):
    return analyze(q)
