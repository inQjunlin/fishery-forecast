from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json

from src.feature_extraction import extract_url_features
from src.llm.llm_interface import llm_explain

app = FastAPI(title="PhishDetector API")

class InputPayload(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None

@app.post("/detect")
def detect(payload: InputPayload):
    features = {}
    if payload.url:
        features = extract_url_features(payload.url)
    # Lightweight heuristic score for demonstration
    score = 0.0
    if payload.url:
        u = features
        if u.get("url_len", 0) > 80:
            score += 0.25
        if u.get("has_https", 0) == 0:
            score += 0.25
        if u.get("has_ip", 0) == 1:
            score += 0.15
        if u.get("n_queries", 0) > 3:
            score += 0.15
        if u.get("contains_http", 0) == 1 and u.get("has_https", 0) == 0:
            score += 0.1
    label = 1 if score >= 0.5 else 0
    explanation = llm_explain(label, features, payload.text or "")
    return {"label": int(label), "score": float(score), "explanation": explanation}
