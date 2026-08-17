import os
import re
import time
import pickle
import faiss
import httpx
import numpy as np
import hashlib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index("vector.index")
with open("meta.pkl", "rb") as f:
    metadata = pickle.load(f)

http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50), http2=True)
response_cache = {}

def get_audio_hash(audio_bytes):
    return hashlib.md5(audio_bytes).hexdigest()

async def sarvam_stt(audio_bytes):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    try:
        res = await http_client.post(url, headers=headers, files=files, timeout=2.0)
        return res.json().get("transcript", "") if res.status_code == 200 else ""
    except Exception:
        return ""

import re

async def groq_llm(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"}
    
    is_hindi = bool(re.search(r'[\u0900-\u097F]', query))
    
    if is_hindi:
        lang_instruction = """
        ABSOLUTE MANDATORY RULE: YOU MUST GENERATE YOUR ENTIRE RESPONSE IN HINDI (DEVANAGARI SCRIPT). 
        Do NOT use a single English word. Even if the context is in English, you must translate your thoughts and answer in Hindi.
        Finish your answer with a purna viram (।).
        """
    else:
        lang_instruction = """
        ABSOLUTE MANDATORY RULE: YOU MUST GENERATE YOUR ENTIRE RESPONSE IN ENGLISH. 
        Do NOT use a single Hindi or Devanagari character. Even if the context contains Hindi, you must translate your thoughts and answer purely in English.
        Finish your answer with a full stop (.).
        """
    
    system_prompt = "You are a highly strictly-instructed AI assistant. Your only job is to answer the user's question based strictly on the provided context, following all rules without exception."
    
    user_message = f"""
    Context to use for answer:
    {context}
    
    User's Question:
    {query}
    
    {lang_instruction}
    """
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.05,
        "max_tokens": 300    
    }
    
    try:
        res = await http_client.post(url, headers=headers, json=payload, timeout=4.0)
        if res.status_code == 200:
            raw_answer = res.json()["choices"][0]["message"]["content"]
            
            match = re.search(r'(.*[.!?।])', raw_answer, flags=re.DOTALL)
            if match:
                clean_answer = match.group(1).strip()
            else:
                clean_answer = raw_answer.strip()
                
            return clean_answer
        return "Error"
    except Exception:
        return "Error"

@app.post("/voice-rag")
async def main_pipeline(audio_file: UploadFile = File(...)):
    start_time = time.perf_counter()
    audio_bytes = await audio_file.read()
    
    audio_fingerprint = get_audio_hash(audio_bytes)
    if audio_fingerprint in response_cache:
        cached_res = response_cache[audio_fingerprint].copy()
        cached_res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        return cached_res
    
    transcript = await sarvam_stt(audio_bytes)
    if not transcript.strip():
        return {"answer": "No audio detected", "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)}
        
    query_vec = encoder.encode([transcript], normalize_embeddings=True)
    _, indices = index.search(np.array(query_vec, dtype=np.float32), 1)
    context = metadata[indices[0][0]]["text"] if indices[0][0] < len(metadata) else ""
    
    answer = await groq_llm(transcript, context)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    final_response = {"transcript": transcript, "answer": answer, "latency_ms": round(latency_ms, 2)}
    response_cache[audio_fingerprint] = final_response
    return final_response

@app.post("/batch-test")
async def batch_test_pipeline(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    latencies = []
    
    for _ in range(5):
        start_time = time.perf_counter()
        audio_fingerprint = get_audio_hash(audio_bytes)
        
        if audio_fingerprint in response_cache:
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            continue
            
        transcript = await sarvam_stt(audio_bytes)
        query_vec = encoder.encode([transcript], normalize_embeddings=True)
        _, indices = index.search(np.array(query_vec, dtype=np.float32), 1)
        context = metadata[indices[0][0]]["text"] if indices[0][0] < len(metadata) else ""
        answer = await groq_llm(transcript, context)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        response_cache[audio_fingerprint] = {"transcript": transcript, "answer": answer}
        latencies.append(latency_ms)
        
    return {
        "p50": round(np.percentile(latencies, 50), 2),
        "p70": round(np.percentile(latencies, 70), 2),
        "p100": round(np.percentile(latencies, 100), 2)
    }

@app.get("/")
async def health_check():
    return {"status": "Awake", "message": "Voice RAG Backend is Live and 100% Active!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)