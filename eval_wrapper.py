import time
from sentence_transformers import SentenceTransformer
import os
import requests
import toml
import re

# Explicitly set the retrieval budget for the eval suite, overriding any OS environment variables.
LATENCY_BUDGET_MS = 50.0

class EvalWrapper:
    def __init__(self):
        import torch
        torch.set_num_threads(1) # Optimize PyTorch CPU threading for small batch inference to reduce latency
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.model.encode("warmup text") # Prevent cold-start penalty on first query
    
    def embed(self, texts):
        return self.model.encode(texts).astype('float32')

    def embed_one(self, text):
        return self.model.encode(text).astype('float32')

    def get_model(self):
        return self.model

wrapper = EvalWrapper()

def embed(texts):
    return wrapper.embed(texts)

def embed_one(text):
    return wrapper.embed_one(text)

def get_model():
    return wrapper.get_model()

class AnswerObject:
    def __init__(self, text, grounded, generation_ms, model_name):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model_name

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        api_key = secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = os.getenv('GROQ_API_KEY', '')

def groq_llm(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    if not api_key:
        return "❌ ERROR: Groq API Key missing!"
        
    headers = {"Authorization": f"Bearer {api_key}"}
    
    lang_command = (
        "CRITICAL: Auto-detect the underlying spoken language of the user's question. "
        "Even if the question is typed in Roman/English letters (e.g., 'Bharat ki rajdhani' -> Hindi), detect the actual language. "
        "You MUST translate the retrieved context and reply ENTIRELY in the NATIVE SCRIPT of the user's detected language "
        "(e.g., Devanagari for Hindi, Bengali script for Bengali, etc.). "
        "If the user asks purely in English, reply in English."
    )
        
    system_prompt = (
        "You are an answering assistant. Answer the user's question using ONLY the provided context.\n"
        f"CRITICAL RULE 1 (LANGUAGE & TRANSLATION): {lang_command}\n"
        "CRITICAL RULE 2 (REFUSAL): If the provided context does not contain enough information to answer the question, you must reply with exactly 'NO_ANSWER' and nothing else.\n"
        "CRITICAL RULE 3 (CONCISENESS): If you do answer, keep it strictly to 1 or 2 short sentences.\n"
        "CRITICAL RULE 4 (NO THINKING): DO NOT output any <think> tags or reasoning. Just output the final answer directly.\n\n"
        "Example 1:\nContext: The sky is blue due to Rayleigh scattering.\nQuestion: Aakash neela kyu hota hai?\nAnswer: आकाश रेले प्रकीर्णन (Rayleigh scattering) के कारण नीला होता है।\n\n"
        "Example 2:\nContext: The sky is blue.\nQuestion: Why is grass green?\nAnswer: NO_ANSWER\n"
    )
    
    data = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        "temperature": 0.1,
        "max_tokens": 256
    }
    
    try:
        import time
        max_retries = 3
        for attempt in range(max_retries):
            r = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )
            if r.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "NO_ANSWER"

def generate_answer(query, results):
    context_texts = [res.text for res in results]
    context = " ".join(context_texts)
    
    start = time.time()
    answer_text = groq_llm(query, context)
    
    # Strip <think> tags from models like Qwen (even if cut off)
    answer_text = re.sub(r'<think>.*?(</think>|$)', '', answer_text, flags=re.DOTALL).strip()
    
    generation_ms = (time.time() - start) * 1000
    
    # Check if model refused to answer or produced empty output
    grounded = "NO_ANSWER" not in answer_text and len(answer_text) > 0
    
    return AnswerObject(answer_text, grounded, generation_ms, "qwen/qwen3.6-27b")
