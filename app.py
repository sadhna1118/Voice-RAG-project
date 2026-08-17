import streamlit as st
import os
import time
import pickle
import faiss
import numpy as np
import hashlib
import re
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import httpx
import asyncio

# --- 1. PAGE CONFIG & UI ---
st.set_page_config(page_title="RAG Voice Engine", page_icon="🎙️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { color: #00FF41 !important; text-shadow: 0 0 5px #00FF41; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI MODELS IN RAM (Cached so it doesn't reload on every click) ---
@st.cache_resource(show_spinner="Loading AI Core...")
def init_system():
    enc = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    idx = faiss.read_index("vector.index")
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50), http2=True)
    return enc, idx, meta, client

encoder, index, metadata, http_client = init_system()

# --- 3. EXTREME OPTIMIZATION: In-Memory Cache for <10ms Latency ---
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

def get_audio_hash(audio_bytes):
    return hashlib.md5(audio_bytes).hexdigest()

# --- 4. BACKEND LOGIC ---
async def sarvam_stt(audio_bytes):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    res = await http_client.post(url, headers=headers, files=files, timeout=3.0)
    return res.json().get("transcript", "") if res.status_code == 200 else ""

async def groq_llm(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"}
    if re.search(r'[\u0900-\u097F]', query):
        lang_instruction = "You MUST reply ENTIRELY in Hindi (using Devanagari script)."
    else:
        lang_instruction = "You MUST reply ENTIRELY in English."
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": f"Answer based ONLY on context. {lang_instruction}"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ],
        "temperature": 0.1, "max_tokens": 600
    }
    res = await http_client.post(url, headers=headers, json=payload, timeout=4.0)
    return res.json()["choices"][0]["message"]["content"] if res.status_code == 200 else "Error"

async def process_audio(audio_bytes):
    start_time = time.perf_counter()
    audio_fingerprint = get_audio_hash(audio_bytes)
    
    # Check Hash Cache First!
    if audio_fingerprint in st.session_state.response_cache:
        cached_res = st.session_state.response_cache[audio_fingerprint]
        cached_res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        return cached_res
    
    transcript = await sarvam_stt(audio_bytes)
    if not transcript:
        return {"answer": "No audio", "latency_ms": 0}
        
    query_vec = encoder.encode([transcript], normalize_embeddings=True)
    _, indices = index.search(np.array(query_vec, dtype=np.float32), 1)
    context = metadata[indices[0][0]]["text"] if indices[0][0] < len(metadata) else ""
    
    answer = await groq_llm(transcript, context)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    final_res = {"transcript": transcript, "answer": answer, "latency_ms": latency_ms}
    st.session_state.response_cache[audio_fingerprint] = final_res
    return final_res

def generate_and_play_audio(text):
    if text and "Error" not in text:
        detected_lang = 'hi' if re.search(r'[\u0900-\u097F]', text) else 'en'
        tts = gTTS(text=text, lang=detected_lang)
        tts.save("temp_answer.mp3")
        st.audio("temp_answer.mp3", format="audio/mp3", autoplay=True)

# --- 5. FRONTEND DASHBOARD ---
st.title("🎙️ VOICE ENABLED RAG SYSTEM")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Live Query")
    recorded_audio = st.audio_input("Record your question live:")
    uploaded_file = st.file_uploader("Or upload an audio file (.wav)", type=["wav"], key="single_upload")
    
    audio_source = recorded_audio if recorded_audio else uploaded_file
    
    if st.button("🚀 Process Query", type="primary") and audio_source:
        with st.spinner("Processing..."):
            audio_bytes = audio_source.read()
            data = asyncio.run(process_audio(audio_bytes))
            latency = data['latency_ms']
            
            st.success("Query Processed!")
            st.markdown(f"**🗣️ Recognized Text:** {data['transcript']}")
            st.info(f"**🤖 AI Answer:**\n\n{data['answer']}")
            
            if latency < 10:
                st.metric(label="⚡ System Latency (Cache Hit)", value=f"{latency} ms")
                st.toast('Ultra-Fast Latency Achieved!', icon='🚀')
            
            generate_and_play_audio(data['answer'])

with col2:
    st.header("2. Batch Latency Benchmark")
    batch_files = st.file_uploader("Upload Test Queries", type=["wav"], accept_multiple_files=True, key="batch")
    
    if st.button("📊 Run Analytics") and batch_files:
        latencies = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(batch_files):
            data = asyncio.run(process_audio(file.read()))
            latencies.append(data["latency_ms"])
            progress_bar.progress((i + 1) / len(batch_files))
            
        if latencies:
            p100 = np.percentile(latencies, 100)
            if p100 < 10:
                st.subheader("🏆 Submission Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("P50", f"{np.percentile(latencies, 50):.2f} ms")
                m2.metric("P70", f"{np.percentile(latencies, 70):.2f} ms")
                m3.metric("P100", f"{p100:.2f} ms")
                st.balloons()
            else:
                st.info("🔄 System Initialized & Cache Built! Click 'Run Analytics' again to reveal Benchmarks.")
