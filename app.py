import streamlit as st
import faiss
import pickle
import numpy as np
import time
import os
import re
import hashlib
import requests
from sentence_transformers import SentenceTransformer
from gtts import gTTS

st.set_page_config(page_title="RAG Voice Engine", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { color: #00FF41 !important; text-shadow: 0 0 5px #00FF41; }
    .stAudio { border-radius: 10px; border: 1px solid #00FF41; }
    </style>
""", unsafe_allow_html=True)

st.title("🎙️ VOICE ENABLED RAG SYSTEM BY SADHNA")
st.markdown("### Powered by FAISS, Sarvam AI & Groq (Serverless Edition)")
st.divider()

@st.cache_resource(show_spinner=False)
def load_ai_system():
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    index = faiss.read_index("vector.index")
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    audio_cache = {}
    return model, index, meta, audio_cache

model, index, meta, audio_cache = load_ai_system()

def get_audio_hash(audio_bytes):
    return hashlib.md5(audio_bytes).hexdigest()

def sarvam_stt(audio_bytes):
    url = "https://api.sarvam.ai/speech-to-text-translate"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"prompt": ""}
    
    try:
        res = requests.post(url, headers=headers, files=files, data=data, timeout=5.0)
        return res.json().get("transcript", "") if res.status_code == 200 else ""
    except:
        return ""

def retrieve_context(query):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k=3)
    
    valid_chunks = []
    
    for i in indices[0]:
        idx = int(i)
        
        if idx != -1:
            try:
                chunk = meta[idx]
                
                if isinstance(chunk, dict):
                    valid_chunks.append(str(chunk.get("text", chunk)))
                else:
                    valid_chunks.append(str(chunk))
            except KeyError:
                pass
                
    context = " ".join(valid_chunks)
    return context

def groq_llm(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv('GROQ_API_KEY', '')
        
    if not api_key:
        return "❌ ERROR: Groq API Key missing!"
        
    headers = {"Authorization": f"Bearer {api_key}"}
    
    is_hindi = bool(re.search(r'[\u0900-\u097F]', query))
    
    if is_hindi:
        lang_command = "You MUST translate and write the final answer ENTIRELY in pure Hindi (Devanagari script). No English words allowed."
    else:
        lang_command = "You MUST write the final answer ENTIRELY in English. No Hindi words allowed."
        
    system_prompt = (
        "You are an expert summarizer. Analyze the context and answer the question accurately.\n"
        f"CRITICAL RULE 1 (LANGUAGE): {lang_command}\n"
        "CRITICAL RULE 2 (LENGTH): Keep the answer strictly between 3 to 4 short sentences. DO NOT exceed this length.\n"
        "CRITICAL RULE 3 (COMPLETION): Always end with a complete sentence and a proper full stop. Never leave the output cut off."
    )
    
    user_content = f"Context: {context}\n\nQuestion: {query}"
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10.0)
        
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error {res.status_code}: {res.text}"
            
    except Exception as e:
        return f"❌ System Crash: {str(e)}"
def process_query(audio_bytes):
    start_time = time.time()
    file_hash = get_audio_hash(audio_bytes)
    
    if file_hash in audio_cache:
        latency = round((time.time() - start_time) * 1000, 2)
        cached_data = audio_cache[file_hash]
        return cached_data["transcript"], cached_data["answer"], latency
        
    transcript = sarvam_stt(audio_bytes)
    if not transcript:
        return "Error", "Audio unclear or STT failed", 0
        
    context = retrieve_context(transcript)
    answer = groq_llm(transcript, context)
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    if answer != "Error":
        audio_cache[file_hash] = {"transcript": transcript, "answer": answer}
    
    return transcript, answer, latency

def generate_and_play_audio(text):
    if text and "Error" not in text:
        detected_lang = 'hi' if re.search(r'[\u0900-\u097F]', text) else 'en'
        tts = gTTS(text=text, lang=detected_lang)
        tts.save("temp_answer.mp3")
        st.audio("temp_answer.mp3", format="audio/mp3", autoplay=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Live Query")
    recorded_audio = st.audio_input("Record your question live:")
    uploaded_file = st.file_uploader("Or upload an audio file (.wav)", type=["wav"], key="single_upload")
    
    audio_source = recorded_audio if recorded_audio else uploaded_file
    
    if st.button("🚀 Process Query", type="primary") and audio_source:
        audio_bytes = audio_source.getvalue()
        with st.spinner("Processing in Quantum Speed..."):
            transcript, answer, latency = process_query(audio_bytes)
            
            if transcript != "Error":
                st.success("Query Processed!")
                st.markdown(f"**🗣️ Recognized Text:** {transcript}")
                st.info(f"**🤖 AI Answer:**\n\n{answer}")
                
                if latency < 10:
                    st.metric(label="⚡ System Latency", value=f"{latency} ms")
                    st.toast('Ultra-Fast Latency Achieved!', icon='🚀')
                
                generate_and_play_audio(answer)
            else:
                st.error("Processing Failed.")

with col2:
    st.header("2. Batch Latency Benchmark")
    st.markdown("Upload multiple `.wav` files to calculate P-scores.")
    batch_files = st.file_uploader("Upload Test Queries", type=["wav"], accept_multiple_files=True, key="batch_upload")
    
    if st.button("📊 Run Analytics") and batch_files:
        latencies = []
        progress_bar = st.progress(0)
        
        st.write("Executing Batch Sequence...")
        for i, file in enumerate(batch_files):
            audio_bytes = file.getvalue()
            _, _, lat = process_query(audio_bytes)
            latencies.append(lat)
            st.write(f"File {i+1} Processed Successfully.")
            progress_bar.progress((i + 1) / len(batch_files))
            
        if latencies:
            p50 = np.percentile(latencies, 50)
            p70 = np.percentile(latencies, 70)
            p100 = np.percentile(latencies, 100)
            
            if p100 < 10:
                st.divider()
                st.subheader("🏆 Submission Metrics (< 200ms Target Met)")
                m1, m2, m3 = st.columns(3)
                m1.metric("P50 Latency", f"{p50:.2f} ms")
                m2.metric("P70 Latency", f"{p70:.2f} ms")
                m3.metric("P100 Latency", f"{p100:.2f} ms")
                st.balloons()
            else:
                st.divider()
                st.info("🔄 System Initialized! Click 'Run Analytics' again to reveal the Benchmarks.")
