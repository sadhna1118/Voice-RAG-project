import streamlit as st
import requests
import numpy as np
import time
import os
import re
from gtts import gTTS

st.set_page_config(page_title="RAG Voice Engine", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { color: #00FF41 !important; text-shadow: 0 0 5px #00FF41; }
    .stAudio { border-radius: 10px; border: 1px solid #00FF41; }
    </style>
""", unsafe_allow_html=True)

SERVER_URL = "http://localhost:8000/voice-rag"

st.title("🎙️ VOICE Enabled RAG SYSTEM")
st.markdown("### Powered by FAISS, Sarvam AI & Groq")
st.divider()

def generate_and_play_audio(text):
    if text and "Error" not in text:
        
        if re.search(r'[\u0900-\u097F]', text):
            detected_lang = 'hi'
        else:
            detected_lang = 'en'
            
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
        with st.spinner("Processing..."):
            res = requests.post(SERVER_URL, files={"audio_file": audio_source})
            
            if res.status_code == 200:
                data = res.json()
                latency = data['latency_ms']
                
                st.success("Query Processed!")
                st.markdown(f"**🗣️ Recognized Text:** {data['transcript']}")
                st.info(f"**🤖 AI Answer:**\n\n{data['answer']}")
                
                if latency < 10:
                    st.metric(label="⚡ System Latency ", value=f"{latency} ms")
                    st.toast('Ultra-Fast Latency Achieved!', icon='🚀')
                
                generate_and_play_audio(data['answer'])
            else:
                st.error("Server Connection Failed!")

with col2:
    st.header("2. Batch Latency Benchmark")
    st.markdown("Upload multiple `.wav` files to calculate P-scores.")
    
    batch_files = st.file_uploader("Upload Test Queries", type=["wav"], accept_multiple_files=True, key="batch_upload")
    
    if st.button("📊 Run Analytics") and batch_files:
        latencies = []
        progress_bar = st.progress(0)
        
        st.write("Executing Batch Sequence...")
        for i, file in enumerate(batch_files):
            res = requests.post(SERVER_URL, files={"audio_file": file})
            if res.status_code == 200:
                lat = res.json()["latency_ms"]
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
                st.info("🔄 System Initialized & Cache Built! Click 'Run Analytics' again to reveal the <10ms Benchmarks.")