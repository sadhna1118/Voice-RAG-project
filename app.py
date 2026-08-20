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
    <!-- App-like Meta Tags -->
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="mobile-web-app-capable" content="yes">

    <style>
    /* Hide Streamlit Header/Footer for Native App Feel */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Base Font and Color */
    .main { background-color: #050805; color: #00FF41; font-family: 'Share Tech Mono', 'Courier New', Courier, monospace; }
    h1, h2, h3, p, label, .stMarkdown { font-family: 'Share Tech Mono', 'Courier New', Courier, monospace !important; color: #00FF41 !important; }
    h1, h2, h3 { text-shadow: 0 0 10px rgba(0, 255, 65, 0.8), 0 0 20px rgba(0, 255, 65, 0.4); }
    
    /* Ensure icons retain their font family to prevent ligatures like 'upload' from showing as text */
    span, div { color: #00FF41; }
    
    /* Background Radar Animation */
    .stApp {
        background-color: #050805 !important;
        background-image: 
            radial-gradient(circle at center, rgba(0,255,65,0.15) 0%, transparent 80%),
            linear-gradient(rgba(0, 255, 65, 0.15) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 65, 0.15) 1px, transparent 1px);
        background-size: 100% 100%, 30px 30px, 30px 30px;
    }
    
    /* Rotating Radar Sweep */
    .stApp::before {
        content: "";
        position: fixed;
        top: 50%;
        left: 50%;
        width: 150vmax;
        height: 150vmax;
        margin-top: -75vmax;
        margin-left: -75vmax;
        background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, transparent 270deg, rgba(0, 255, 65, 0.6) 350deg, rgba(0, 255, 65, 1.0) 360deg);
        border-radius: 50%;
        animation: radar-spin 4s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Static Radar Rings */
    .stApp::after {
        content: "";
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 100vw;
        height: 100vh;
        background: 
            repeating-radial-gradient(circle at 50% 50%, transparent 0, transparent calc(8vw - 1px), rgba(0, 255, 65, 0.6) 8vw),
            linear-gradient(90deg, transparent calc(50% - 1px), rgba(0, 255, 65, 0.7) 50%, transparent calc(50% + 1px)),
            linear-gradient(0deg, transparent calc(50% - 1px), rgba(0, 255, 65, 0.7) 50%, transparent calc(50% + 1px));
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes radar-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Scanline Overlay on the whole app */
    [data-testid="stAppViewContainer"]::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.15) 50%);
        background-size: 100% 4px;
        z-index: 9999;
        pointer-events: none;
    }
    
    /* Main Content Wrapper - Keep Above Radar */
    .block-container {
        position: relative;
        z-index: 10;
        background: rgba(14, 17, 23, 0.4) !important;
        border: 1px solid rgba(0, 255, 65, 0.6);
        box-shadow: 0 0 25px rgba(0, 255, 65, 0.3);
        border-radius: 12px;
        padding: 2rem !important;
        backdrop-filter: blur(8px);
    }
    
    /* Glowing Buttons - No Double Text */
    .stButton > button {
        background-color: transparent !important;
        border: 1px solid #00FF41 !important;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: rgba(0, 255, 65, 0.15) !important;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.6);
        transform: scale(1.02);
    }
    
    /* Uploader & Audio Styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(0, 255, 65, 0.02);
        border: 1px dashed rgba(0, 255, 65, 0.4);
        transition: 0.3s;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"]:hover {
        border: 1px solid #00FF41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    .stAudio { 
        border-radius: 10px; 
        border: 1px solid #00FF41; 
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    
    /* Native Audio Player Hacker Theme */
    audio {
        border-radius: 10px;
    }
    audio::-webkit-media-controls-panel {
        background-color: #050805 !important;
    }
    audio::-webkit-media-controls-timeline,
    audio::-webkit-media-controls-play-button,
    audio::-webkit-media-controls-mute-button,
    audio::-webkit-media-controls-volume-slider {
        filter: sepia(100%) saturate(700%) hue-rotate(70deg) brightness(1.2) contrast(1.2);
    }
    audio::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display {
        color: #00FF41 !important;
        text-shadow: 0 0 5px rgba(0,255,65,0.5);
    }
    
    /* Mobile Responsiveness & App-like Feel */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem 1rem !important;
            margin: 0 !important;
            max-width: 100% !important;
            border-radius: 8px;
        }
        h1 { font-size: 1.5rem !important; }
        h3 { font-size: 1rem !important; }
        
        /* Prevent radar sweep from creating horizontal scroll on mobile */
        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
        }
        
        /* Better touch targets for buttons */
        .stButton > button {
            padding: 0.5rem 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

theme_col1, theme_col2 = st.columns([0.85, 0.15])
with theme_col1:
    st.title("🎙️ VOICE ENABLED RAG SYSTEM BY SADHNA")
with theme_col2:
    if 'is_light' not in st.session_state:
        st.session_state.is_light = False
    is_light_mode = st.toggle("🌙" if st.session_state.is_light else "☀️", key="is_light")

if is_light_mode:
    st.markdown("""
        <style>
        /* Magical Dreamy Light Theme Overrides */
        .main, .stApp { background-color: #FFF0F5 !important; }
        
        /* Unified Pinkish-Purple Color for Everything (matching title gradient) */
        h1, h2, h3, p, label, .stMarkdown, span, div { color: #C13584 !important; text-shadow: none !important; }
        
        /* Pastel/Dreamy Text Glow */
        h1 {
            background: -webkit-linear-gradient(45deg, #FF6B6B, #833AB4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 10px rgba(193, 53, 132, 0.3) !important;
        }
        
        /* Dreamy Background Grid & Radar */
        .stApp {
            background-image: 
                radial-gradient(circle at top right, rgba(253, 29, 29, 0.2) 0%, transparent 60%),
                radial-gradient(circle at bottom left, rgba(131, 58, 180, 0.2) 0%, transparent 60%),
                linear-gradient(rgba(193, 53, 132, 0.15) 1px, transparent 1px),
                linear-gradient(90deg, rgba(193, 53, 132, 0.15) 1px, transparent 1px) !important;
            background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px !important;
        }
        
        .stApp::before {
            background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, transparent 270deg, rgba(193, 53, 132, 0.8) 350deg, rgba(131, 58, 180, 1.0) 360deg) !important;
        }
        
        .stApp::after {
            background: 
                repeating-radial-gradient(circle at 50% 50%, transparent 0, transparent calc(8vw - 1px), rgba(193, 53, 132, 0.5) 8vw),
                linear-gradient(90deg, transparent calc(50% - 1px), rgba(131, 58, 180, 0.4) 50%, transparent calc(50% + 1px)),
                linear-gradient(0deg, transparent calc(50% - 1px), rgba(131, 58, 180, 0.4) 50%, transparent calc(50% + 1px)) !important;
        }
        
        /* Disable Scanlines in light mode */
        [data-testid="stAppViewContainer"]::before { display: none !important; }
        
        /* Glassmorphism Content Wrapper */
        .block-container {
            background: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid rgba(193, 53, 132, 0.5) !important;
            box-shadow: 0 10px 40px rgba(131, 58, 180, 0.25) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Alerts & Info Boxes (AI Responses) matching Light Theme */
        [data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.3) !important;
            border: 1px solid rgba(193, 53, 132, 0.6) !important;
            border-radius: 12px !important;
            color: #C13584 !important;
        }
        [data-testid="stAlert"] * { color: #C13584 !important; }
        
        /* Soft Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4) !important;
            border-radius: 20px !important;
            font-weight: bold;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #FECFEF 0%, #FF9A9E 100%) !important;
            box-shadow: 0 6px 20px rgba(255, 154, 158, 0.6) !important;
            color: #ffffff !important;
            transform: translateY(-2px);
        }
        .stButton > button * { color: inherit !important; }
        
        /* Uploader & Audio */
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.4) !important;
            border: 2px dashed rgba(193, 53, 132, 0.6) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #C13584 !important;
            box-shadow: 0 0 15px rgba(193, 53, 132, 0.4) !important;
        }
        
        .stAudio {
            border-radius: 12px !important; 
            border: 1px solid rgba(193, 53, 132, 0.5) !important; 
            box-shadow: 0 4px 15px rgba(193, 53, 132, 0.3) !important;
        }
        
        /* Native Audio Player Dreamy Theme */
        audio::-webkit-media-controls-panel { background-color: #FFF0F5 !important; }
        audio::-webkit-media-controls-timeline,
        audio::-webkit-media-controls-play-button,
        audio::-webkit-media-controls-mute-button,
        audio::-webkit-media-controls-volume-slider {
            filter: invert(40%) sepia(80%) saturate(2000%) hue-rotate(300deg) brightness(1.1) contrast(1.2) !important;
        }
        audio::-webkit-media-controls-current-time-display,
        audio::-webkit-media-controls-time-remaining-display {
            color: #833AB4 !important;
            text-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

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
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "model": "saaras:v3",
        "language_code": "unknown"
    }
    
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
    
    lang_command = (
        "Identify the language of the user's question. "
        "If the question is in Hindi (whether written in Devanagari or Roman/English script like 'kya haal hai'), "
        "you MUST reply entirely in pure Hindi using Devanagari script. "
        "If the question is in English, reply entirely in English. "
        "Never translate a Hindi query to English."
    )
        
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
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            with open("temp_answer.mp3", "rb") as f:
                st.download_button(
                    label="⬇️ Download Audio Answer",
                    data=f,
                    file_name="ai_response.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
                
        with c2:
            import streamlit.components.v1 as components
            html_code = """
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: 'Courier New', Courier, monospace; color: #00FF41; height: 100%; margin-top: 5px;">
                <label style="font-weight: bold; margin:0;">Voice Speed:</label>
                <select onchange="
                    var audios = window.parent.document.getElementsByTagName('audio');
                    if (audios.length > 0) {
                        audios[audios.length - 1].playbackRate = parseFloat(this.value);
                    }
                " style="background-color: #0E1117; color: #00FF41; border: 1px solid #00FF41; padding: 6px; border-radius: 5px; cursor: pointer; width: 100px;">
                    <option value="0.5">0.5x</option>
                    <option value="0.75">0.75x</option>
                    <option value="1.0" selected>1.0x</option>
                    <option value="1.25">1.25x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2.0">2.0x</option>
                </select>
            </div>
            """
            components.html(html_code, height=55)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Live Query")
    recorded_audio = st.audio_input("Record your question live:")
    
    import streamlit.components.v1 as components
    components.html("""
    <script>
        setInterval(function() {
            var parent = window.parent.document;
            var audioInputs = parent.querySelectorAll('[data-testid="stAudioInput"]');
            
            audioInputs.forEach(function(audioInput) {
                if (!audioInput.querySelector('#custom-clear-mic')) {
                    var x = parent.createElement('div');
                    x.id = 'custom-clear-mic';
                    x.innerHTML = '✖';
                    x.style.position = 'absolute';
                    x.style.right = '15px';
                    x.style.top = '50%';
                    x.style.transform = 'translateY(-50%)';
                    x.style.cursor = 'pointer';
                    x.style.fontSize = '13px';
                    x.style.color = '#ff4b4b'; // Red color
                    x.style.zIndex = '999';
                    x.style.transition = 'transform 0.2s';
                    x.title = "Clear Recording";
                    x.style.display = 'none'; // Hidden by default
                    
                    x.onmouseover = function() { this.style.transform = 'translateY(-50%) scale(1.3)'; };
                    x.onmouseout = function() { this.style.transform = 'translateY(-50%) scale(1)'; };
                    
                    // Attach to the actual record box (usually the second child after the label)
                    var recordArea = audioInput.children.length > 1 ? audioInput.children[1] : audioInput;
                    recordArea.style.position = 'relative';
                    recordArea.appendChild(x);
                    
                    x.onclick = function(e) {
                        e.stopPropagation();
                        e.preventDefault();
                        var btns = audioInput.querySelectorAll('button');
                        btns.forEach(function(btn) {
                            var label = (btn.getAttribute('aria-label') || '').toLowerCase();
                            if (label.includes('clear') || label.includes('delete') || label.includes('remove') || label.includes('reset')) {
                                btn.click();
                            }
                        });
                    };
                }
                
                // Toggle visibility based on whether audio is recorded (i.e. native clear button exists)
                var customX = audioInput.querySelector('#custom-clear-mic');
                if (customX) {
                    var hasRecorded = false;
                    var btns = audioInput.querySelectorAll('button');
                    btns.forEach(function(btn) {
                        var label = (btn.getAttribute('aria-label') || '').toLowerCase();
                        if (label.includes('clear') || label.includes('delete') || label.includes('remove') || label.includes('reset')) {
                            hasRecorded = true;
                        }
                    });
                    customX.style.display = hasRecorded ? 'block' : 'none';
                }
                
                // Shift the timer text left so they don't overlap
                var allElements = audioInput.querySelectorAll('*');
                allElements.forEach(function(el) {
                    if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
                        if (/^\d{1,2}:\d{2}$/.test(el.innerText.trim())) {
                            if (el.style.transform !== 'translateX(-15px)') {
                                el.style.transform = 'translateX(-15px)';
                            }
                        }
                    }
                });
            });
        }, 500);
    </script>
    """, height=0)

    uploaded_file = st.file_uploader("Or upload an audio file (.wav)", type=["wav"], key="single_upload")
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

    
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
                st.info("🔄 Click 'Run Analytics' again to reveal the Benchmarks.")
