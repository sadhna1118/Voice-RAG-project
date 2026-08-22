<h1 align="center">🎙️ Multilingual Voice-Enabled RAG System By Sadhna ❤️</h1>

<div align="center">
  <h3><a href="https://x.com/hashtag/RAGInGoa" target="_blank">#RAGInGoa</a></h3>
</div>

<div align="center">
  
  [![Live Application](https://img.shields.io/badge/🚀_Play_Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://multilingual-voice-enabled-rag-system-by-sadhna.streamlit.app)
  
</div>

<br>

<div align="center">
  <h3>🛠️ Tech Stack & Tools Used</h3>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Groq_API-F65F20?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/Sarvam_AI-000000?style=for-the-badge" alt="Sarvam AI" />
  <img src="https://img.shields.io/badge/Google_TTS-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="gTTS" />
</div>

<br>

---

## 🌟 Overview

Welcome to the **Voice-Enabled RAG (Retrieval-Augmented Generation) System**, a state-of-the-art conversational AI platform that bridges the gap between voice input and intelligent, context-aware responses. This project allows users to speak their questions (in English or Hindi) and receive highly accurate, contextually relevant answers in the exact same language, natively converted back to a natural voice.

## 🚀 Key Features

*   **🗣️ Seamless Voice Interaction:** Directly speak into your device. The system supports live audio recording directly through the browser for both query processing and direct benchmark testing.
*   **🌐 True Multilingual Support (Pan-India):** Ask a question in *any* major Indian language (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, etc.) or English. The system intelligently detects your language and its native script, generating highly accurate responses and Native Text-to-Speech audio exactly in your chosen language!
*   **🧠 Intelligent RAG Engine:** Powered by **FAISS** and **Sentence Transformers**, the system retrieves relevant context from a curated vector database to ensure responses are grounded, accurate, and free from hallucinations.
*   **⚡ Ultra-Fast Inference:** Utilizes the cutting-edge **Groq Serverless Architecture** running `openai/gpt-oss-20b` for near-instantaneous LLM generation.
*   **🔊 Audio Response Generation:** Converts the AI's intelligent text response back to speech using **gTTS**, providing an end-to-end voice assistant experience.
*   **🎛️ Interactive Audio Player:** Features dynamic UI elements like real-time voice clearing (✖), variable playback speeds (0.5x to 2.0x), and native download support for audio responses.
*   **📊 Batch Latency Benchmarking:** A dedicated mode to record live audio or upload multiple `.wav` files simultaneously and calculate system P-scores (P50, P70, P100) to monitor raw performance.
*   **📱 Progressive Web App (PWA) Ready:** Install the app directly to your mobile home screen. Features custom meta-tags for a native app feel, including a transparent status bar, fullscreen bleed, and responsive fluid design without horizontal scrolling or accidental zoom.
*   **🎨 Dynamic Dual Themes:** Experience two entirely unique aesthetic modes seamlessly toggled via a smart Sun/Moon switch:
    *   **Dark Mode (Matrix Hacker):** Immersive neon green terminal vibes with dynamic rotating radar sweeps and scanlines.
    *   **Light Mode (Magical Dreamy):** A vibrant, pinkish-purple glassmorphic aesthetic with pastel text glows and clean, responsive layouts.

## ⚙️ How It Works

1.  **Voice Input:** User records a live audio query using the intuitive Streamlit audio interface.
2.  **Speech-to-Text:** The audio is sent to **Sarvam AI's STT model (`saaras:v3`)**, which accurately transcribes the query into text, maintaining the original language semantics.
3.  **Context Retrieval:** The transcribed query is vectorized using a Hugging Face `SentenceTransformer` and matched against a local **FAISS** index to retrieve top contextual documents.
4.  **AI Generation:** The context and query are passed to **Groq LLM**, instructed via an advanced prompt to deliver a concise, 3-4 sentence answer in the user's native language.
5.  **Text-to-Speech:** The final response is synthesized into an MP3 file, ready for the user to listen to, download, or adjust playback speed.

---

<div align="center">
  <p>Crafted with ❤️ and Innovation</p>
  <p>&copy; 2026 <a href="https://github.com/sadhna1118" target="_blank">Sadhna</a><br> All Rights Reserved.</p>
</div>
