# 🎬 YouTube ChatBot

A Streamlit-based intelligent chatbot that answers questions about any YouTube video using its transcript. It supports multilingual transcripts (English, Hindi, Marathi, German, Russian, etc.) and uses Retrieval-Augmented Generation (RAG) with LangChain, Groq's LLM (Gemma 2B), and HuggingFace embeddings.

---

## 🔍 Features

- 🎥 Extracts transcripts from YouTube videos via video ID
- 🌍 Supports multiple transcript languages (en, hi, mr, de, ru, es, zh-Hans, ja)
- 💬 Ask multiple questions without reloading the video
- 🧠 Uses RAG with Groq LLM + HuggingFace embeddings
- 🎛️ Interactive Streamlit interface with video preview

---

## 🧠 How It Works

1. Enter a YouTube video ID
2. The transcript is fetched in preferred languages
3. Text is split, embedded using `intfloat/e5-small` (HuggingFace)
4. FAISS indexes the transcript
5. LangChain retrieves context and Groq's LLM answers the question

--
## 🌍 Supported Languages

| Language             | Code    |
| -------------------- | ------- |
| English              | en      |
| Hindi                | hi      |
| Marathi              | mr      |
| German               | de      |
| Russian              | ru      |
| Spanish              | es      |
| Chinese (Simplified) | zh-Hans |
| Japanese             | ja      |

---

## 👨‍💻 Author

### Aditya Aher
### Aspiring AI Engineer-

---
## 🚀 Run the App

```bash
streamlit run youtube_chatbot.py



