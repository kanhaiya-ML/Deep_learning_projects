# 🤖 Conversation Chatbot

A conversational AI chatbot powered by Microsoft's DialoGPT model, built with PyTorch and HuggingFace Transformers.

---

## 📁 Project Structure
chatBot/
├── conversation_Bot.py   ← terminal version (core logic)
├── app.py                ← streamlit web interface
└── requirements.txt

---

## 🚀 What It Does

- Understands and responds to natural conversation
- Remembers entire conversation history (There is a bug in History chat if anyone helps me to understand i am very greatful)
- Gives contextual responses based on previous messages
- Built on Microsoft DialoGPT-medium model

---

## 💻 How To Run

### Terminal Version
```bash
python conversation_Bot.py
```

### Web Version
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| DialoGPT-medium | Conversational AI model |
| HuggingFace Transformers | Model loading and inference |
| PyTorch | Deep learning framework |
| Streamlit | Web interface |

---

## 📸 Features

- 💬 Natural conversation interface
- 🧠 Conversation memory across turns
- 🔄 Clear chat option
- ⚡ Powered by DialoGPT

---

## 📦 Installation

```bash
pip install transformers torch streamlit
```

---

## 🧠 How It Works
User types message
↓
Message encoded with tokenizer
↓
Combined with conversation history
↓
DialoGPT generates response
↓
Response decoded and displayed
↓
History updated for next turn

---

## ⚠️ Limitations

- DialoGPT trained on Reddit conversations
- May give inconsistent responses on long conversations
- No internet access or real-time knowledge
- Runs on CPU — responses may be slow

---

## 👨‍💻 Author

Kanhaiya — Self-taught Deep Learning Engineer
GitHub: github.com/kanhaiya-ml
