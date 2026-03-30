# ⚖️ AI Legal Document Analyzer & Translator

> 🚀 Production-grade AI system for contract risk analysis, amendment generation, and grounded legal QA — optimized for CPU-only deployment

---

## 📌 Overview

This project is an end-to-end **AI-powered legal intelligence system** designed to make legal documents understandable, fair, and accessible.

It enables users to:

- 🔍 Detect risky or biased clauses in contracts  
- ⚖️ Classify clauses into **Pro / Neutral / Con**  
- ✍️ Generate safer legal amendments  
- ❓ Ask questions over documents (RAG-based QA)  
- 🌐 Translate contracts into multiple languages  
- 🔊 Convert legal text into speech (TTS)  

> ❗ Built with a strong focus on **zero hallucination**, interpretability, and accessibility.

---

## 🧠 Key Features

- 📊 **Clause Risk Classification (3-class framework)**
- 🧩 **Graph-based contextual understanding (GAT)**
- ✍️ **Controlled Amendment Generation**
- 🔎 **RAG-based Legal Q&A (extractive, grounded)**
- 🌍 **Multilingual Translation Support**
- 🔊 **Text-to-Speech (TTS)**
- ⚡ **CPU-first deployment (no GPU required)**

---

## 🏗️ System Architecture
    User Upload
    ↓
    Clause Parser (PyMuPDF + Regex)
    ↓
    Classification Pipeline (4 Layers)
    ↓
    Amendment Generator (Template + RAG)
    ↓
    RAG QA System (FAISS + RoBERTa)
    ↓
    Translation + TTS
    ↓
    Frontend Output


---

## ⚙️ Core Pipeline

### 🔹 1. Clause Segmentation
- Extracts text from PDF/TXT using PyMuPDF  
- Rule-based clause splitting  
- Handles unstructured and noisy documents  

---

### 🔹 2. Classification Pipeline (4-Layer Cascade)

1. **LLM Semantic Biasing (Qwen via Ollama)**  
2. **Rule-based Weak Supervision**  
3. **MLP Classifier (DeBERTa embeddings)**  
4. **LLM Fallback (Mistral)**  

👉 Designed to **minimize LLM usage → faster, cost-efficient inference**

---

### 🔹 3. Amendment Generation

- Identifies **“Con” (risky) clauses**  
- Retrieves similar templates using **InLegalBERT + FAISS**  
- Generates **legally safer rewritten clauses**  

---

### 🔹 4. RAG-Based QA System

- Retrieves:
  - Top 3 relevant contract clauses  
  - Top 2 constitution/legal references  
- Uses **RoBERTa-SQuAD2** for extractive QA  
- ❗ Ensures **no hallucination (answers are extracted, not generated)**  

---

### 🔹 5. Accessibility Layer

- 🌍 Translation via Google Translate API  
- 🔊 Text-to-Speech using gTTS  
- Supports multiple Indian languages  

---

## 📊 Evaluation Metrics

### 🔍 Clause Classification Accuracy

| Category        | Accuracy |
|----------------|---------|
| Clear Bias     | 93.33%  |
| Ambiguous Bias | 73.33%  |
| **Overall**    | **83.33%** |

---

### ✍️ Amendment Generation Accuracy

| Category        | Accuracy |
|----------------|---------|
| Clear Clauses  | 80%     |
| Ambiguous      | 66.67%  |
| **Overall**    | **75%** |

---

### 🧠 Key Observations

- Strong performance on **explicitly biased clauses**  
- Reduced accuracy on **ambiguous legal language**  
- Effective amendment generation for high-risk clauses  

---

## ⚙️ Tech Stack

| Layer            | Technology |
|------------------|-----------|
| **Backend**      | FastAPI, Uvicorn |
| **ML Models**    | PyTorch, GAT, MLP |
| **Embeddings**   | SentenceTransformers, InLegalBERT |
| **Vector DB**    | FAISS |
| **LLM**          | Ollama (Qwen, Mistral) |
| **QA Model**     | RoBERTa-SQuAD2 |
| **Parsing**      | PyMuPDF |
| **Translation**  | Google Translate API |
| **TTS**          | gTTS |
| **Frontend**     | HTML, CSS, JavaScript |

---

## 💻 Deployment

- ✅ Runs entirely on **CPU-only hardware**  
- ✅ Lightweight and locally deployable  
- ✅ No dependency on paid APIs  

---

## 🚀 Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Tinaprabhat/CoursePlanner.git
cd CoursePlanner
```
###2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
###3️⃣ Start Ollama
```bash
ollama serve
ollama pull qwen
ollama pull mistral
```
###4️⃣ Run Backend
```bash
uvicorn backend.main:app --reload
```
## 🧪 Testing

Run all tests:

```bash
python -m pytest -s
```
Coverage Includes:
✅ LLM interaction
✅ Embedding generation
✅ Clause chunking
✅ Ingestion pipeline
✅ RAG-based QA

## 🛡️ Design Principles
❌ Avoid hallucination → extractive QA only
⚖️ Fairness-first clause rewriting
⚡ Minimize LLM dependency (hybrid pipeline)
🧩 Modular and scalable design

## 🚧 Limitations
Lower performance on ambiguous legal clauses
Template retrieval may not always match context
Small LLM limits deep legal reasoning

## 🔮 Future Improvements
🔼 Upgrade to larger LLMs (Qwen 7B / API-based)
🧠 Improved semantic understanding
🔁 Cross-encoder reranking
⚡ GPU / distributed inference
🌍 Multi-domain legal expansion

## 🏆 Highlights
✅ Hybrid AI system (NLP + Graph ML + RAG)
✅ Real-world evaluation metrics (not toy demo)
✅ CPU-first deployment (high practicality)
✅ Handles real legal documents

## 🧩 Architecture Diagram (Conceptual)
                +----------------------+
                |   User Interface     |
                +----------+-----------+
                           |
                           ↓
                +----------------------+
                |   FastAPI Backend    |
                +----------+-----------+
                           |
        -----------------------------------------
        |                |                      |
        ↓                ↓                      ↓
    
    +---------------+  +--------------+   +------------------+
    | Classification|  | Amendment    |   |   RAG QA System  |
    | Pipeline      |  | Generator    |   | (FAISS + QA)     |
    +---------------+  +--------------+   +------------------+
            |                |                      |
            -----------------------------------------
                               ↓
                    +----------------------+
                    |  Response Formatter  |
                    +----------------------+

## 🔬 Design Decisions
Why Hybrid Pipeline?

Pure LLM → expensive + hallucination risk
Pure ML → lacks contextual reasoning

👉 Hybrid approach provides:

⚡ Efficiency
🎯 Accuracy
🛡️ Reliability

### Why Extractive QA (RAG)?
Guarantees no hallucination
Ensures legal correctness
Improves trustworthiness
Why CPU-first Design?
Increases accessibility
No dependency on GPUs or paid APIs
Enables local deployment

## 📈 Performance Considerations
Component	Optimization
Retrieval	FAISS indexing
LLM Calls	Minimized via cascade
Inference	CPU-efficient models
Chunking	Balanced size (context vs speed)

## 🔐 Security & Reliability
No external API dependency for core logic
Local inference ensures privacy
Deterministic outputs (low temperature)
Structured responses for consistency

## 🤝 Contribution Guidelines
Fork the repository
Create a feature branch
Commit changes with clear messages
Submit a pull request

## 🐛 Known Issues
Complex legal ambiguity reduces classification accuracy
Amendment templates may require manual validation
Long documents may increase latency

## 📚 References
SentenceTransformers
FAISS (Facebook AI Similarity Search)
HuggingFace Transformers
Ollama LLM Runtime
RoBERTa-SQuAD2
## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author
Tina Prabhat, Reyan Das, Sreshtho Sen, Anubhav Shaha, Sourish Das
B.Tech CSE — KIIT University
