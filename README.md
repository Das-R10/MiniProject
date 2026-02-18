

# 📄 AI Contract Reader & Amendment System

A full-stack **AI-powered contract analysis platform** that automatically:

✔ Parses and understands contract clauses
✔ Detects potentially unfair or risky (“Con”) clauses
✔ Classifies clauses as *Pro / Con / Neutral*
✔ Suggests fair, role-based amendments for risky clauses
✔ Supports multilingual translation of contracts

The system combines:

* Transformer embeddings
* Weak labeling
* Graph-based learning
* Controlled GenAI amendment generation
* Translation utilities

---

## 🚀 Overview

Contracts — especially employment contracts — often contain complex and ambiguous language that may disadvantage one party. This AI tool helps users by:

🎯 **Analyzing and classifying clauses**
🤖 **Identifying risk**
✍️ **Suggesting fair amendments**
🌐 **Translating text to other languages**

---

## 🧠 Technical Architecture

### 1. **Clause Embeddings & Semantic Features**

Uses Sentence Transformers (`all-mpnet-base-v2`) to create semantic vector representations of each clause.

---

### 2. **Graph Construction**

Builds a clause graph using:

* Sequential connections
* Same section links
* Semantic similarity edges

This helps capture context beyond isolated sentences.

---

### 3. **Weak Labeling**

Rule-based heuristic weak labeler identifies Pro/Con/Neutral signals using patterns like:

* Risky: `"sole discretion"`, `"without notice"`, `"terminate"`
* Fair: `"prior written notice"`, `"severance"`

Confidence scores guide later modeling.

---

### 4. **Graph Model / MLP Classifier**

The system:

* uses a Graph Attention Network (GAT) if `torch_geometric` is available
* otherwise falls back to MLP

This produces final clause classifications from learned context.

---

### 5. **Amendment Generation (Controlled GenAI / RAG)**

For risky clauses (label = Con):

* Builds role-based prompts for desired fairness perspective
* Generates controlled amendment suggestions
* Validates and merges into original text
* Prevents unsafe wandering language or hallucination

This is **not free LLM rewriting** — it uses constrained prompting to maintain legal meaning.

---

### 6. **Translation Support**

`translate_api.py` provides utilities to translate clauses or full contracts between languages via external APIs.

*(Frontend can integrate translation for multilingual users.)*

---

## 📁 Repository Structure

```
MiniProject/
│
├── backend/
│   ├── clause_parser.py       # Breaks full contract into clauses
│   ├── pipeline.py            # Core AI pipeline
│   ├── translate_api.py       # Translation utilities
│   ├── rag.py                 # Amendment generation logic
│   ├── test.py                # Test harness
│
├── frontend/                  # UI integration 
│   └── index.html
    └── ext.html
│
├── sample_contract.txt        # Example input contract
├── sdtest.txt                 # Additional sample
├── requirements.txt           # Python dependencies
└── README.md                 # Project overview
```

---

## 🧪 How to Run (Local)

1️⃣ Clone the repository:

```bash
git clone https://github.com/Das-R10/MiniProject.git
cd MiniProject/backend
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run tests:

```bash
python test.py
```

This runs the pipeline on sample clauses and outputs:

```
Clause ID | Section | Label | Confidence | Original | Amended
```

---

## 📌 Classification Output Example

Example output from `test.py`:

```
Clause: The employer may modify terms at its sole discretion.
Label : Con
Amendment: Employer may modify terms only after prior written notice and employee consultation.
```

---

## 🧠 Design Rationale

### ✅ Weak supervision + graph model

This combines pattern-based labeling with learned contextual signals.

### 🔒 Controlled amendment generation

Avoids freeform AI hallucinations by designing *role-based fair amendment prompts*, producing safe outputs.

### 🌍 Translation

Supports multi-language contract analysis.

---

## ✨ Features

### 🔍 Clause Parsing

Breaks long contracts into manageable clause units.

### 📘 Graph Learning

Contextualizes clauses across sections & semantic similarity.

### 📊 Classification

Pro / Con / Neutral with confidence scores.

### 📝 Fair Amendment Recommendations

Powered by safe and targeted LLM prompts.

### 🌐 Translation API

Clause or contract translation support.

---

## 💡 Project Significance

This project demonstrates an end-to-end pipeline for contract automation, blending classical NLP, graph modeling, and generative AI under safety constraints — ideal for real-world legal tech applications.

---

## ⚡ Author

Developed as an advanced academic and applied AI project combining ML, NLP, and practical contract analysis system design.

---

## 📝 License

MIT License

---




