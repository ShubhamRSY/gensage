# ⚖️ Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) application designed to answer legal questions using document-based context. Built using LangChain, Hugging Face Transformers, FAISS, and Streamlit.

## 📌 Problem Statement

**"Can AI accurately and concisely answer legal questions based only on the contents of legal agreements?"**

Traditional language models may hallucinate answers or rely on general knowledge. This project solves that by grounding the answers strictly in the context of uploaded legal documents using RAG.

---

## 📄 Data Description

- **Source**: Internal legal documents (e.g., PDF agreements).
- **Preprocessing**: PDFs are split into text chunks, embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored in a FAISS vectorstore.
- **Example Topics**:
  - Debenture holders and their rights
  - Liquidation process
  - Secured vs. unsecured creditors

---

## 🧠 Model & Architecture

- **Vector Store**: FAISS
- **Embeddings**: Hugging Face Sentence Transformers
- **LLM**: `google/flan-t5-base` used via Transformers pipeline
- **Framework**: LangChain
- **Frontend**: Streamlit app with user question input, generated answers, and feedback mechanism

### Flow:
1. User submits a legal question.
2. FAISS retrieves top-k similar document chunks.
3. LLM uses those chunks to generate a response.
4. Answer is displayed with an option for user feedback.

---

## 🚀 Features

- Ask any legal question based on your uploaded contract PDFs.
- Retrieves only the most relevant context.
- Ensures output is grounded, not hallucinated.
- Accepts user feedback (positive/negative) for evaluation.

---

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/ShubhamRSY/gensage.git
cd gensage
```
# Set up virtual environment and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit app
streamlit run rag_qa_ui.py


📁 Folder Structure
bash
Copy
Edit
gensage/
├── faiss_index/                # FAISS index built from PDF chunks
├── feedback_log.csv            # Logged user feedback
├── rag_qa_ui.py                # Streamlit-based RAG interface
├── chunk_embed.py              # Script to chunk PDFs and embed them
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
🧪 Sample Questions
"Is a debenture holder considered a secured creditor during liquidation?"

"What remedies does a secured creditor have if a company defaults?"

"Can a company issue both secured and unsecured debentures?"

✅ Status
 Vector search and RAG working

 Streamlit frontend operational

 Feedback system logging to CSV

 Uploaded to GitHub

📌 Next Steps
 Improve prompt tuning for more robust answers

 Deploy on Streamlit Cloud or Hugging Face Spaces

 Add multi-query and batch support

 Include document upload support in UI


