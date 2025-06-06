# ⚖️ Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) application that accurately answers legal questions using document-based context. Built with LangChain, Hugging Face Transformers, FAISS, and Streamlit, this tool is ideal for navigating the intricacies of legal agreements with grounded AI responses.

---

## 📌 Problem Statement

**"Can AI accurately and concisely answer legal questions based only on the contents of legal agreements?"**

Most AI models hallucinate answers or rely on general knowledge. This project solves that by grounding the responses **strictly in legal documents** via retrieval-augmented generation (RAG).

---

## 📄 Dataset Description

- **Source**: [LexGLUE - LEDGAR dataset](https://huggingface.co/datasets/lex_glue), a public dataset of legal clauses extracted from contracts.
- **Preprocessing**: Text is chunked and embedded using the `sentence-transformers/all-MiniLM-L6-v2` model, stored in a FAISS vector index.
- **Topics Covered**:
  - Rights of debenture holders
  - Liquidation and insolvency
  - Secured vs. unsecured creditors
  - Remedies upon company default

---

## 🧠 Model Architecture

| Component        | Technology Used                        |
|------------------|----------------------------------------|
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store     | FAISS                                  |
| LLM              | `google/flan-t5-base`                  |
| Framework        | LangChain                              |
| UI               | Streamlit                              |

### 🔄 RAG Workflow

1. **User asks a legal question**
2. **FAISS retrieves** relevant document chunks
3. **FLAN-T5** uses the retrieved context to generate an answer
4. **Output shown in Streamlit** with feedback options

---

## 🚀 Features

- 🔍 Ask legal questions grounded in contract data
- 📚 Context retrieved via semantic search
- ✅ Clear, formal, and document-based answers
- 📩 Feedback collection with CSV logging

---

## 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/ShubhamRSY/gensage.git
cd gensage
```
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run rag_qa_ui.py


📁 Project Structure

gensage/
├── chunk_embed.py           # Converts PDF/docs into vector chunks
├── rag_qa_ui.py             # Streamlit app interface
├── faiss_index/             # Vectorstore with embedded chunks
├── feedback_log.csv         # User feedback records
├── requirements.txt         # Dependencies
└── README.md                # This file

🧪 Sample Questions to Try
Is a debenture holder considered a secured creditor during liquidation?

What legal remedies are available to a secured creditor if the company defaults?

Can a company issue both secured and unsecured debentures?


✅ Current Status
✅ Vector search and retrieval working

✅ Streamlit UI with feedback logging

✅ GitHub repo live at: ShubhamRSY/gensage

🌐 Next Steps
🚀 Deploy to Streamlit Cloud or Hugging Face Spaces

🎯 Prompt tuning for better LLM outputs

🔄 Batch question support

📎 File/document upload in UI

📘 Acknowledgments
Hugging Face Datasets

LangChain

Streamlit

FAISS


