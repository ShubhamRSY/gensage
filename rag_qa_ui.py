import streamlit as st
import torch
import os
import pandas as pd
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

# Set page configuration
st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️")

# Device info
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
st.sidebar.markdown(f"**Using device:** `{device}`")

# Load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load text generation model
text_gen_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # Upgraded model for better responses
    device=0 if device == "cuda" else -1
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a legal assistant AI. Based only on the legal text below, answer clearly and accurately.\n\n"
        "Legal Context:\n{context}\n\n"
        "Legal Question: {question}\n"
        "Answer:"
    )
)

# Chain with updated Runnable syntax
qa_chain = (
    RunnableMap({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
    | prompt
    | llm
)

# Streamlit UI
st.title("⚖️ Legal RAG Assistant")
query = st.text_input("Ask your legal question:")

if query:
    with st.spinner("🔍 Retrieving legal context..."):
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        st.write("### Retrieved Context:")
        st.write(context)  # Displaying context for debugging

    with st.spinner("🤖 Generating answer..."):
        response = qa_chain.invoke({"context": context, "question": query})
        answer = response.strip()

    st.success("✅ Final Answer:")
    st.markdown(f"**{answer}**")

    # Feedback section
    st.markdown("---")
    st.markdown("### Was this answer helpful?")
    col1, col2 = st.columns(2)

    feedback = None
    with col1:
        if st.button("👍 Yes"):
            feedback = {"question": query, "answer": answer, "feedback": "positive"}
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No"):
            feedback = {"question": query, "answer": answer, "feedback": "negative"}
            st.warning("Thanks! We'll work to improve.")

    if feedback:
        feedback_df = pd.DataFrame([feedback])
        feedback_file = "feedback_log.csv"
        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)
