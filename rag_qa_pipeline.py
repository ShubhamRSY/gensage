import os
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Set your Hugging Face API key if needed (not required for local models)
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice set to use {device}")

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Input query
query = "What is a debenture?"

# Step 1: Retrieve relevant documents
print("\n⏳ Querying vectorstore...")
start_time = time.time()
docs = vectorstore.similarity_search(query, k=3)
print(f"🔍 Retrieved {len(docs)} documents in {round(time.time() - start_time, 2)}s")

# Step 2: Initialize FLAN-T5 model
print(f"Device set to use {device}")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

text_gen_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_new_tokens=256,
    temperature=0.3,
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Step 3: Define Prompt Template
prompt_template = """
You are a legal assistant. Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Step 4: Run QA chain
print("\n🤖 Generating answer using FLAN-T5...")
qa_chain = LLMChain(llm=llm, prompt=prompt)

start_gen = time.time()
response = qa_chain.invoke({
    "context": "\n\n".join(doc.page_content for doc in docs),
    "question": query
})
print(f"✅ Answer generated in {round(time.time() - start_gen, 2)}s")

# Step 5: Display final answer
print("\n📘 Final Answer:", response['text'])

print(f"\n⏱️ Total pipeline time: {round(time.time() - start_time, 2)}s")
