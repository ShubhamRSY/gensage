from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import torch

# Setup device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load vector store and embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Ask a question
query = "Can a company issue both secured and unsecured debentures?"
docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([doc.page_content for doc in docs])

# Load the FLAN-T5 model
text_gen = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if device == "cuda" else -1)
llm = HuggingFacePipeline(pipeline=text_gen)

# Define prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a legal assistant AI. Based on the legal context provided below, answer the question clearly and concisely.
    
    Legal Context:
    {context}

    Question: {question}
    Answer:"""
)
print("\n🔎 Retrieved Context:\n", context)


# Format and invoke
formatted_prompt = prompt.format(context=context, question=query)
response = llm.invoke(formatted_prompt)
print("\n📘 Answer:", response)
