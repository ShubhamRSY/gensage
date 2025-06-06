from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load the dataset (LEDGAR legal dataset)
dataset = load_dataset("lex_glue", "ledgar", split="train[:500]")  # using 500 samples for faster test

# Step 2: Preprocess the documents
documents = [entry['text'] for entry in dataset]

# Step 3: Split documents into smaller overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.create_documents(documents)

# Step 4: Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 6: Save the vector store
vectorstore.save_local("faiss_index")

print("✅ Vector store created and saved successfully!")
