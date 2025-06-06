from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the saved FAISS index
db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# 3. Ask a question
query = "What is a debenture?"
results = db.similarity_search(query, k=3)

# 4. Print the top 3 matched chunks
print(f"\n🔍 Top matches for: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"Match {i}:\n{doc.page_content}\n{'-'*60}")
