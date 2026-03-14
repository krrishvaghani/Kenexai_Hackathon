import os
import sys

# Ensure parent directory is in path to allow module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from genai.data_to_documents import extract_and_convert_data

def build_vector_store():
    persist_dir = os.path.join(os.path.dirname(__file__), "vector_store")
    
    # 1. Get Documents
    docs = extract_and_convert_data()
    
    # 2. Init Embeddings (local sentence transformers)
    print("Loading sentence-transformers/all-MiniLM-L6-v2 embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Build Vector Store
    print("Building ChromaDB Vector Store...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"? Successfully persisted Chroma vector DB to {persist_dir}")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
