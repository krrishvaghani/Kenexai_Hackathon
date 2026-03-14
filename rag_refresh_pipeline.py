import pandas as pd
from sqlalchemy import create_engine
import os
import urllib.parse
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_db_engine():
    load_dotenv()
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")
    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_uri)

def refresh_rag_vector_db():
    print("Extracting full updated dataset for RAG refresh...")
    engine = get_db_engine()
    
    # In a real environment, you'd only query the new rows representing diff/upsert.
    # To ensure completeness after synthetic generation, we pull the freshest dataset.
    df = pd.read_sql("SELECT * FROM processed_feature_data ORDER BY credit_score DESC LIMIT 2000", engine)
    
    docs = []
    print("Converting tabular records into Vector Documents...")
    for idx, row in df.iterrows():
        # Minimal safely mapped string focusing on core fields guaranteed to exist.
        text_elements = []
        for col in df.columns:
            text_elements.append(f"{col}: {row[col]}")
            
        doc_content = " ".join(text_elements)
        docs.append(Document(page_content=doc_content, metadata={"row_id": str(idx), "source": "synthetic_expansion"}))
        
    persist_dir = os.path.join(os.path.dirname(__file__), "genai", "vector_store")
    
    print("Loading HuggingFace embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"Refreshing Vector DB persisting at {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # Make sure to persist
    vectorstore.persist()
    print("RAG Knowledge Base updated successfully!")
    return True

if __name__ == "__main__":
    refresh_rag_vector_db()