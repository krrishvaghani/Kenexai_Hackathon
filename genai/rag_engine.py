import os
import json
import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Suppress chromadb and HuggingFace warnings
import warnings
warnings.filterwarnings('ignore')

persist_dir = os.path.join(os.path.dirname(__file__), "vector_store")

def get_rag_chain():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load local vector store
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Create retriever returning top 5 matches
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Initialize Local Ollama (Make sure ollama serve is running with llama3-chatqa:8b)
    llm = Ollama(model="llama3-chatqa:8b")
    
    # Memory for follow up queries
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    # Custom System Prompt for Insurance Context
    template = '''You are a highly capable AI Insurance Analytics Copilot. Your job is to analyze the provided company data and write a professional, natural-language response for an insurance manager.

CRITICAL INSTRUCTIONS:
1. You MUST answer in full, complete sentences.
2. Formulate your response as a professional business summary paragraph.
3. UNDER NO CIRCUMSTANCES should you output a bare list of metrics or mimic the raw dataset text.
4. You act as an analytical reasoning engine. Even if the text does not explicitly state a correlation, you must synthesize the data points provided in the context (like Risk Score, Credit Score, and Claim Probability) to draw a logical conclusion and answer the user's question directly.

Context from Company Data:
{context}

User Question: {question}

Professional Business Summary:'''

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return chain

def log_interaction(question, context_docs, answer):
    log_file = os.path.join(os.path.dirname(__file__), "..", "copilot_logs.json")
    
    context_texts = [doc.page_content for doc in context_docs]
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_question": question,
        "retrieved_context": context_texts,
        "llm_response": answer
    }
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                pass
                
    logs.append(log_entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)
