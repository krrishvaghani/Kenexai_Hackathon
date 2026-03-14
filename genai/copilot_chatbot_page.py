import streamlit as st
import os
import sys

# Optional: Add parent dir to path if run standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genai.rag_engine import get_rag_chain, log_interaction

def render_copilot_page():
    st.title("Insurance AI Copilot")
    st.markdown("Ask natural language questions about our insurance parameters, drivers, claim risks, and internal data. *Runs 100% locally via Ollama.*")
    
    # Initialize the RAG chain in session state
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing Vector Database and Local LLM via Ollama..."):
            try:
                st.session_state.rag_chain = get_rag_chain()
                st.session_state.messages = []
                st.toast("Copilot Initialized and Ready!")
            except Exception as e:
                st.error(f"Failed to initialize Copilot: {str(e)}")
                st.stop()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display retrieved docs if present
            if msg["role"] == "assistant" and "sources" in msg and len(msg["sources"]) > 0:
                with st.expander("Show Retrieved Context"):
                    for i, doc in enumerate(msg["sources"]):
                        st.text(f"Document {i+1}: {doc}")

    # User Input
    user_query = st.chat_input("Ask about risk clusters, specific driver features, etc...")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"question": user_query})
                    
                    answer = response["answer"]
                    sources_docs = response.get("source_documents", [])
                    source_texts = [doc.page_content for doc in sources_docs]
                    
                    st.markdown(answer)
                    
                    if source_texts:
                        with st.expander("Show Retrieved Context"):
                            for i, text in enumerate(source_texts):
                                st.text(f"Document {i+1}: {text}")
                                
                    # Log to JSON
                    log_interaction(user_query, sources_docs, answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_texts
                    })
                except Exception as e:
                    st.error(f"Error querying model: {str(e)}\n\nMake sure Ollama is running and 'llama3-chatqa:8b' is pulled.")
