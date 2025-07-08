import streamlit as st
import sys
import os
from pathlib import Path
from rag_pipeline import (
    load_vector_store,
    load_llm,
    retrieve_context,
    generate_answer
)

# Get absolute path to model directory
#MODEL_DIR = str(Path("../models/flan-t5-base").resolve())
MODEL_DIR = r'C:\Users\filimon.hailemariam\Documents\Week_6\Intelligent-Complaint-Analysis\models\flan-t5-base'

# Improved initialization with better resource management
@st.cache_resource
def initialize():
    try:
        # Load vector store first
        vector_store = load_vector_store("vector_store")
        
        # Initialize LLM as None for lazy loading
        llm = None
        
        return vector_store, llm
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()  # Stop the app if initialization fails
        return None, None

vector_store, llm = initialize()

# UI Layout
st.set_page_config(
    page_title="CrediTrust Complaint Assistant", 
    layout="wide",
    page_icon="📊"
)
st.title("📊 CrediTrust Complaint Assistant")
st.markdown("Ask a question based on customer complaint data.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "How can I help you with customer complaints today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Type your question about customer complaints...")
if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            try:
                # Lazy load LLM only when first needed
                if llm is None:
                    with st.spinner("Loading AI model..."):
                        llm = load_llm(MODEL_DIR)  # Use the resolved path
                
                # Retrieve context
                context_chunks = retrieve_context(prompt, vector_store)
                
                # Generate answer
                answer = generate_answer(prompt, context_chunks, llm)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                if context_chunks:
                    with st.expander("📚 Relevant Sources"):
                        for i, chunk in enumerate(context_chunks[:3]):
                            st.markdown(f"**Source {i+1}:**")
                            st.info(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your request."
                })

# Clear button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "How can I help you with customer complaints today?"
    }]
    st.rerun()