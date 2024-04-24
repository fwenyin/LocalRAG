"""
Streamlit Chatbot RAG Integration Module. 

This module is designed for integrating the RAG pipeline with a Streamlit application to create a chatbot interface. It leverages the QueryHandler class from the query.py module to handle user queries and generate responses using the RAG model.
This example makes use of HuggingFace Embeddings.

Requirements:
- Azure authentication is required to access Azure Cognitive Services. Use 'az login' to authenticate the Azure CLI before running any operations.
- In cases where the HTTPS connection exceeds maximum retries due to network issues, execute the following command:
    az account get-access-token --resource https://cognitiveservices.azure.com --query "accessToken" -o tsv

Usage:
This module can be used as a POC of the RAG pipeline. It can be run directly to start a Streamlit application with chatbot functionality, allowing users to interact with the RAG model through a simple text interface.
"""

import re
import streamlit as st
import streamlit_chat as stc
import bleach
from langchain_community.embeddings import HuggingFaceEmbeddings
from query import QueryHandler


st.set_page_config(layout="wide")

def clean_text(text: str) -> str:
    """
    Cleans the input text by replacing sequences of whitespace characters with a single space 
    and stripping leading and trailing whitespace.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = bleach.clean(text, strip=True)
    cleaned_text = re.sub(r'(\n\s*)+\n', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def main(query_handler):
    """
    Main function to run the Streamlit application with chatbot functionality.
    
    Args:
        query_handler (QueryHandler): The query handler object to handle user queries and generate responses.
    """
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'source_docs' not in st.session_state:
        st.session_state['source_docs'] = []
    if 'pending_response' not in st.session_state:
        st.session_state['pending_response'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    st.title('FC Chatbot')

    for i, message in enumerate(st.session_state.messages):
        stc.message(message["content"], is_user=(message["role"] == "user"), key=str(i), avatar_style="avataaars" if message["role"] == "user" else "bottts")
        # with st.chat_message(message["role"]):
        #    st.markdown(message["content"])

    with st.sidebar:
        st.subheader("Source Documents")
        for index, doc in enumerate(st.session_state.source_docs):
            with st.container():
                st.markdown("#### Document Content:")
                st.text_area("", value=doc['content'], height=150, key=f"doc_content_{index}", disabled=True)
                #st.markdown(f"\"{doc['content']}\"")
                st.markdown(f"**Source:** {doc['source']}")
                st.markdown(f"**Content Match Score:** {doc['score']:.2f}")
                st.markdown("---")

    with st.form("Question", clear_on_submit=True):
        prompt = st.text_input("Enter a message...")
        submitted = st.form_submit_button("Send")

    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state['pending_response'] = True
        st.rerun()

    if st.session_state['pending_response']:
        response = query_handler.handle_question(
            prompt, st.session_state['chat_history'])
        answer = response["answer"]
        st.session_state['chat_history'].append((prompt, answer))
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
        st.session_state.source_docs = [{'content': clean_text(doc.page_content),
                                         'source': doc.metadata['source'],
                                         'score': doc.metadata['score']} for doc in response['source_documents']]
        st.session_state['pending_response'] = False
        st.rerun()


if __name__ == "__main__":
    query_handler = QueryHandler(embedding_function=HuggingFaceEmbeddings(model_name="embedding_models/baai/bge-small-en-v1.5"),
                             llm_name="ChatGpt",
                             collection_name="chroma_index",
                             )
    main(query_handler)
