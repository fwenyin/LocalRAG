"""
Query Handling and Conversational Retrieval Module.

This module is designed for managing queries using Azure Cognitive Services, particularly through Azure OpenAI services. It facilitates creating and managing conversational retrieval chains to fetch relevant information based on queries.

Requirements:
- Azure authentication is needed to utilize Azure Cognitive Services. Ensure 'az login' is used for Azure CLI authentication prior to use.
- In case of network issues leading to HTTPS connection retries exceeding the limit, use the command:
    az account get-access-token --resource https://cognitiveservices.azure.com --query "accessToken" -o tsv

Classes:
- QueryHandler: Initializes the conversational retrieval chains with Azure OpenAI and Chroma vector stores and manages query processing.

Usage:
This module can be imported where there is a need to incorporate question answering functionalities into applications, leveraging deep learning models for natural language understanding and retrieval.
"""

from azure.identity import AzureCliCredential
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import os
import logging
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up environment for Azure Cognitive Services
load_dotenv()
credential = AzureCliCredential()
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token


class QueryHandler:
    """Handles creating and managing a conversational query chain for retrieving answers."""

    def __init__(self, embeddings_name, llm_name, collection_name):
        """
        Initializes the query handler with specified embeddings, language model, and collection name.

        Args:
            embeddings_name (str): The name of the Azure deployment for generating embeddings.
            llm_name (str): The name of the language model for the conversational chain.
            collection_name (str): The name of the collection for storing embeddings.
        """
        try:
            self.embeddings = AzureOpenAIEmbeddings(azure_deployment=embeddings_name)
            self.llm = AzureChatOpenAI(deployment_name=llm_name)
            self.vectorstore = Chroma(persist_directory=collection_name, embedding_function=self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)
            self.chat_history = []
            logging.info("QueryHandler initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize QueryHandler: {e}")
            raise

    def ask_question(self, query):
        """
        Asks a question and retrieves an answer using the conversational retrieval chain.

        Args:
            query (str): The question to be asked.

        Returns:
            dict: The result containing the answer and other relevant information.
        """
        try:
            result = self.qa_chain.invoke({"question": query, "chat_history": self.chat_history})
            
            # Update chat history with the current query and its answer
            self.chat_history.append((query, result["answer"]))
            logging.info(f"Question asked: {query}, Answer received: {result['answer']}")
            
            return result
        except Exception as e:
            logging.error(f"Error asking question: {query}, Error: {e}")
            return {}