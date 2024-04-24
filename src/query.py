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
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from typing import List, Callable, Dict
import os
import logging
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up environment for Azure Cognitive Services
load_dotenv()
credential = AzureCliCredential()
os.environ["OPENAI_API_KEY"] = credential.get_token(
    "https://cognitiveservices.azure.com/.default").token


class MyVectorStoreRetriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs


class QueryHandler:
    """Handles creating and managing a conversational query chain for retrieving answers."""

    def __init__(self, embedding_function: Callable, llm_name: str, collection_name: str):
        """
        Initializes the query handler with specified embeddings, language model, and collection name.

        Args:
            embedding_function (Callable): Function used to generate embeddings.
            llm_name (str): The name of the language model for the conversational chain.
            collection_name (str): The name of the collection for storing embeddings.
        """
        try:
            self.embedding_function = embedding_function
            self.llm = AzureChatOpenAI(deployment_name=llm_name)
            self.vectorstore = Chroma(
                persist_directory=collection_name, embedding_function=self.embedding_function)
            self.retriever = MyVectorStoreRetriever(
                vectorstore=self.vectorstore, search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.6})
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)
            logging.info("QueryHandler initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize QueryHandler: {e}")
            raise

    def handle_question(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Handle a question and returns an answer using the conversational retrieval chain.

        Args:
            query (str): The question to be asked.
            chat_history (List[Dict[str, str]]): A list of previous chat messages represented as dictionaries.

        Returns:
            Dict[str, any]: The result containing the answer and other relevant information.
        """
        try:
            result = self.qa_chain.invoke(
                {"question": query, "chat_history": chat_history})
            logging.info(
                f"Question asked: {query}, Answer received: {result['answer']}")

            return result
        except Exception as e:
            logging.error(f"Error asking question: {query}, Error: {e}")
            return {}