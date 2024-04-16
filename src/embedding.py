"""
Document Processing and Embedding Management Module.

This module is designed for document processing and embedding management, utilizing Azure Cognitive Services for its operations. It specifically leverages the Azure OpenAI service to generate embeddings for documents.

Requirements:
- Azure authentication is required to access Azure Cognitive Services. Use 'az login' to authenticate the Azure CLI before running any operations.
- In cases where the HTTPS connection exceeds maximum retries due to network issues, execute the following command:
    az account get-access-token --resource https://cognitiveservices.azure.com --query "accessToken" -o tsv

Classes:
- DocumentHandler: Manages the loading, preprocessing, and splitting of documents into manageable chunks.
- EmbeddingManager: Handles the generation of embeddings and manages vector storage operations, ensuring efficient data retrieval and storage.

Usage:
This module can be imported to integrate document processing and embedding functionalities into applications requiring advanced text analysis and processing capabilities.
"""

from unstructured.cleaners.core import clean
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import AzureCliCredential
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


class DocumentHandler:
    """
    Handles operations related to document processing such as loading, preprocessing and splitting into manageable chunks.
    """
    
    def __init__(self, file_path):
        """
        Initializes the DocumentHandler with a file path for documents.
        
        Args:
            file_path (str): The path to the directory of raw files to process.
        """
        self.file_path = file_path
        self.loader = self.create_loader(file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '. ', '? ', '! '])

    def create_loader(self, path):
        """
        Creates a loader for unstructured files with specific text cleaning post-processors.

        Args:
            path (str): The path to the directory of raw files to process.

        Returns:
            DirectoryLoader: A loader configured for handling unstructured files.
        """
        try:
            return DirectoryLoader(
                path,
                loader_cls=lambda p: UnstructuredFileLoader(
                    p,
                    post_processors=[
                        lambda text: clean(
                            text, bullets=True, lowercase=True, extra_whitespace=True,
                            dashes=True, trailing_punctuation=True
                        )
                    ]
                )
            )
        except Exception as e:
            logging.error(f"Error creating loader for {path}: {e}")
            return None

    def load_documents(self):
        """
        Loads documents using the provided loader.
        
        Returns:
            list: A list of loaded documents.
        """
        return self.loader.load()

    def split_documents(self, documents):
        """
        Splits loaded documents into smaller chunks based on specified separators.
        
        Args:
            documents (list): The documents to split.
        
        Returns:
            list: A list of document chunks.
        """
        return self.text_splitter.split_documents(documents)


class EmbeddingManager:
    """
    Manages embedding generation and vector storage operations.
    """
    def __init__(self, collection_path):
        """
        Initializes the EmbeddingManager with an embedding function.
        
        Args:
            collection_path (str): The collection path of vectorstore.
        """
        try:
            self.collection_path = collection_path
            self.embedding_function = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")
            self.vectorstore = None
            self.namespace = f"chromadb/{collection_path}"
            self.record_manager = SQLRecordManager(
                self.namespace, db_url="sqlite:///record_manager_cache.sql"
            )
        except Exception as e:
            logging.error(f"Failed to initialize EmbeddingManager: {e}")
            raise
        
    def setup_vectorstore(self, docs_split, persist=True):
        """
        Sets up or loads a vectorstore and stores the embedded document chunks.
        
        Args:
            docs_split (list): The document chunks to store.
            persist (bool): Whether to persist the store on disk.
        """
        if persist and os.path.isdir(self.collection_path):
            self.vectorstore = Chroma(persist_directory=self.collection_path, embedding_function=self.embedding_function)
        else:
            self.vectorstore = Chroma.from_documents(docs_split, self.embedding_function, persist_directory=self.collection_path)
            self.record_manager.create_schema()

    def clear_index(self):
        """
        Clears old duplicated records from the storage by performing a cleanup operation.
        """
        try:
            index(docs_source=[], record_manager=self.record_manager, vector_store=self.vectorstore, cleanup="full", source_id_key="source")
            logging.info("Index cleared successfully.")
        except Exception as e:
            logging.error(f"Error clearing the index: {e}")
            
    def add_index(self, docs_split):
        """
        Adds new document chunks to the vectorstore and updates the index.
        """
        try:
            index(docs_source=docs_split, record_manager=self.record_manager, vector_store=self.vectorstore, cleanup="full", source_id_key="source")
            logging.info("Index updated successfully.")
        except Exception as e:
            logging.error(f"Error updating the index: {e}")      

def main():
    # Document processing
    document_handler = DocumentHandler(file_path="raw")
    documents = document_handler.load_documents()
    documents_split = document_handler.split_documents(documents=documents)

    # Embedding and vector storage management
    embedding_manager = EmbeddingManager(collection_path="chroma_index")
    embedding_manager.setup_vectorstore(docs_split=documents_split, persist=True)
    embedding_manager.clear_index()
    embedding_manager.add_index(docs_split=documents_split)


if __name__ == "__main__":
    main()