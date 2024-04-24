"""
Document Processing and Embedding Management Module. 

This module is designed for document processing and embedding management, utilizing Azure Cognitive Services for its operations. It specifically leverages the Azure OpenAI service to generate embeddings for documents.
Dynamic path embedding by updating json file of documents that have been embedded, only documents that have new filenames will be embedded.

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

import json
from typing import Dict, List, Callable, Optional
from unstructured.cleaners.core import clean, group_broken_paragraphs
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import AzureCliCredential
import re
import os
import logging
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up environment for Azure Cognitive Services
load_dotenv()
credential = AzureCliCredential()
os.environ["OPENAI_API_KEY"] = credential.get_token(
    "https://cognitiveservices.azure.com/.default").token
# Set to True if using OpenAI embeddings, False for Hugging Face embeddings
IS_OPENAI_EMBEDDING = False


class DocumentHandler:
    """
    Handles operations related to document processing such as loading, preprocessing and splitting into manageable chunks.
    """

    def __init__(self, file_path: str, metadata_file: str = "metadata.json"):
        """
        Initializes the DocumentHandler with a file path for documents.

        Args:
            file_path (str): The path to the directory of raw files to process.
            metadata_file (str): Path to the JSON file storing document metadata.
        """
        self.file_path: str = file_path
        self.metadata_file: str = metadata_file
        self.loader: Optional[DirectoryLoader] = self.create_loader(file_path)
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                            chunk_overlap=50,
                                                            separators=[
                                                                '\n\n', '\n', '. ', '? ', '! '],
                                                            keep_separator=False)

    def load_metadata(self) -> Dict[str, any]:
        """
        Load metadata from a JSON file.

        Returns:
            dict: A dictionary containing the metadata of the documents. If no metadata file exists,
                returns an empty dictionary.
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_metadata(self, metadata: Dict[str, any]) -> None:
        """
        Save metadata to a JSON file.

         Args:
            metadata (dict): A dictionary containing the metadata for each document processed.
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def check_documents(self) -> List[str]:
        """
        Check each document to see if it needs embedding and update metadata for new documents.

        Returns:
            list: A list of file paths that are already present in the metadata, and will be excluded from the document loader.
        """
        metadata = self.load_metadata()
        documents_repeated = List[str] = []
        for filename in os.listdir(self.file_path):
            file_path = os.path.join(self.file_path, filename)
            if os.path.isfile(file_path) and filename not in metadata:
                metadata[filename] = {
                    'timestamp': os.path.getmtime(file_path)
                }
            else:
                documents_repeated.append(file_path)
        print(f"Documents repeated: {documents_repeated}")
        self.save_metadata(metadata)
        return documents_repeated

    def create_loader(self, path: str) -> Optional[DirectoryLoader]:
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
                exclude=self.check_documents(),
                loader_cls=lambda p: UnstructuredFileLoader(
                    p,
                    post_processors=[
                        lambda text: clean(text, bullets=True, lowercase=True, extra_whitespace=True,
                                           dashes=True, trailing_punctuation=True
                                           ),
                        lambda text: group_broken_paragraphs(
                            text, paragraph_split=re.compile(r"(\s*\n\s*){5}"))
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

    def __init__(self, collection_path, embedding_function):
        """
        Initializes the EmbeddingManager with an embedding function.

        Args:
            collection_path (str): The collection path of vectorstore.
            embedding_function (callable): Function to generate embeddings.
        """
        try:
            self.collection_path = collection_path
            self.embedding_function = embedding_function
            self.vectorstore = None
        except Exception as e:
            logging.error(f"Failed to initialize EmbeddingManager: {e}")
            raise

    def setup_vectorstore(self, docs_split, persist=True):
        """
        Sets up or loads a vectorstore and stores the embedded document chunks.

        Args:
            docs_split (list): Document chunks to embed and store.
            persist (bool):  If True, persist data on disk.
        """
        if persist and os.path.isdir(self.collection_path):
            self.vectorstore = Chroma(
                persist_directory=self.collection_path, embedding_function=self.embedding_function)
        else:
            self.vectorstore = Chroma.from_documents(
                docs_split, self.embedding_function, persist_directory=self.collection_path)
        self.vectorstore.persist()


def main():
    # Document processing
    document_handler = DocumentHandler(file_path="raw")
    documents = document_handler.load_documents()
    documents_split = document_handler.split_documents(documents=documents)
    print(len(documents_split))

    # Embedding and vector storage management
    if IS_OPENAI_EMBEDDING:
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002")
    else:
        embedding_function = HuggingFaceEmbeddings(
            model_name="embedding_models/baai/bge-small-en-v1.5")
    embedding_manager = EmbeddingManager(
        collection_path="chroma_index", embedding_function=embedding_function)
    embedding_manager.setup_vectorstore(
        docs_split=documents_split, persist=True)


if __name__ == "__main__":
    main()
