"""
Example Usage of QueryHandler from the Query Handling Module.

This module demonstrates the use of the QueryHandler class defined in query.py, showing how to interact with the query processing capabilities provided by Azure Cognitive Services through a simple text query example.
This example makes use of OpenAI Embeddings.
 
Requirements:
- Proper setup and authentication with Azure Cognitive Services as detailed in the query.py module are necessary before running this example.
- The Azure CLI must be authenticated using 'az login' to interact with Azure services effectively.

Usage:
Run this script directly to see how the QueryHandler class is utilized to send a query and receive a response. The main function showcases the setup and execution of a text query, making it a practical example for understanding module integration.
"""

from src.query import QueryHandler
import os
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from azure.identity import AzureCliCredential
import sys
sys.path.append('.')

# Set up environment for Azure Cognitive Services
load_dotenv()
credential = AzureCliCredential()
os.environ["OPENAI_API_KEY"] = credential.get_token(
    "https://cognitiveservices.azure.com/.default").token


def main():
    """
    Main function to handle querying with predefined settings.
    Initializes the QueryHandler with specific configurations and sends a query to get an answer.
    """
    query_handler = QueryHandler(embedding_function=AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002"),
                                 llm_name="ChatGpt",
                                 collection_name="chroma_index",
                                 )

    # Example query
    query = 'what is the optimal whey casein ratio for infants'
    result = query_handler.handle_question(query)

    print(result)


if __name__ == "__main__":
    main()
