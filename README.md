# LocalRAG

## Table of Contents

- [Structure](#structure)
- [Installation](#installation)
- [Documentation](#documentation)

## Structure

- `src/`: Source code for the project.
    - `record_manager_embedding.py`: Module for document processing and embedding management using record manager.
    - `path_embedding.py`: Module for document processing and embedding management by updating json file of documents that have been embedded.
    - `query.py`: Module for creating and managing conversational retrieval chains to fetch relevant information.
    - `app.py`: Module for creating Streamlit chatbot interface.
- `examples/`: Example scripts demonstrating the usage of GENAI.
    - `example.py`: A sample script showing the output of model based on query given.
- `docs/`: HTML documentation for the library.
- `embedding_models/`: Open-source text embedding models from HuggingFace.
    - `baai/`: Models from BAAI.
    - `sentence-transformers/`: Models from Sentence Transformers.
- `tests/`: Test suite for the GENAI library.
- `requirements.txt`: Lists all the Python dependencies for the project.
- `raw/`: Directory where raw source documents reside.
- `chroma_index/`: Directory of persisted Chroma vectorstore.
- `archive/`: Notebooks used for testing.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/fwenyin/GenAI.git
cd [repository directory]
pip install -r requirements.txt
```

To embed new documents, add the documents to the raw folder and run the path_embedding.py or record_embedding.py file.
- If vectorstore already exists, ensure that embedding function used remains constant. Else, replace the pre-existing vectorstore. 
- Current chroma_index is created with Huggingface Embeddings using path_embedding.py module. Archive chroma_index is created with OpenAI Embeddings.

To query the existing vectorstore, use QueryHandler in query.py. Example of use is in examples.py.

To run the Streamlit chatbot interface, run:
```bash
streamlit run src/app.py
```

## Documentation
The full documentation is available in the docs/ directory. Open the index.html file in a web browser to access it.