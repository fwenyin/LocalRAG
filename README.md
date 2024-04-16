# LocalRAG

## Table of Contents

- [Structure](#structure)
- [Installation](#installation)
- [Documentation](#documentation)

## Structure

- `src/`: Source code for the project.
    - `embedding.py`: Module for document processing and embedding management.
    - `query.py`: Module for creating and managing conversational retrieval chains to fetch relevant information.
- `examples/`: Example scripts demonstrating the usage of GENAI.
    - `example.py`: A sample script showing the output of model based on query given.
- `docs/`: Sphinx-generated HTML documentation for the library.
- `tests/`: Test suite for the GENAI library.
- `requirements.txt`: Lists all the Python dependencies for the project.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/fwenyin/GenAI.git
cd [repository directory]
pip install -r requirements.txt
```

## Documentation
The full documentation, which was generated with Sphinx, is available in the docs/ directory. Open the index.html file in a web browser to access it.