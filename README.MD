# Notebook

This project implements a very lightweight Retrieval-Augmented Generation (RAG) system for querying markdown documents.
It leverages [FAISS](https://ai.meta.com/tools/faiss/) for document retrieval and a LLM for generating context-aware
answers.

## How It Works

- **Document Retrieval:**  
  The system loads markdown files from a selected folder within the `notebooks` directory and builds (or loads) a FAISS
  index for fast similarity search.

- **Embedding Generation:**  
  A pre-trained Hugging Face model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)) generates
  embeddings used to compare and retrieve documents
  based on user queries.

- **LLM Integration:**  
  Retrieved documents are concatenated as context and passed to an LLM (currently claude-haiku-3 as it works well) to
  generate answers that
  include in-text references.

## How to Run

1. **Prepare Your Documents:**
    - Place your markdown files in a folder inside the `notebooks` directory.
    - Make sure its a folder inside the `notebooks` directory and not `.md` files directly.
    - You can keep multiple folders with markdown files inside the `notebooks` directory and pick the one you want to
      query during runtime.

2. **Install UV:**
    - Install UV
    - In your terminal run `uv sync`
    - Set an environment variable `ANTHROPIC_API_KEY` with your API key from [Anthropic](https://console.anthropic.com/)
    - In your terminal run `uv run python main.py`