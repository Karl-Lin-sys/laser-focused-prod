# Multilingual RAG System (Canadian English, Japanese, Taiwanese Mandarin)

This project provides a lightweight and optimized Retrieval-Augmented Generation (RAG) system, designed to run smoothly even on older laptops with limited CPU and memory.

## Architecture

1. **Document Ingestion (`app/ingest.py`)**: 
   - Reads documents from the `data/` directory.
   - Utilizes a custom chunking strategy (`app/custom_splitter.py`) tailored for language-specific tokenization (Janome for Japanese, Jieba for Taiwanese Mandarin, standard splitting for English). This ensures morphological integrity is preserved.
   - Embeds the chunks using OpenAI Embeddings (via API to save local resources).
   - Stores the embeddings in a lightweight local vector database (ChromaDB).

2. **Retrieval & Generation (`app/app.py`)**:
   - Accepts a user query.
   - Retrieves the most relevant chunks from ChromaDB.
   - Generates an answer using a lightweight LLM API call (OpenAI) to avoid local computation overhead.

3. **Containerization (`Dockerfile` & `docker-compose.yml`)**:
   - Uses a multi-stage Docker build to keep the final image size minimal.
   - Resource limits are set in `docker-compose.yml` to prevent the container from freezing an older host machine.

4. **CI/CD (`.github/workflows/main.yml`)**:
   - GitHub Actions pipeline runs linting (flake8) and unit tests (pytest) on every push.
   - Automatically builds the Docker image if tests pass.

## Directory Structure

```
laser-focused-prod/
├── .github/
│   └── workflows/
│       └── main.yml        # CI/CD pipeline configuration
├── app/
│   ├── __init__.py
│   ├── app.py              # Main QA application
│   ├── ingest.py           # Data ingestion script
│   └── custom_splitter.py  # Language-aware chunking strategy
├── tests/
│   ├── __init__.py
│   └── test_tokenize.py    # Pytest for custom splitter
├── data/                   # Directory for source documents
├── chroma_db/              # Directory for ChromaDB storage
├── Dockerfile              # Multi-stage lightweight Dockerfile
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Run

1. Place your text documents in the `data/` directory.
2. Set your API key in an `.env` file or export it:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
3. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
