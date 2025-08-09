# RAG Tutorial Project

This project demonstrates several Retrieval-Augmented Generation (RAG) pipelines using Python and popular open-source libraries. It provides examples for document ingestion, vector indexing, and conversational querying with Large Language Models (LLMs).

## Features
- **PDF/Text Ingestion:** Load documents from the `data/` folder (PDFs, text files).
- **Vector Indexing:** Index documents using HuggingFace embeddings and ChromaDB or LlamaIndex.
- **Conversational Querying:** Query indexed documents using Groq's Llama3 model.
- **Multiple Pipelines:**
  - `rag_pipeline.py`: Advanced pipeline with persistent ChromaDB storage and similarity postprocessing.
  - `simple_rag.py`: Minimal pipeline for quick experimentation.
  - `rag_prompt.py`: Uses LlamaParse for PDF extraction and LangChain for RAG.

## Folder Structure
```
rag_pipeline.py         # Advanced RAG pipeline with ChromaDB
simple_rag.py          # Simple RAG pipeline with LlamaIndex
rag_prompt.py          # RAG pipeline using LangChain and LlamaParse
requirements.txt       # Python dependencies
/data/                 # Source documents (PDFs, etc.)
```

## Setup
1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Create a `.env` file in the root directory with your API keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     LLAMA_API_KEY=your_llama_api_key
     ```

## Usage
- **Run the advanced pipeline:**
  ```powershell
  python rag_pipeline.py
  ```
- **Run the simple pipeline:**
  ```powershell
  python simple_rag.py
  ```
- **Run the LangChain pipeline:**
  ```powershell
  python rag_prompt.py
  ```

## Notes
- Place your source documents (PDFs, text files) in the `data/` folder.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
This project is for educational purposes.
