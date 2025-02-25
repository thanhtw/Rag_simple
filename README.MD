## Example of RAG implementation

## Overview

## [Application Screenshot]![Image](https://github.com/user-attachments/assets/cbfcc5bd-7c64-4627-9376-b240ccb3d370)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline, enabling users to upload various data files (CSV, JSON, PDF, DOCX), store their content in a **Chroma vector store**, and interact with it via a chatbot. The chatbot is local models accessible through **OLLAMA**, retrieving relevant information and using **Large Language Models (LLMs)** to enhance responses based on user queries.

## Features
1. **Flexible File Upload** – Supports uploading CSV, JSON, PDF, and DOCX files, allowing users to choose which columns or sections to index.
2. **Chroma-based Storage and Retrieval** – Uses **Chroma** to store vector embeddings and perform efficient vector-based searches.
3. **Interactive Chatbot** – Chatbot interaction is enhanced by local LLMs to generate context-aware responses.
4. **Customizable LLM Choices** – The local LLMs like **OLLAMA**, with support for various open-source models.
5. **Dynamic Chunking Options** – Provides multiple chunking strategies: **Recursive Token Chunking**, **Agentic Chunking**, **Semantic Chunking**, or no chunking.
## Running the Application

### 1. Clone the Repository
```bash
git clone https://github.com/thanhtw/Rag_simple.git
cd Rag_simple
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Usage Instructions

### Step 1: Upload Data
Upload a CSV, JSON, PDF, or DOCX file. You can specify which columns to index for vector-based search.

### Step 2: Embedding and Storage
The data is stored in **Chroma**, and embeddings are generated using models like `all-MiniLM-L6-v2` (for English) or `keepitreal/vietnamese-sbert` (for Vietnamese).

### Step 3: Choose LLM
Select from:
- **Local LLMs via OLLAMA**, supporting models like `llama`, `gpt-j`, and more.

### Step 4: Configure Chunking
Select a chunking method to organize the content:
- **No Chunking**: Use the entire document.
- **Recursive Token Chunking**: Divide text based on token count.
- **Semantic Chunking**: Group text semantically.
- **Agentic Chunking**: Use an LLM to dynamically manage text chunks (requires Gemini API).

### Step 5: Interact with the Chatbot
Start chatting with the bot, which will enhance responses using the retrieved content.

## Supported Models for OLLAMA

Here is a list of models supported by **OLLAMA**:

| Model Name                           | Size          | Identifier               |
|--------------------------------------|---------------|--------------------------|
| Llama 3.2 (3B)                       | 3B (2.0GB)    | `llama3.2`               |
| Phi 3 Medium (14B)                   | 14B (7.9GB)   | `phi3:medium`            |
| Code Llama (7B)                      | 7B (3.8GB)    | `codellama`              |
| Mistral (7B)                         | 7B (4.1GB)    | `mistral`                |
| ...                                  | ...           | ...                      |



### Search Methods
Choose from:
- **Vector Search**: Based on vector similarity.

## Troubleshooting
- **No Results?** Ensure you've indexed the correct columns and stored embeddings.
