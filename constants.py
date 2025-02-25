"""
Constants used throughout the application.
"""

# Languages
EN = "en"
VI = "vi"
NONE = "None"
ENGLISH = "English"
VIETNAMESE = "Vietnamese"

# Chat roles
USER = "user"
ASSISTANT = "assistant"

# LLM types
LOCAL_LLM = "local_llm"
ONLINE_LLM = "online_llm"
DEFAULT_LOCAL_LLM = "gemma2:2b"

# Data sources
UPLOAD = "UPLOAD"
DB = "DB"

# Chunking options
NO_CHUNKING = "No Chunking"
RECURSIVE_CHUNKER = "RecursiveTokenChunker"
SEMANTIC_CHUNKER = "SemanticChunker"
AGENTIC_CHUNKER = "AgenticChunker"

# Ollama models
OLLAMA_MODEL_OPTIONS = {
    "DeepSeek R1 1.5B": "deepseek-r1:5b",
    "DeepSeek R1 7B": "deepseek-r1:7b",
    "DeepSeek R1 14B": "deepseek-r1:14b",
    "Llama 3.2 (3B - 2.0GB)": "llama3.2",
    "Llama 3.2 (1B - 1.3GB)": "llama3.2:1b",
    "Llama 3.1 (8B - 4.7GB)": "llama3.1",
    "Llama 3.1 (70B - 40GB)": "llama3.1:70b",
    "Llama 3.1 (405B - 231GB)": "llama3.1:405b",
    "Phi 3 Mini (3.8B - 2.3GB)": "phi3",
    "Phi 3 Medium (14B - 7.9GB)": "phi3:medium",
    "Gemma 2 (2B - 1.6GB)": "gemma2:2b",
    "Gemma 2 (9B - 5.5GB)": "gemma2",
    "Gemma 2 (27B - 16GB)": "gemma2:27b",
    "Mistral (7B - 4.1GB)": "mistral",
    "Moondream 2 (1.4B - 829MB)": "moondream",
    "Neural Chat (7B - 4.1GB)": "neural-chat",
    "Starling (7B - 4.1GB)": "starling-lm",
    "Code Llama (7B - 3.8GB)": "codellama",
    "Llama 2 Uncensored (7B - 3.8GB)": "llama2-uncensored",
    "LLaVA (7B - 4.5GB)": "llava",
    "Solar (10.7B - 6.1GB)": "solar"
}

# Search options
VECTOR_SEARCH = "Vector Search"
HYDE_SEARCH = "Hyde Search"

# Default settings
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 10
DEFAULT_NUM_DOCS_RETRIEVAL = 3