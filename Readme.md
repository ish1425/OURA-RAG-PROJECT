# Oura 📚 - RAG Document Analysis Platform

<img width="1889" height="921" alt="Screenshot 2026-06-26 124739" src="https://github.com/user-attachments/assets/579b99c8-642d-4b44-bfea-00a44445617a" />

A powerful Retrieval Augmented Generation (RAG) application that enables intelligent document analysis and context-aware question-answering. Upload PDFs, ask questions, and get accurate answers backed by source citations.

## 🌟 Key Features

- **📄 Smart Document Management** - Upload and process PDFs with automatic chunking and embedding generation
- **💬 Intelligent Q&A** - Ask natural language questions and get context-aware responses with MMR retrieval
- **📎 Source Attribution** - Every answer includes relevant document excerpts and sources for transparency
- **📊 Document Statistics** - Track uploaded documents, chunks, and manage your knowledge base
- **🔄 Chat History** - Full conversation history with persistent message tracking
- **⚡ Fast Processing** - Optimized embedding generation and retrieval using OpenAI and Mistral AI

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher**
- **OpenAI API key** (for embeddings)
- **Mistral AI API key** (for LLM responses)
- **pip** (Python package manager)

## 🚀 Quick Start

### 1. Clone or Download the Project
```bash
cd "c:\Users\DELL\Desktop\RAG PROJECT"
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-key-here
MISTRAL_API_KEY=your-mistral-key-here
```

### 5. Initialize the Database
```bash
python create_db.py
```

### 6. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents
1. Click **"Upload a PDF"** in the sidebar
2. Select your PDF file
3. Click **"Create Knowledge Base"** to process the document
4. Wait for the "Creating embeddings..." spinner to complete
5. You'll see "Knowledge Base Created!" confirmation


### Managing Your Knowledge Base
- **📝 Clear Chat** - Use sidebar button to clear message history
- **📚 New Upload** - Upload a different PDF (replaces current knowledge base)
- **🔄 Persistent History** - Chat history is maintained within your session

### Best Practices
- Ask specific questions about document content
- Use clear language for better retrieval
- Reference specific sections when asking follow-ups

## 🏗️ Architecture

```
Oura Pipeline:
├── PDF Upload (Streamlit UI)
├── PyPDF Loader (PDF extraction)
├── RecursiveCharacterTextSplitter (1000 chars + 200 overlap)
├── OpenAI Embeddings (Vector generation)
├── Chroma Vector Store (Persistent storage)
├── MMR Retrieval (k=4, fetch_k=10, lambda_mult=0.5)
├── Mistral AI Small (mistral-small-2506 LLM)
└── Streamlit Chat Interface (Response display)
```

### Data Flow
```
PDF → Chunks → Embeddings → Vector Store
                               ↓
                          User Question
                               ↓
                       Retriever (MMR) → Context
                               ↓
                         Mistral LLM
                               ↓
                          AI Response
```

## 📂 Project Structure

```
RAG PROJECT/
├── app.py                          # Main Streamlit application (entry point)
├── create_db.py                    # Database initialization script
├── main.py                         # Core application logic
├── main_draft.py                   # Alternative implementation
├── requirements.txt                # Python dependencies
├── Readme.md                       # This file
├── .env                            # Environment variables (create this)
│
├── document_loaders/               # PDF processing & storage
│   ├── pdf.py                      # PDF extraction utilities
│   ├── recursive_text.py           # Text chunking strategies
│   ├── page.py                     # Page-level processing
│   ├── test.py                     # Testing utilities
│   └── *.pdf                       # Uploaded PDFs stored here
│
├── vector_store/                   # Vector database
│   ├── db.py                       # Chroma vector store management
│   └── chroma-db/                  # Vector embeddings (persistent)
│
├── images/                         # Documentation images
│   └── oura-demo.png              # Application screenshot
│
└── uploaded_pdfs/                  # Legacy PDF storage (alternative)
    └── (optional file storage)
```

## ⚙️ Configuration

Your app.py uses these optimized settings:

```python
# Text Processing
CHUNK_SIZE = 1000              # Characters per text chunk
CHUNK_OVERLAP = 200            # Character overlap between chunks

# MMR Retrieval Strategy
search_type = "mmr"            # Maximum Marginal Relevance for diverse results
k = 4                          # Number of results to return
fetch_k = 10                   # Number of initial results to fetch
lambda_mult = 0.5              # Balance between relevance and diversity

# LLM Model
model = "mistral-small-2506"   # Mistral AI small model optimized for speed

# Storage Locations
pdf_directory = "document_loaders/"
vector_store = "chroma-db/"
```


## 📦 Dependencies

Key libraries used:
- **Streamlit** - Web UI framework
- **LangChain** - LLM orchestration
- **Chroma** - Vector database
- **PyPDF2** - PDF processing
- **OpenAI** - Embeddings
- **Mistral** - LLM responses

See `requirements.txt` for complete list.

```


---


## File Structure

```
oura/
├── oura_app.py           # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── chroma-db/            # Vector database (auto-created)
└── uploaded_pdfs/        # Temporary PDF storage (auto-created)
```



### LLM Model
Change the model in the `get_llm()` function:
```python
return ChatMistralAI(model="mistral-large-2406")  # For better quality
```

### Retrieval Strategy
Modify the retriever configuration:
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Options: "similarity", "mmr"
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)
```






