# Oura 📚 - A Retrieval-Augmented Generation (RAG) Based Document Assistant

<img width="1889" height="927" alt="Screenshot 2026-06-26 170749" src="https://github.com/user-attachments/assets/85a45e60-7896-4677-bef8-fe715812405e" />


A RAG (Retrieval Augmented Generation) application that answers questions based on uploaded PDF documents using AI.

## ✨ Features

- Upload and index PDF documents
- Ask questions about your documents
- Get answers with source page references
- Chat history tracking
- MMR retrieval for relevant results

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for embeddings)
- Mistral AI API key (for responses)

## 🚀 Quick Start

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
```

### 3. Run
```bash
streamlit run app.py
```

## 📖 How to Use

1. **Upload PDFs** - Click "Upload PDFs" in the sidebar and select one or more PDF files
2. **Create Knowledge Base** - Click "Create Knowledge Base" to process and index the documents
3. **Ask Questions** - Type your question in the chat input
4. **View Sources** - See which page and file the answer came from

## ⚙️ Configuration

Edit in `create_db.py`:
```python
chunk_size = 1000        # Characters per text chunk
chunk_overlap = 200      # Overlap between chunks
```

Edit in `app.py`:
```python
"k": 4                   # Number of documents to retrieve
"fetch_k": 10           # Initial fetch count
"lambda_mult": 0.5      # Relevance vs diversity balance
```

## 🔧 Troubleshooting

**OpenAI API Error (401)** → Check `.env` file has valid OpenAI API key

**"Knowledge base not found"** → Upload PDF and create knowledge base first

**Slow responses** → Reduce `k` value or increase `chunk_size`

**Wrong pages showing** → Delete `chroma-db/` folder and recreate knowledge base

## 📦 Project Structure

```
RAG PROJECT/
├── app.py                 # Main Streamlit app
├── create_db.py          # Database creation
├── main.py               # CLI version
├── requirements.txt      # Dependencies
├── .env                  # API keys (create this)
├── document_loaders/     # PDF storage
├── chroma-db/            # Vector database
└── Readme.md             # This file
```

## 📚 Stack

- **Streamlit** - UI
- **LangChain** - LLM orchestration
- **Chroma** - Vector database
- **OpenAI** - Embeddings
- **Mistral AI** - LLM responses
- **PyPDF** - PDF processing

---


