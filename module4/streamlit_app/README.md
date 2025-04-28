# Document Q&A System

A Streamlit-based application for document question answering using Retrieval-Augmented Generation (RAG).

## Features

- 📄 Upload and process multiple document formats (PDF, TXT, DOCX, CSV)
- 🔍 Ask questions about your documents and get accurate answers
- 📊 View document analysis and metadata
- 📝 Generate document summaries
- 📤 Export conversation history and document collection

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone this repository or download the source code

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

1. (Optional) Create a `.env` file in the application directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. (Optional) Configure Streamlit settings in the `.streamlit/config.toml` file

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Upload documents using the "Document Upload" tab

4. Ask questions about your documents in the "Question Answering" tab

5. Explore document analysis, summaries, and export options in the other tabs

## Document Processing

The application supports the following document formats:

- PDF (.pdf)
- Text (.txt)
- Microsoft Word (.docx)
- CSV (.csv)

## RAG System

The application uses a Retrieval-Augmented Generation (RAG) system with the following components:

- Document processing and chunking
- Embedding generation using sentence-transformers
- Vector storage using FAISS or ChromaDB
- Question answering using Groq API (or simulated responses if no API key is provided)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses sentence-transformers for embeddings
- Uses FAISS/ChromaDB for vector search
- Uses Groq API for text generation
