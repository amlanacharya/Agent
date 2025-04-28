"""
Streamlit App for Document Q&A System
-----------------------------------
This app provides a user-friendly interface for the Document Q&A system,
allowing users to upload documents, ask questions, and view answers with sources.
"""

import os
import sys
import json
import tempfile
import logging
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import base64
import io
from typing import List, Dict, Any, Optional

# Set page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from rag_components import (
    SimpleRAGSystem,
    DocumentQASystem,
    SimpleEmbedding,
    GroqClient,
    SimpleLLMClient,
    process_document
)

# Page configuration is already set at the top of the file

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-item {
        background-color: #f5f5f5;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None


# Header
def render_header():
    """Render the app header."""
    st.markdown('<div class="main-header">📚 Document Q&A System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-text">Upload documents, ask questions, and get answers with source attribution.</div>',
        unsafe_allow_html=True
    )
    st.divider()


# Sidebar
def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.markdown("## ⚙️ Configuration")

    # API Key input
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")

    if api_key:
        if st.sidebar.button("Validate API Key"):
            try:
                # Try to initialize the client
                client = GroqClient(api_key=api_key)
                # Test with a simple query
                response = client.generate_text("Hello, this is a test.")
                if response:
                    st.sidebar.success("✅ API key is valid!")
                    st.session_state.api_key_valid = True
                    st.session_state.llm_client = client
                else:
                    st.sidebar.error("❌ API key validation failed.")
                    st.session_state.api_key_valid = False
            except Exception as e:
                st.sidebar.error(f"❌ API key validation failed: {str(e)}")
                st.session_state.api_key_valid = False
    else:
        st.sidebar.warning("⚠️ No API key provided. Using simulated responses.")
        st.session_state.llm_client = SimpleLLMClient()

    st.sidebar.divider()

    # Vector database selection
    vector_db = st.sidebar.selectbox(
        "Vector Database",
        ["FAISS", "ChromaDB"],
        index=0,
        help="Select the vector database to use for document retrieval."
    )

    # Embedding model selection
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["SentenceTransformer", "Hash-based (Fallback)"],
        index=0,
        help="Select the embedding model to use for document embeddings."
    )

    # Number of results
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve for each question."
    )

    # Use hybrid search
    use_hybrid = st.sidebar.checkbox(
        "Use Hybrid Search",
        value=True,
        help="Combine semantic search with keyword search for better results."
    )

    st.sidebar.divider()

    # System status
    st.sidebar.markdown("## 📊 System Status")

    # Document count
    doc_count = len(st.session_state.documents)
    st.sidebar.metric("Documents Loaded", doc_count)

    # Embedding count
    embedding_count = len(st.session_state.embeddings)
    st.sidebar.metric("Embeddings Generated", embedding_count)

    # RAG system status
    rag_status = "Ready" if st.session_state.rag_system else "Not Initialized"
    st.sidebar.markdown(f"**RAG System:** {rag_status}")

    # QA system status
    qa_status = "Ready" if st.session_state.qa_system else "Not Initialized"
    st.sidebar.markdown(f"**QA System:** {qa_status}")

    st.sidebar.divider()

    # About section
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info(
        "This app uses Retrieval-Augmented Generation (RAG) to answer questions "
        "based on your documents. Upload documents, ask questions, and get answers "
        "with source attribution."
    )

    # Dependencies note
    st.sidebar.markdown("## 📦 Dependencies")
    st.sidebar.warning(
        "For full functionality, install these optional dependencies:\n"
        "- `pip install PyPDF2` for PDF processing\n"
        "- `pip install python-docx` for DOCX processing\n"
        "- `pip install sentence-transformers` for better embeddings"
    )

    return {
        "api_key": api_key,
        "vector_db": vector_db.lower(),
        "embedding_model": embedding_model,
        "top_k": top_k,
        "use_hybrid": use_hybrid
    }


# Document upload
def render_document_upload():
    """Render the document upload section."""
    st.markdown('<div class="sub-header">📄 Document Upload</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX, CSV)",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Process each uploaded file
                new_documents = []
                new_contents = []

                for uploaded_file in uploaded_files:
                    try:
                        # Save the uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        # Process the document
                        doc = process_document(temp_path, uploaded_file.name)

                        if doc:
                            new_documents.append(doc)
                            new_contents.append(doc["content"])
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process: {uploaded_file.name}")

                        # Remove the temporary file
                        os.unlink(temp_path)

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

                # Add new documents to session state
                if new_documents:
                    st.session_state.documents.extend(new_documents)

                    # Generate embeddings for new documents
                    if not st.session_state.embedding_model:
                        st.session_state.embedding_model = SimpleEmbedding()

                    new_embeddings = st.session_state.embedding_model.embed_documents(new_contents)
                    st.session_state.embeddings.extend(new_embeddings)

                    st.success(f"✅ Added {len(new_documents)} documents with embeddings.")

                    # Reset RAG and QA systems to force reinitialization
                    st.session_state.rag_system = None
                    st.session_state.qa_system = None

    # Display loaded documents
    if st.session_state.documents:
        with st.expander("View Loaded Documents", expanded=False):
            for i, doc in enumerate(st.session_state.documents):
                st.markdown(f"**Document {i+1}:** {doc.get('metadata', {}).get('source', 'Unknown')}")
                st.markdown(f"**Content Preview:** {doc['content'][:200]}...")
                st.markdown("---")

        # Option to clear documents
        if st.button("Clear All Documents"):
            st.session_state.documents = []
            st.session_state.embeddings = []
            st.session_state.rag_system = None
            st.session_state.qa_system = None
            st.success("✅ All documents cleared.")
            st.rerun()


# Initialize RAG system
def initialize_rag_system(config):
    """Initialize the RAG system if not already initialized."""
    if not st.session_state.documents or not st.session_state.embeddings:
        st.warning("⚠️ Please upload and process documents first.")
        return False

    if not st.session_state.rag_system:
        with st.spinner("Initializing RAG system..."):
            try:
                # Create RAG system
                st.session_state.rag_system = SimpleRAGSystem(
                    st.session_state.documents,
                    st.session_state.embeddings,
                    vector_store_type=config["vector_db"]
                )

                # Create QA system
                st.session_state.qa_system = DocumentQASystem(
                    rag_system=st.session_state.rag_system,
                    embedding_model=st.session_state.embedding_model,
                    llm_client=st.session_state.llm_client
                )

                st.success("✅ RAG system initialized successfully.")
                return True

            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {str(e)}")
                return False

    return True


# Question answering
def render_qa_interface(config):
    """Render the question answering interface."""
    st.markdown('<div class="sub-header">❓ Ask Questions</div>', unsafe_allow_html=True)

    # Check if RAG system is initialized
    if not initialize_rag_system(config):
        return

    # Question input
    question = st.text_input("Enter your question about the documents")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        answer_btn = st.button("Get Answer")

    with col2:
        synthesis_btn = st.button("Get Synthesized Answer")

    with col3:
        metadata_btn = st.button("Query Metadata")

    # Process question
    if question and (answer_btn or synthesis_btn or metadata_btn):
        with st.spinner("Generating answer..."):
            try:
                if answer_btn:
                    # Get regular answer
                    response = st.session_state.qa_system.answer_question(
                        question,
                        k=config["top_k"],
                        use_hybrid=config["use_hybrid"]
                    )
                    response_type = "standard"

                elif synthesis_btn:
                    # Get synthesized answer
                    response = st.session_state.qa_system.answer_with_synthesis(
                        question,
                        k=config["top_k"]
                    )
                    response_type = "synthesis"

                elif metadata_btn:
                    # Force metadata query
                    analysis = st.session_state.rag_system.analyze_question(question)
                    analysis["is_metadata_query"] = True

                    metadata_results = st.session_state.rag_system.retrieve_metadata(question)
                    answer = st.session_state.rag_system.answer_metadata_query(
                        question, metadata_results, st.session_state.llm_client
                    )

                    response = {
                        "answer": answer,
                        "sources": [r["metadata"] for r in metadata_results[:3]],
                        "is_metadata_query": True,
                        "confidence": 0.9 if metadata_results else 0.1
                    }
                    response_type = "metadata"

                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response,
                    "response_type": response_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="sub-header">💬 Conversation History</div>', unsafe_allow_html=True)

        for i, item in enumerate(reversed(st.session_state.chat_history)):
            # Question
            st.markdown(f"**Question:** {item['question']}")

            # Answer
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown(f"**Answer:** {item['response']['answer']}")

            # Confidence score
            if 'confidence' in item['response']:
                confidence = item['response']['confidence']
                if confidence >= 0.8:
                    confidence_class = "confidence-high"
                    confidence_label = "High"
                elif confidence >= 0.5:
                    confidence_class = "confidence-medium"
                    confidence_label = "Medium"
                else:
                    confidence_class = "confidence-low"
                    confidence_label = "Low"

                st.markdown(
                    f"**Confidence:** <span class='{confidence_class}'>{confidence_label} ({confidence:.2f})</span>",
                    unsafe_allow_html=True
                )

            # Response type
            response_type_label = {
                "standard": "Standard Answer",
                "synthesis": "Synthesized from Multiple Sources",
                "metadata": "Metadata Query"
            }.get(item['response_type'], "Unknown")

            st.markdown(f"**Response Type:** {response_type_label}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Sources
            if 'sources' in item['response'] and item['response']['sources']:
                with st.expander("View Sources", expanded=False):
                    for j, source in enumerate(item['response']['sources']):
                        if isinstance(source, dict):
                            st.markdown('<div class="source-item">', unsafe_allow_html=True)
                            source_name = source.get('source', 'Unknown')
                            st.markdown(f"**Source {j+1}:** {source_name}")

                            # Display other metadata
                            for key, value in source.items():
                                if key != 'source':
                                    st.markdown(f"**{key.capitalize()}:** {value}")

                            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

        # Option to clear history
        if st.button("Clear Conversation History"):
            st.session_state.chat_history = []
            st.success("✅ Conversation history cleared.")
            st.rerun()


# Document analysis
def render_document_analysis():
    """Render the document analysis section."""
    st.markdown('<div class="sub-header">📊 Document Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.documents:
        st.warning("⚠️ Please upload and process documents first.")
        return

    # Document statistics
    st.markdown("### 📈 Document Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Documents", len(st.session_state.documents))

    with col2:
        total_tokens = sum(len(doc["content"].split()) for doc in st.session_state.documents)
        st.metric("Total Words", total_tokens)

    with col3:
        unique_sources = len(set(doc.get("metadata", {}).get("source", "") for doc in st.session_state.documents))
        st.metric("Unique Sources", unique_sources)

    # Metadata analysis
    st.markdown("### 🏷️ Metadata Analysis")

    # Collect metadata fields
    metadata_fields = set()
    for doc in st.session_state.documents:
        metadata = doc.get("metadata", {})
        metadata_fields.update(metadata.keys())

    # Display metadata field coverage
    if metadata_fields:
        st.markdown("#### Metadata Field Coverage")

        field_coverage = {}
        for field in metadata_fields:
            count = sum(1 for doc in st.session_state.documents if field in doc.get("metadata", {}))
            percentage = (count / len(st.session_state.documents)) * 100
            field_coverage[field] = percentage

        # Sort by coverage
        sorted_fields = sorted(field_coverage.items(), key=lambda x: x[1], reverse=True)

        # Display as a bar chart
        field_names = [field for field, _ in sorted_fields]
        field_values = [value for _, value in sorted_fields]

        # Create a DataFrame for the bar chart
        chart_data = pd.DataFrame({
            "Field": field_names,
            "Coverage (%)": field_values
        })

        st.bar_chart(chart_data, x="Field", y="Coverage (%)")

        # Display common values for each field
        st.markdown("#### Common Metadata Values")

        selected_field = st.selectbox("Select Metadata Field", list(metadata_fields))

        if selected_field:
            # Collect values for the selected field
            field_values = [
                str(doc.get("metadata", {}).get(selected_field, ""))
                for doc in st.session_state.documents
                if selected_field in doc.get("metadata", {})
            ]

            # Count occurrences
            value_counts = {}
            for value in field_values:
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1

            # Display as a table
            if value_counts:
                value_data = {
                    "Value": list(value_counts.keys()),
                    "Count": list(value_counts.values()),
                    "Percentage": [
                        f"{(count / len(field_values)) * 100:.1f}%"
                        for count in value_counts.values()
                    ]
                }

                st.dataframe(value_data)
            else:
                st.info("No values found for this field.")
    else:
        st.info("No metadata fields found in the documents.")

    # Content analysis
    st.markdown("### 📝 Content Analysis")

    # Word frequency
    st.markdown("#### Word Frequency")

    # Collect all words
    all_words = []
    for doc in st.session_state.documents:
        words = doc["content"].lower().split()
        all_words.extend(words)

    # Filter out common words
    common_words = {"the", "and", "a", "to", "of", "in", "is", "it", "that", "for", "on", "with", "as", "by", "this", "be", "are", "an", "or", "at", "from"}
    filtered_words = [word for word in all_words if word not in common_words and len(word) > 3]

    # Count occurrences
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Get top words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    # Display as a bar chart
    if top_words:
        word_names = [word for word, _ in top_words]
        word_values = [count for _, count in top_words]

        # Create a DataFrame for the bar chart
        word_chart_data = pd.DataFrame({
            "Word": word_names,
            "Frequency": word_values
        })

        st.bar_chart(word_chart_data, x="Word", y="Frequency")
    else:
        st.info("No words to analyze.")


# Document summary
def render_document_summary():
    """Render the document summary section."""
    st.markdown('<div class="sub-header">📝 Document Summaries</div>', unsafe_allow_html=True)

    if not st.session_state.documents:
        st.warning("⚠️ Please upload and process documents first.")
        return

    if not st.session_state.qa_system:
        st.warning("⚠️ Please initialize the QA system first.")
        return

    # Select document to summarize
    document_sources = [doc.get("metadata", {}).get("source", f"Document {i+1}") for i, doc in enumerate(st.session_state.documents)]
    selected_source = st.selectbox("Select Document to Summarize", document_sources)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                # Find the document index
                doc_index = None
                for i, doc in enumerate(st.session_state.documents):
                    if doc.get("metadata", {}).get("source", f"Document {i+1}") == selected_source:
                        doc_index = i
                        break

                if doc_index is not None:
                    # Generate summary
                    summary = st.session_state.qa_system.get_document_summary(document_id=doc_index)

                    if "error" in summary:
                        st.error(f"❌ {summary['error']}")
                    else:
                        st.markdown('<div class="highlight">', unsafe_allow_html=True)
                        st.markdown(f"**Summary:** {summary['summary']}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Display metadata
                        if "metadata" in summary:
                            with st.expander("Document Metadata", expanded=False):
                                for key, value in summary["metadata"].items():
                                    st.markdown(f"**{key.capitalize()}:** {value}")
                else:
                    st.error("❌ Document not found.")

            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")


# Export functionality
def render_export_section():
    """Render the export section."""
    st.markdown('<div class="sub-header">📤 Export Data</div>', unsafe_allow_html=True)

    # Export options
    export_option = st.radio(
        "Select what to export",
        ["Conversation History", "Document Collection", "System Configuration"]
    )

    if export_option == "Conversation History" and st.session_state.chat_history:
        # Prepare conversation history for export
        history_data = []
        for item in st.session_state.chat_history:
            history_data.append({
                "question": item["question"],
                "answer": item["response"]["answer"],
                "confidence": item["response"].get("confidence", "N/A"),
                "response_type": item["response_type"],
                "timestamp": item["timestamp"]
            })

        # Convert to JSON
        history_json = json.dumps(history_data, indent=2)

        # Create download button
        st.download_button(
            label="Download Conversation History (JSON)",
            data=history_json,
            file_name="conversation_history.json",
            mime="application/json"
        )

    elif export_option == "Document Collection" and st.session_state.documents:
        # Prepare document collection for export
        doc_data = []
        for doc in st.session_state.documents:
            doc_data.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            })

        # Convert to JSON
        doc_json = json.dumps(doc_data, indent=2)

        # Create download button
        st.download_button(
            label="Download Document Collection (JSON)",
            data=doc_json,
            file_name="document_collection.json",
            mime="application/json"
        )

    elif export_option == "System Configuration":
        # Prepare system configuration for export
        config_data = {
            "document_count": len(st.session_state.documents),
            "embedding_count": len(st.session_state.embeddings),
            "rag_system_initialized": st.session_state.rag_system is not None,
            "qa_system_initialized": st.session_state.qa_system is not None,
            "api_key_valid": st.session_state.api_key_valid,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Convert to JSON
        config_json = json.dumps(config_data, indent=2)

        # Create download button
        st.download_button(
            label="Download System Configuration (JSON)",
            data=config_json,
            file_name="system_configuration.json",
            mime="application/json"
        )

    else:
        st.info("No data available to export for the selected option.")


# Main app
def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    init_session_state()

    # Render header
    render_header()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Create tabs
    tabs = st.tabs([
        "📄 Document Upload",
        "❓ Question Answering",
        "📊 Document Analysis",
        "📝 Document Summaries",
        "📤 Export Data"
    ])

    # Tab 1: Document Upload
    with tabs[0]:
        render_document_upload()

    # Tab 2: Question Answering
    with tabs[1]:
        render_qa_interface(config)

    # Tab 3: Document Analysis
    with tabs[2]:
        render_document_analysis()

    # Tab 4: Document Summaries
    with tabs[3]:
        render_document_summary()

    # Tab 5: Export Data
    with tabs[4]:
        render_export_section()


if __name__ == "__main__":
    main()
