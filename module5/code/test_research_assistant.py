"""
Test script for the Research Literature Assistant.

This script demonstrates the basic functionality of the Research Literature Assistant
by creating an instance and running some simple tests.
"""

import os
import sys
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the Research Literature Assistant
sys.path.insert(0, current_dir)
from research_assistant import ResearchLiteratureAssistant

def main():
    """Run a simple test of the Research Literature Assistant."""

    # Check if GROQ_API_KEY is set
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY environment variable not set. Please set it before running this script.")
        print("You can set it with: $env:GROQ_API_KEY = 'your-api-key'")
        return

    # Initialize language model using LangChain's ChatGroq
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2,
        api_key=api_key
    )

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Create the Research Literature Assistant
    print("Initializing Research Literature Assistant...")
    assistant = ResearchLiteratureAssistant(llm, embedding_model)

    # Test question analysis
    print("\n--- Testing Question Analysis ---")
    question = "What are the recent advances in transformer models for natural language processing?"
    print(f"Question: {question}")

    analysis = assistant.question_analyzer.analyze_question(question)
    print("\nQuestion Analysis:")
    print(f"- Type: {analysis.get('question_type', 'Unknown')}")
    print(f"- Key Concepts: {', '.join(analysis.get('key_concepts', []))}")
    print(f"- Complexity: {analysis.get('complexity', 'Unknown')}")

    # Test search query generation
    print("\n--- Testing Search Query Generation ---")
    search_queries = assistant.question_analyzer.generate_search_queries(question)
    print("Generated Search Queries:")
    for i, query in enumerate(search_queries, 1):
        print(f"{i}. {query}")

    # Note about adding papers
    print("\n--- Adding Papers ---")
    print("To fully test the assistant, you would need to add academic papers:")
    print("assistant.add_paper('path/to/paper.pdf')")

    # Check if there's a papers directory with PDFs
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
    if os.path.exists(papers_dir):
        pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
        if pdf_files:
            print(f"\nFound {len(pdf_files)} PDF files in the 'papers' directory.")
            print("Adding the first paper for demonstration...")
            paper_path = os.path.join(papers_dir, pdf_files[0])
            try:
                paper_data = assistant.add_paper(paper_path)
                print(f"Successfully added paper: {paper_data['metadata'].get('title', 'Unknown Title')}")

                # Build index
                print("\nBuilding vector index...")
                assistant.build_index()

                # Test literature review generation
                print("\n--- Testing Literature Review Generation ---")
                print("Generating a simple literature review...")
                review = assistant.generate_literature_review("transformer models")
                print("\nLiterature Review Preview (first 500 chars):")
                print(review[:500] + "..." if len(review) > 500 else review)

                # Test paper summarization
                print("\n--- Testing Paper Summarization ---")
                summaries = assistant.summarize_papers()
                for paper_id, summary in summaries.items():
                    print(f"\nSummary for paper {paper_id}:")
                    print(summary[:300] + "..." if len(summary) > 300 else summary)

            except Exception as e:
                print(f"Error processing paper: {e}")
        else:
            print("No PDF files found in the 'papers' directory.")
    else:
        print("No 'papers' directory found. Create one and add academic PDFs to fully test the assistant.")

    print("\nTest completed.")

if __name__ == "__main__":
    main()
