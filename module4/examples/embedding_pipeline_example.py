"""
Embedding Pipeline Example
------------------------
This script demonstrates how to use the embedding pipelines module.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the code directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from embedding_pipelines import (
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
    EmbeddingPipeline,
    cosine_similarity,
    evaluate_embedding_model,
    get_embedding_model
)


def basic_embedding_example():
    """Basic example of using different embedding models."""
    print("\n=== Basic Embedding Example ===\n")

    # Sample text
    text = "Embeddings are numerical representations of text that capture semantic meaning."
    print(f"Sample text: '{text}'\n")

    # Try different embedding models
    models = [
        ("Hash Embeddings", HashEmbeddings()),
        ("Sentence Transformer", SentenceTransformerEmbeddings()),
        ("OpenAI Compatible", OpenAIEmbeddings())
    ]

    for name, model in models:
        try:
            # Generate embedding
            embedding = model.embed_text(text)

            # Print info
            print(f"{name}:")
            print(f"  Dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print(f"  Norm: {np.linalg.norm(embedding):.4f}")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}\n")


def similarity_comparison_example():
    """Example of comparing text similarities with embeddings."""
    print("\n=== Similarity Comparison Example ===\n")

    # Create embedding model
    model = SentenceTransformerEmbeddings()

    # Sample texts
    texts = [
        "Embeddings are numerical representations of text.",
        "Vector representations capture semantic meaning.",
        "Machine learning models use embeddings for NLP tasks.",
        "The weather today is sunny and warm.",
        "I enjoy hiking in the mountains on sunny days."
    ]

    # Generate embeddings
    try:
        embeddings = model.embed_documents(texts)

        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(len(texts)):
                similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

        # Print similarity matrix
        print("Similarity Matrix:")
        for i in range(len(texts)):
            for j in range(len(texts)):
                print(f"{similarity_matrix[i, j]:.2f}", end=" ")
            print()

        # Print most similar pair
        max_sim = 0
        max_pair = (0, 0)
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    max_pair = (i, j)

        print(f"\nMost similar pair (similarity: {max_sim:.2f}):")
        print(f"1. '{texts[max_pair[0]]}'")
        print(f"2. '{texts[max_pair[1]]}'")

    except Exception as e:
        print(f"Error: {e}")


def pipeline_with_caching_example():
    """Example of using the embedding pipeline with caching."""
    print("\n=== Embedding Pipeline with Caching Example ===\n")

    # Create a pipeline with caching
    pipeline = EmbeddingPipeline(
        embedding_model=HashEmbeddings(),  # Use hash embeddings for consistency
        batch_size=2,
        cache_dir="./embedding_cache",
        use_preprocessing=True
    )

    # Sample texts
    texts = [
        "This is the first example text.",
        "This is the second example text.",
        "This is the third example text.",
        "This is a completely different text."
    ]

    # First run - should generate new embeddings
    print("First run (generating embeddings)...")
    start_time = time.time()
    embeddings1 = pipeline.embed_documents(texts)
    first_run_time = time.time() - start_time
    print(f"  Time: {first_run_time:.4f} seconds")

    # Second run - should use cache
    print("\nSecond run (using cache)...")
    start_time = time.time()
    embeddings2 = pipeline.embed_documents(texts)
    second_run_time = time.time() - start_time
    print(f"  Time: {second_run_time:.4f} seconds")
    if second_run_time > 0:
        print(f"  Speedup: {first_run_time / second_run_time:.2f}x")
    else:
        print(f"  Speedup: Infinite (cached result returned instantly)")

    # Verify embeddings are the same
    all_same = all(np.array_equal(e1, e2) for e1, e2 in zip(embeddings1, embeddings2))
    print(f"  Embeddings identical: {all_same}")

    # Clear cache
    pipeline.clear_cache()
    print("\nCache cleared.")


def model_evaluation_example():
    """Example of evaluating embedding models."""
    print("\n=== Embedding Model Evaluation Example ===\n")

    # Create test pairs
    test_pairs = [
        {"text1": "Dogs are popular pets.", "text2": "Cats are common household pets.", "expected_similar": True},
        {"text1": "Machine learning is a subset of AI.", "text2": "Artificial intelligence includes machine learning.", "expected_similar": True},
        {"text1": "Python is a programming language.", "text2": "JavaScript is used for web development.", "expected_similar": True},
        {"text1": "The weather is sunny today.", "text2": "I enjoy hiking in the mountains.", "expected_similar": False},
        {"text1": "Quantum physics studies subatomic particles.", "text2": "Cooking involves preparing food with heat.", "expected_similar": False},
        {"text1": "Electric cars use batteries.", "text2": "The history of ancient Rome is fascinating.", "expected_similar": False}
    ]

    # Models to evaluate
    models = [
        ("Hash Embeddings", HashEmbeddings()),
        ("Sentence Transformer", SentenceTransformerEmbeddings())
    ]

    # Evaluate each model
    for name, model in models:
        try:
            print(f"Evaluating {name}...")
            results = evaluate_embedding_model(model, test_pairs)

            print(f"  Accuracy: {results['accuracy']:.2f}")
            print(f"  Correct predictions: {results['correct_predictions']}/{results['total_pairs']}")
            print(f"  False positives: {results['false_positives']}")
            print(f"  False negatives: {results['false_negatives']}")
            print(f"  Execution time: {results['execution_time']:.4f} seconds")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}\n")


if __name__ == "__main__":
    import time

    # Run examples
    basic_embedding_example()
    similarity_comparison_example()
    pipeline_with_caching_example()
    model_evaluation_example()

    print("\nAll examples completed.")
