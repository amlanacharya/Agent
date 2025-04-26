"""
Test Script for Personalized Retrieval System
--------------------------------------------
This script demonstrates and tests the PersonalizedRetrievalSystem class
implemented in personalized_retrieval.py.
"""

import os
import time
import json
from module2.exercises.personalized_retrieval import PersonalizedRetrievalSystem


class MockMemorySystem:
    """A simple mock memory system for testing the personalized retrieval"""

    def __init__(self):
        """Initialize with some test data"""
        self.items = {
            "item1": {
                "id": "item1",
                "text": "Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data.",
                "metadata": {"category": "AI", "complexity": "beginner"}
            },
            "item2": {
                "id": "item2",
                "text": "Natural language processing (NLP) helps computers understand, interpret, and generate human language in a valuable way.",
                "metadata": {"category": "NLP", "complexity": "intermediate"}
            },
            "item3": {
                "id": "item3",
                "text": "Vector databases are specialized database systems optimized for similarity search operations on high-dimensional vectors.",
                "metadata": {"category": "Databases", "complexity": "advanced"}
            },
            "item4": {
                "id": "item4",
                "text": "Large language models like GPT-4 use transformer architectures with attention mechanisms to process and generate text.",
                "metadata": {"category": "NLP", "complexity": "advanced"}
            },
            "item5": {
                "id": "item5",
                "text": "Python is a popular programming language for AI development due to its simplicity and extensive libraries.",
                "metadata": {"category": "Programming", "complexity": "beginner"}
            },
            "item6": {
                "id": "item6",
                "text": "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment to maximize rewards.",
                "metadata": {"category": "AI", "complexity": "intermediate"}
            },
            "item7": {
                "id": "item7",
                "text": "Deep learning uses neural networks with many layers to learn hierarchical representations of data.",
                "metadata": {"category": "AI", "complexity": "intermediate"}
            }
        }

    def retrieve(self, query, top_k=5):
        """
        Simple keyword-based retrieval

        Args:
            query (str): The search query
            top_k (int): Number of results to return

        Returns:
            list: Top k matching items
        """
        results = []
        query_terms = query.lower().split()

        for item_id, item in self.items.items():
            text = item["text"].lower()
            # Count how many query terms appear in the text
            matches = sum(term in text for term in query_terms)
            if matches > 0:
                # Create a copy of the item with a score
                result = item.copy()
                result["score"] = matches / len(query_terms)
                results.append(result)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def print_separator(title):
    """Print a section separator with title"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")


def main():
    """Main test function"""
    # Clean up any existing test files
    test_profile_path = "test_user_profiles.json"
    if os.path.exists(test_profile_path):
        os.remove(test_profile_path)

    print_separator("PERSONALIZED RETRIEVAL SYSTEM TEST")

    # Create the memory system and retrieval system
    memory = MockMemorySystem()
    retrieval_system = PersonalizedRetrievalSystem(memory, test_profile_path)

    # Create test user profiles
    print("Creating user profiles...")

    # Beginner user interested in AI and Python
    retrieval_system.create_user_profile("beginner_user", {
        "interests": ["AI", "machine learning", "Python"],
        "expertise_level": "beginner"
    })

    # Expert user interested in NLP and transformers
    retrieval_system.create_user_profile("expert_user", {
        "interests": ["NLP", "language models", "transformers"],
        "expertise_level": "expert"
    })

    # Intermediate user interested in databases
    retrieval_system.create_user_profile("db_user", {
        "interests": ["databases", "vector search", "indexing"],
        "expertise_level": "intermediate"
    })

    print("User profiles created successfully.")

    # Test query
    test_query = "AI and machine learning"
    print(f"\nTest query: '{test_query}'")

    # Standard retrieval (no personalization)
    print_separator("STANDARD RETRIEVAL (NO PERSONALIZATION)")
    standard_results = retrieval_system.retrieve(test_query, top_k=3)

    for i, result in enumerate(standard_results):
        print(f"Result {i+1}:")
        print(f"- {result['text']}")
        print(f"- Score: {result.get('score', 0):.2f}")
        print()

    # Personalized retrieval for beginner AI user
    print_separator("PERSONALIZED RETRIEVAL FOR BEGINNER AI USER")
    beginner_results = retrieval_system.retrieve(test_query, "beginner_user", top_k=3)

    for i, result in enumerate(beginner_results):
        print(f"Result {i+1}:")
        print(f"- {result['text']}")
        print(f"- Original score: {result.get('original_score', 0):.2f}")
        print(f"- Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"- Personalization factors: {result['personalization_factors']}")
        print()

    # Personalized retrieval for expert NLP user
    print_separator("PERSONALIZED RETRIEVAL FOR EXPERT NLP USER")
    expert_results = retrieval_system.retrieve(test_query, "expert_user", top_k=3)

    for i, result in enumerate(expert_results):
        print(f"Result {i+1}:")
        print(f"- {result['text']}")
        print(f"- Original score: {result.get('original_score', 0):.2f}")
        print(f"- Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"- Personalization factors: {result['personalization_factors']}")
        print()

    # Test user feedback
    print_separator("TESTING USER FEEDBACK")
    print("Recording feedback from beginner user...")

    # Record positive feedback for an item
    retrieval_system.record_user_feedback("beginner_user", "item1", "helpful")
    print("Recorded 'helpful' feedback for item1")

    # Record negative feedback for an item
    retrieval_system.record_user_feedback("beginner_user", "item3", "not_helpful")
    print("Recorded 'not_helpful' feedback for item3")

    # Retrieve again to see the effect of feedback
    print("\nResults after feedback:")
    after_feedback_results = retrieval_system.retrieve(test_query, "beginner_user", top_k=3)

    for i, result in enumerate(after_feedback_results):
        print(f"Result {i+1}:")
        print(f"- {result['text']}")
        print(f"- Original score: {result.get('original_score', 0):.2f}")
        print(f"- Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"- Personalization factors: {result['personalization_factors']}")
        print()

    # Test content adaptation
    print_separator("TESTING CONTENT ADAPTATION")
    test_content = "Vector embeddings represent semantic meaning in high-dimensional space."

    # Adapt for beginner
    beginner_content = retrieval_system.adapt_to_expertise(test_content, "beginner_user")
    print(f"Content adapted for beginner:\n{beginner_content}\n")

    # Adapt for expert
    expert_content = retrieval_system.adapt_to_expertise(test_content, "expert_user")
    print(f"Content adapted for expert:\n{expert_content}\n")

    # Test retrieval with explanation
    print_separator("RETRIEVAL WITH EXPLANATION")
    response = retrieval_system.retrieve_with_explanation(test_query, "beginner_user", top_k=2)

    print("Results:")
    for i, result in enumerate(response['results']):
        print(f"Result {i+1}: {result['text']}")

    print("\nExplanation:")
    explanation = response['explanation']
    for key, value in explanation.items():
        print(f"- {key}: {value}")

    # Clean up
    if os.path.exists(test_profile_path):
        os.remove(test_profile_path)

    print_separator("TEST COMPLETED")


if __name__ == "__main__":
    main()
