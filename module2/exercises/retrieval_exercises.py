"""
Module 2: Memory Systems - Retrieval Exercises
---------------------------------------------------
This file contains solutions for the practice exercises from Lesson 3.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import Counter


class QueryExpansionSystem:
    """
    Exercise 1: A system that expands queries with synonyms and related terms
    to improve retrieval recall.
    """

    def __init__(self, memory_system, synonym_dict=None):
        """
        Initialize the query expansion system

        Args:
            memory_system: The memory system to search (must have a retrieve method)
            synonym_dict (dict, optional): Dictionary of synonyms for common terms
        """
        self.memory_system = memory_system

        # Initialize synonym dictionary if not provided
        self.synonym_dict = synonym_dict or {
            # Common AI/ML terms
            "ai": ["artificial intelligence", "machine intelligence", "computer intelligence"],
            "ml": ["machine learning", "statistical learning", "predictive modeling"],
            "nlp": ["natural language processing", "text processing", "language understanding"],
            "llm": ["large language model", "language model", "generative model"],

            # Common action terms
            "create": ["generate", "build", "develop", "make"],
            "analyze": ["examine", "study", "investigate", "review"],
            "improve": ["enhance", "upgrade", "optimize", "refine"],

            # Common descriptive terms
            "good": ["effective", "quality", "excellent", "superior"],
            "fast": ["quick", "rapid", "swift", "high-speed"],
            "important": ["critical", "essential", "key", "vital"]
        }

    def expand_query(self, query: str, expansion_types: List[str] = None) -> List[str]:
        """
        Generate multiple variations of the original query

        Args:
            query (str): The original search query
            expansion_types (list, optional): Types of expansions to apply
                Options: 'synonym', 'specification', 'generalization', 'instruction'

        Returns:
            list: List of expanded query variations
        """
        # Default to all expansion types if none specified
        if expansion_types is None:
            expansion_types = ['synonym', 'specification', 'generalization', 'instruction']

        # Start with the original query
        expanded_queries = [query]

        # Tokenize the query
        query_terms = self._tokenize(query)

        # Apply different expansion types
        if 'synonym' in expansion_types:
            # Add synonym-based expansions
            synonym_queries = self._generate_synonym_expansions(query_terms)
            expanded_queries.extend(synonym_queries)

        if 'specification' in expansion_types:
            # Add more specific versions of the query
            expanded_queries.append(f"detailed information about {query}")
            expanded_queries.append(f"specific examples of {query}")
            expanded_queries.append(f"{query} detailed explanation")

        if 'generalization' in expansion_types:
            # Add more general versions of the query
            expanded_queries.append(f"overview of {query}")
            expanded_queries.append(f"{query} basics")
            expanded_queries.append(f"{query} concepts")

        if 'instruction' in expansion_types:
            # Add instruction-style queries
            expanded_queries.append(f"how to use {query}")
            expanded_queries.append(f"explain {query}")
            expanded_queries.append(f"define {query}")

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in expanded_queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)

        return unique_queries

    def retrieve(self, query: str, top_k: int = 5, expansion_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve information using query expansion

        Args:
            query (str): The original search query
            top_k (int): Number of results to return
            expansion_types (list, optional): Types of expansions to apply

        Returns:
            list: Top k results from expanded queries
        """
        # Generate expanded queries
        expanded_queries = self.expand_query(query, expansion_types)

        # Collect results from all queries
        all_results = []
        seen_ids = set()  # To track unique results

        # Keep track of which query found each result
        result_sources = {}  # result_id -> list of queries that found it

        # Allocate a smaller top_k for each query to avoid too many results
        per_query_top_k = max(3, top_k // len(expanded_queries))

        for expanded_query in expanded_queries:
            # Get results for this query
            try:
                results = self.memory_system.retrieve(expanded_query, top_k=per_query_top_k)

                for result in results:
                    # Check if we've already seen this result
                    result_id = result.get('id', '')

                    if not result_id:
                        # Generate an ID if none exists
                        result_id = f"result_{len(all_results)}"
                        result['id'] = result_id

                    # Track which query found this result
                    if result_id not in result_sources:
                        result_sources[result_id] = []
                    result_sources[result_id].append(expanded_query)

                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        all_results.append(result)
                    else:
                        # If we've seen this result before, update its score
                        # The more queries that find a result, the more relevant it likely is
                        for existing_result in all_results:
                            if existing_result.get('id') == result_id:
                                existing_score = existing_result.get('score', 0)
                                new_score = result.get('score', 0)
                                # Use the maximum score found
                                existing_result['score'] = max(existing_score, new_score)
                                break

            except Exception as e:
                print(f"Error retrieving results for query '{expanded_query}': {e}")

        # Add information about which queries found each result
        for result in all_results:
            result_id = result.get('id', '')
            if result_id in result_sources:
                result['found_by_queries'] = result_sources[result_id]
                # Boost score based on how many queries found this result
                result['score'] = result.get('score', 0) * (1 + 0.1 * len(result_sources[result_id]))

        # Re-rank combined results by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Return top k unique results
        return all_results[:top_k]

    def retrieve_with_explanation(self, query: str, top_k: int = 5, expansion_types: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve information using query expansion and provide explanations

        Args:
            query (str): The original search query
            top_k (int): Number of results to return
            expansion_types (list, optional): Types of expansions to apply

        Returns:
            dict: Results and explanations
        """
        # Generate expanded queries
        expanded_queries = self.expand_query(query, expansion_types)

        # Get results
        results = self.retrieve(query, top_k, expansion_types)

        # Create explanation
        explanation = {
            'original_query': query,
            'expanded_queries': expanded_queries,
            'expansion_types_used': expansion_types or ['synonym', 'specification', 'generalization', 'instruction'],
            'num_unique_results_found': len(results)
        }

        # Add query-specific explanations
        query_explanations = {}
        for expanded_query in expanded_queries:
            if expanded_query == query:
                explanation_text = "Original query"
            elif any(term in expanded_query for term in self._tokenize(query)):
                if "detailed" in expanded_query or "specific" in expanded_query:
                    explanation_text = "More specific version of the original query"
                elif "overview" in expanded_query or "basics" in expanded_query:
                    explanation_text = "More general version of the original query"
                elif "how to" in expanded_query or "explain" in expanded_query:
                    explanation_text = "Instruction-style version of the original query"
                else:
                    explanation_text = "Variation with synonym substitution"
            else:
                explanation_text = "Alternative phrasing"

            query_explanations[expanded_query] = explanation_text

        explanation['query_explanations'] = query_explanations

        # Add result-specific explanations
        for result in results:
            queries_that_found_it = result.get('found_by_queries', [])
            if query in queries_that_found_it:
                result['explanation'] = "Found by your original query"
            elif len(queries_that_found_it) > 1:
                result['explanation'] = f"Found by {len(queries_that_found_it)} query variations, suggesting high relevance"
            else:
                result['explanation'] = f"Found by the query variation: '{queries_that_found_it[0]}'"

        return {
            'results': results,
            'explanation': explanation
        }

    def add_synonyms(self, term: str, synonyms: List[str]):
        """
        Add new synonyms to the synonym dictionary

        Args:
            term (str): The term to add synonyms for
            synonyms (list): List of synonyms for the term
        """
        term = term.lower()
        if term in self.synonym_dict:
            # Add new synonyms to existing list
            current_synonyms = set(self.synonym_dict[term])
            for synonym in synonyms:
                current_synonyms.add(synonym.lower())
            self.synonym_dict[term] = list(current_synonyms)
        else:
            # Create new entry
            self.synonym_dict[term] = [s.lower() for s in synonyms]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text (str): Text to tokenize

        Returns:
            list: List of words
        """
        # Simple tokenization for demonstration
        text = text.lower()
        # Remove punctuation
        for char in '.,;:!?"\'()[]{}':
            text = text.replace(char, ' ')

        # Split into words and filter out stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                      'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about'}

        words = [word for word in text.split() if word not in stop_words and len(word) > 1]

        return words

    def _generate_synonym_expansions(self, query_terms: List[str]) -> List[str]:
        """
        Generate query variations by substituting synonyms

        Args:
            query_terms (list): Tokenized query terms

        Returns:
            list: List of query variations with synonyms
        """
        if not query_terms:
            return []

        # Find terms that have synonyms
        expansions = []

        # Try replacing one term at a time with its synonyms
        for i, term in enumerate(query_terms):
            if term in self.synonym_dict:
                for synonym in self.synonym_dict[term]:
                    # Create a new query with this term replaced by its synonym
                    new_query_terms = query_terms.copy()
                    new_query_terms[i] = synonym
                    expansions.append(" ".join(new_query_terms))

        return expansions


# Example usage
if __name__ == "__main__":
    # Create a simple mock memory system for testing
    class MockMemorySystem:
        def __init__(self):
            self.items = {
                "item1": {
                    "id": "item1",
                    "text": "Machine learning is a subset of artificial intelligence",
                    "metadata": {"category": "AI"}
                },
                "item2": {
                    "id": "item2",
                    "text": "Natural language processing helps computers understand human language",
                    "metadata": {"category": "NLP"}
                },
                "item3": {
                    "id": "item3",
                    "text": "Vector databases are optimized for similarity search",
                    "metadata": {"category": "Databases"}
                },
                "item4": {
                    "id": "item4",
                    "text": "Large language models can generate human-like text",
                    "metadata": {"category": "NLP"}
                },
                "item5": {
                    "id": "item5",
                    "text": "Artificial intelligence systems can learn from examples",
                    "metadata": {"category": "AI"}
                }
            }

        def retrieve(self, query, top_k=5):
            # Simple keyword matching for demonstration
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

    # Test the query expansion system
    print("Testing Query Expansion System...")
    memory = MockMemorySystem()
    expansion_system = QueryExpansionSystem(memory)

    # Test query expansion
    query = "AI learning"
    expanded_queries = expansion_system.expand_query(query)
    print(f"\nOriginal query: {query}")
    print("Expanded queries:")
    for i, q in enumerate(expanded_queries):
        print(f"  {i+1}. {q}")

    # Test retrieval with expansion
    print("\nRetrieval results with query expansion:")
    results = expansion_system.retrieve(query, top_k=3)
    for result in results:
        print(f"\n- {result['text']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Found by queries: {result.get('found_by_queries', [])}")

    # Test retrieval with explanation
    print("\nRetrieval with explanation:")
    response = expansion_system.retrieve_with_explanation(query, top_k=2)

    print("\nResults:")
    for result in response['results']:
        print(f"- {result['text']}")
        print(f"  Explanation: {result.get('explanation', '')}")

    print("\nQuery Expansion Explanation:")
    explanation = response['explanation']
    print(f"Original query: {explanation['original_query']}")
    print(f"Number of expanded queries: {len(explanation['expanded_queries'])}")
    print(f"Number of unique results: {explanation['num_unique_results_found']}")

    # Test adding custom synonyms
    expansion_system.add_synonyms("learning", ["training", "education", "knowledge acquisition"])
    expanded_queries = expansion_system.expand_query(query)
    print("\nExpanded queries after adding custom synonyms:")
    for i, q in enumerate(expanded_queries):
        print(f"  {i+1}. {q}")
