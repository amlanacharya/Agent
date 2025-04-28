"""
Tests for Embedding Pipelines
---------------------------
This module contains tests for the embedding_pipelines module.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from typing import List, Dict, Any

from embedding_pipelines import (
    BaseEmbeddings,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
    EmbeddingPipeline,
    cosine_similarity,
    evaluate_embedding_model,
    get_embedding_model
)


class TestHashEmbeddings(unittest.TestCase):
    """Test cases for HashEmbeddings."""

    def setUp(self):
        """Set up test fixtures."""
        self.embeddings = HashEmbeddings(dimension=384)

    def test_embed_text(self):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = self.embeddings.embed_text(text)

        # Check dimensions
        self.assertEqual(len(embedding), 384)

        # Check normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is a completely different sentence."
        ]
        embeddings = self.embeddings.embed_documents(texts)

        # Check number of embeddings
        self.assertEqual(len(embeddings), 3)

        # Check dimensions
        for emb in embeddings:
            self.assertEqual(len(emb), 384)

        # Check that similar texts have similar embeddings
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        self.assertGreater(sim_1_2, sim_1_3)

    def test_get_dimension(self):
        """Test getting the embedding dimension."""
        dimension = self.embeddings.get_dimension()
        self.assertEqual(dimension, 384)


class TestSentenceTransformerEmbeddings(unittest.TestCase):
    """Test cases for SentenceTransformerEmbeddings."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a try-except block to handle the case where sentence_transformers is not available
        try:
            self.embeddings = SentenceTransformerEmbeddings()
            self.has_sentence_transformers = True
        except:
            self.has_sentence_transformers = False

    def test_embed_text(self):
        """Test embedding a single text."""
        if not self.has_sentence_transformers:
            self.skipTest("sentence_transformers not available")

        text = "This is a test sentence."
        embedding = self.embeddings.embed_text(text)

        # Check that we got an embedding
        self.assertIsNotNone(embedding)

        # Check dimensions
        dimension = self.embeddings.get_dimension()
        self.assertEqual(len(embedding), dimension)

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        if not self.has_sentence_transformers:
            self.skipTest("sentence_transformers not available")

        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is a completely different sentence."
        ]
        embeddings = self.embeddings.embed_documents(texts)

        # Check number of embeddings
        self.assertEqual(len(embeddings), 3)

        # Check dimensions
        dimension = self.embeddings.get_dimension()
        for emb in embeddings:
            self.assertEqual(len(emb), dimension)

        # Check that similar texts have similar embeddings
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        self.assertGreater(sim_1_2, sim_1_3)


class TestOpenAIEmbeddings(unittest.TestCase):
    """Test cases for OpenAIEmbeddings."""

    def setUp(self):
        """Set up test fixtures."""
        # Check if API key is available
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.has_api_key = self.api_key is not None

        # Initialize with fallback to ensure tests run even without API key
        self.embeddings = OpenAIEmbeddings()

    def test_embed_text_fallback(self):
        """Test embedding a single text with fallback."""
        text = "This is a test sentence."
        embedding = self.embeddings.embed_text(text)

        # Check that we got an embedding
        self.assertIsNotNone(embedding)

        # Check dimensions (fallback uses 384)
        self.assertEqual(len(embedding), 384)

    def test_embed_documents_fallback(self):
        """Test embedding multiple documents with fallback."""
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is a completely different sentence."
        ]
        embeddings = self.embeddings.embed_documents(texts)

        # Check number of embeddings
        self.assertEqual(len(embeddings), 3)

        # Check dimensions (fallback uses 384)
        for emb in embeddings:
            self.assertEqual(len(emb), 384)


class TestEmbeddingPipeline(unittest.TestCase):
    """Test cases for EmbeddingPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()

        # Use HashEmbeddings for consistent testing
        self.embedding_model = HashEmbeddings()
        self.pipeline = EmbeddingPipeline(
            embedding_model=self.embedding_model,
            batch_size=2,
            cache_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_preprocess(self):
        """Test text preprocessing."""
        text = "  This   is  a test   sentence.  "
        processed = self.pipeline.preprocess(text)

        # Check that extra whitespace is removed
        self.assertEqual(processed, "This is a test sentence.")

    def test_embed_text_with_cache(self):
        """Test embedding a single text with caching."""
        text = "This is a test sentence."

        # First embedding should not be cached
        embedding1 = self.pipeline.embed_text(text)

        # Second embedding should be retrieved from cache
        embedding2 = self.pipeline.embed_text(text)

        # Embeddings should be identical
        self.assertEqual(embedding1, embedding2)

        # Check cache file exists
        cache_key = self.pipeline._get_cache_key(self.pipeline.preprocess(text))
        cache_file = os.path.join(self.temp_dir, f"{cache_key}.json")
        self.assertTrue(os.path.exists(cache_file))

    def test_embed_documents_with_cache(self):
        """Test embedding multiple documents with caching."""
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is a completely different sentence."
        ]

        # First batch should not be cached
        embeddings1 = self.pipeline.embed_documents(texts)

        # Second batch should be retrieved from cache
        embeddings2 = self.pipeline.embed_documents(texts)

        # Embeddings should be identical
        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertEqual(emb1, emb2)

    def test_clear_cache(self):
        """Test clearing the cache."""
        text = "This is a test sentence."

        # Generate embedding to populate cache
        self.pipeline.embed_text(text)

        # Clear cache
        self.pipeline.clear_cache()

        # Check in-memory cache is empty
        self.assertEqual(len(self.pipeline.cache), 0)

        # Check disk cache is empty
        cache_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        self.assertEqual(len(cache_files), 0)


class TestEvaluationUtilities(unittest.TestCase):
    """Test cases for evaluation utilities."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0)

        # Orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0)

        # Opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0)

    def test_evaluate_embedding_model(self):
        """Test embedding model evaluation."""
        # Create a simple embedding model
        model = HashEmbeddings()

        # Create test pairs
        test_pairs = [
            {"text1": "cats are pets", "text2": "dogs are pets", "expected_similar": True},
            {"text1": "cats are pets", "text2": "quantum physics is complex", "expected_similar": False}
        ]

        # Evaluate model
        results = evaluate_embedding_model(model, test_pairs)

        # Check results
        self.assertEqual(results["total_pairs"], 2)
        self.assertIn("accuracy", results)
        self.assertIn("execution_time", results)


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""

    def test_get_embedding_model(self):
        """Test getting embedding models by type."""
        # Test hash embeddings
        model = get_embedding_model("hash", dimension=512)
        self.assertIsInstance(model, HashEmbeddings)
        self.assertEqual(model.get_dimension(), 512)

        # Test sentence transformer embeddings
        model = get_embedding_model("sentence_transformer")
        self.assertIsInstance(model, SentenceTransformerEmbeddings)

        # Test OpenAI embeddings
        model = get_embedding_model("openai")
        self.assertIsInstance(model, OpenAIEmbeddings)

        # Test invalid type
        with self.assertRaises(ValueError):
            get_embedding_model("invalid_type")


if __name__ == "__main__":
    unittest.main()
