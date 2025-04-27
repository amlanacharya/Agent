"""
Test Script for Exercise 4.7.3: Knowledge Agent Validation
--------------------------------------------------------
This script tests the validation patterns for knowledge agents implemented in
exercise4_7_3_knowledge_agent_validator.py.
"""

import unittest
from datetime import datetime, timedelta
# Adjust the import path based on how you're running the script
try:
    # When running from the module3/exercises directory
    from exercise4_7_3_knowledge_agent_validator import (
        Source, Citation, KnowledgeQuery, KnowledgeResponse, KnowledgeAgent,
        KnowledgeAgentValidator, SourceType, FactCheckResult
    )
except ImportError:
    # When running from the project root
    from module3.exercises.exercise4_7_3_knowledge_agent_validator import (
        Source, Citation, KnowledgeQuery, KnowledgeResponse, KnowledgeAgent,
        KnowledgeAgentValidator, SourceType, FactCheckResult
    )


class TestKnowledgeAgentValidator(unittest.TestCase):
    """Test cases for knowledge agent validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a standard source for testing
        self.source = Source(
            name="Encyclopedia Britannica",
            url="https://www.britannica.com/place/Paris",
            source_type=SourceType.ENCYCLOPEDIA,
            reliability_score=0.9,
            last_updated=datetime.now() - timedelta(days=30)
        )

        # Create a standard citation for testing
        self.citation = Citation(
            source_id=self.source.id,
            text="Encyclopedia Britannica - Paris",
            url_fragment="history"
        )

        # Create a standard query for testing
        self.query = KnowledgeQuery(
            text="What is the capital of France?",
            query_type="factual",
            entities=["France"],
            keywords=["capital", "France"]
        )

        # Create a standard agent for testing
        self.agent = KnowledgeAgent(
            name="FactBot",
            description="A knowledge agent for factual queries",
            knowledge_domains=["general", "science", "history"],
            supported_query_types=["factual", "conceptual"],
            confidence_threshold=0.7,
            source_reliability_threshold=0.6,
            requires_citations=True
        )

        # Create a validator
        self.validator = KnowledgeAgentValidator()

    def test_source_validation(self):
        """Test source validation."""
        # Valid source
        valid_source = Source(
            name="Nature Journal",
            url="https://www.nature.com/articles/s41586-021-03275-y",
            author="Smith et al.",
            publication_date=datetime.now() - timedelta(days=60),
            source_type=SourceType.ACADEMIC,
            reliability_score=0.95,
            peer_reviewed=True
        )

        errors = self.validator.validate_source(valid_source)
        self.assertEqual(len(errors), 0, "Valid source should have no errors")

        # Invalid academic source (not peer-reviewed)
        invalid_academic = Source(
            name="Unpublished Paper",
            url="https://example.com/paper",
            author="Smith et al.",
            source_type=SourceType.ACADEMIC,
            reliability_score=0.7,
            peer_reviewed=False  # Not peer-reviewed
        )

        errors = self.validator.validate_source(invalid_academic)
        self.assertGreater(len(errors), 0, "Non-peer-reviewed academic source should have errors")

        # Invalid news source (no URL)
        invalid_news = Source(
            name="Daily News",
            url=None,  # Missing URL
            source_type=SourceType.NEWS,
            reliability_score=0.6
        )

        errors = self.validator.validate_source(invalid_news)
        self.assertGreater(len(errors), 0, "News source without URL should have errors")

    def test_query_validation(self):
        """Test query validation."""
        # Valid query
        valid_query = KnowledgeQuery(
            text="What are the effects of climate change on polar ice caps?",
            query_type="causal",
            entities=["climate change", "polar ice caps"],
            keywords=["effects", "climate", "change", "polar", "ice", "caps"]
        )

        errors = self.validator.validate_query(valid_query)
        self.assertEqual(len(errors), 0, "Valid query should have no errors")

        # Invalid query (too short)
        invalid_query = KnowledgeQuery(
            text="Why?",
            query_type="causal"
        )

        errors = self.validator.validate_query(invalid_query)
        self.assertGreater(len(errors), 0, "Too short query should have errors")

    def test_response_validation(self):
        """Test response validation."""
        # Valid response
        valid_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text=self.query.text,
            answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
            confidence=0.95,
            sources=[self.source],
            citations=[self.citation],
            fact_check_result=FactCheckResult.VERIFIED
        )

        # This should not raise an exception
        valid_response.validate_source_reliability()
        valid_response.validate_citations()

        # Invalid response (high confidence without sources)
        with self.assertRaises(ValueError):
            KnowledgeResponse(
                query_id=self.query.id,
                query_text=self.query.text,
                answer="The capital of France is Paris.",
                confidence=0.95,
                sources=[],  # No sources
                citations=[]  # No citations
            )

        # Invalid response (high confidence with unreliable sources)
        unreliable_source = Source(
            name="Random Blog",
            url="https://randomblog.com/france",
            source_type=SourceType.BLOG,
            reliability_score=0.3  # Low reliability
        )

        with self.assertRaises(ValueError):
            KnowledgeResponse(
                query_id=self.query.id,
                query_text=self.query.text,
                answer="The capital of France is Paris.",
                confidence=0.95,
                sources=[unreliable_source],
                citations=[]
            )

    def test_uncertainty_handling(self):
        """Test uncertainty handling validation."""
        # Valid uncertain response
        uncertain_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text="When exactly did humans first arrive in the Americas?",
            answer="The exact timing of human arrival in the Americas is still debated, but evidence suggests it may have been between 15,000 and 30,000 years ago [1].",
            confidence=0.6,
            sources=[self.source],
            citations=[self.citation],
            contains_uncertainty=True
        )

        # This should not raise an exception
        uncertain_response.validate_uncertainty_handling()

        # Invalid response (contains uncertainty markers but not flagged)
        with self.assertRaises(ValueError):
            KnowledgeResponse(
                query_id=self.query.id,
                query_text="When exactly did humans first arrive in the Americas?",
                answer="The exact timing of human arrival in the Americas is still debated, but evidence suggests it may have been between 15,000 and 30,000 years ago.",
                confidence=0.6,
                sources=[self.source],
                citations=[self.citation],
                contains_uncertainty=False  # Not flagged as uncertain
            )

    def test_opinion_handling(self):
        """Test opinion handling validation."""
        # Valid opinion response
        opinion_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text="What is the best programming language?",
            answer="In my opinion, Python is one of the best programming languages for beginners due to its readability and versatility [1].",
            confidence=0.7,
            sources=[self.source],
            citations=[self.citation],
            contains_opinion=True,
            fact_check_result=FactCheckResult.OPINION
        )

        # This should not raise an exception
        opinion_response.validate_opinion_handling()

        # Invalid response (contains opinion markers but not flagged)
        with self.assertRaises(ValueError):
            KnowledgeResponse(
                query_id=self.query.id,
                query_text="What is the best programming language?",
                answer="In my opinion, Python is one of the best programming languages for beginners due to its readability and versatility.",
                confidence=0.7,
                sources=[self.source],
                citations=[self.citation],
                contains_opinion=False  # Not flagged as opinion
            )

    def test_agent_response_validation(self):
        """Test agent validation of responses."""
        # Valid response for agent
        valid_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text=self.query.text,
            answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
            confidence=0.95,
            sources=[self.source],
            citations=[self.citation],
            fact_check_result=FactCheckResult.VERIFIED
        )

        errors = self.agent.validate_response(valid_response)
        self.assertEqual(len(errors), 0, "Valid response should have no errors for agent")

        # Invalid response for agent (confidence below threshold)
        low_confidence_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text=self.query.text,
            answer="The capital of France might be Paris, but I'm not entirely sure.",
            confidence=0.5,  # Below agent threshold
            sources=[self.source],
            citations=[self.citation],
            contains_uncertainty=True,
            contains_opinion=False  # No opinion markers
        )

        errors = self.agent.validate_response(low_confidence_response)
        self.assertGreater(len(errors), 0, "Low confidence response should have errors for agent")

        # Invalid response for agent (unsupported query type)
        procedural_query = KnowledgeQuery(
            text="How do I make a French omelette?",
            query_type="procedural"
        )

        procedural_response = KnowledgeResponse(
            query_id=procedural_query.id,
            query_text=procedural_query.text,
            answer="To make a French omelette: 1. Beat eggs. 2. Heat butter. 3. Cook eggs. 4. Fold and serve.",
            confidence=0.8,
            sources=[self.source],
            citations=[self.citation]
        )

        errors = self.agent.validate_response(procedural_response)
        self.assertGreater(len(errors), 0, "Response for unsupported query type should have errors")

    def test_fact_check_validation(self):
        """Test fact check validation."""
        # Valid verified fact
        verified_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text=self.query.text,
            answer="The capital of France is Paris.",
            confidence=0.95,
            sources=[self.source],
            citations=[self.citation],
            fact_check_result=FactCheckResult.VERIFIED
        )

        errors = self.validator.validate_fact_check(verified_response)
        self.assertEqual(len(errors), 0, "Valid verified fact should have no errors")

        # Invalid fact check (contradicted but high confidence)
        contradicted_response = KnowledgeResponse(
            query_id=self.query.id,
            query_text=self.query.text,
            answer="The capital of France is Lyon.",
            confidence=0.8,  # High confidence
            sources=[self.source],
            citations=[self.citation],
            fact_check_result=FactCheckResult.CONTRADICTED  # Contradicted
        )

        errors = self.validator.validate_fact_check(contradicted_response)
        self.assertGreater(len(errors), 0, "Contradicted fact with high confidence should have errors")

    def test_query_type_response_validation(self):
        """Test validation of responses for different query types."""
        # Valid factual response
        factual_query = KnowledgeQuery(
            text="What is the capital of France?",
            query_type="factual"
        )

        factual_response = KnowledgeResponse(
            query_id=factual_query.id,
            query_text=factual_query.text,
            answer="The capital of France is Paris.",
            confidence=0.95,
            sources=[self.source],
            citations=[self.citation]
        )

        errors = self.validator.validate_response_for_query_type(factual_response, factual_query)
        self.assertEqual(len(errors), 0, "Valid factual response should have no errors")

        # Valid procedural response
        procedural_query = KnowledgeQuery(
            text="How do I make a French omelette?",
            query_type="procedural"
        )

        procedural_response = KnowledgeResponse(
            query_id=procedural_query.id,
            query_text=procedural_query.text,
            answer="To make a French omelette: 1. Beat eggs. 2. Heat butter. 3. Cook eggs. 4. Fold and serve.",
            confidence=0.8,
            sources=[self.source],
            citations=[self.citation]
        )

        errors = self.validator.validate_response_for_query_type(procedural_response, procedural_query)
        self.assertEqual(len(errors), 0, "Valid procedural response should have no errors")

        # Invalid procedural response (no steps)
        invalid_procedural = KnowledgeResponse(
            query_id=procedural_query.id,
            query_text=procedural_query.text,
            answer="French omelettes are made with eggs, butter, and proper technique.",
            confidence=0.8,
            sources=[self.source],
            citations=[self.citation]
        )

        errors = self.validator.validate_response_for_query_type(invalid_procedural, procedural_query)
        self.assertGreater(len(errors), 0, "Procedural response without steps should have errors")


if __name__ == "__main__":
    unittest.main()
