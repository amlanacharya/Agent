"""
Test suite for Lesson 4 exercises on Metadata Extraction & Management.
"""

import os
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

from lesson4_exercises import (
    DomainSpecificExtractor,
    MetadataEnhancedRetrieval,
    MetadataVisualizer,
    DocumentQualityScorer,
    EducationalContentMetadata
)

# Check if pydantic is available
try:
    from pydantic import ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class TestDomainSpecificExtractor(unittest.TestCase):
    """Test the DomainSpecificExtractor implementation."""

    def setUp(self):
        self.extractor = DomainSpecificExtractor()

        # Sample scientific paper text
        self.sample_paper = """
        Title: Effects of Climate Change on Biodiversity

        Abstract: This study examines the impact of climate change on biodiversity in tropical regions.
        We analyze data from 50 different ecosystems over a 10-year period.

        Introduction: Climate change represents one of the most significant threats to global biodiversity...

        Methods: We collected data from 50 tropical ecosystems between 2010 and 2020.
        Sampling was conducted quarterly using standardized protocols.

        Results: Our analysis shows a 15% decline in species diversity across all studied ecosystems.
        Figure 1 illustrates the correlation between temperature increase and biodiversity loss.
        Table 2 summarizes the changes in species composition over time.

        Discussion: These findings suggest that climate change is having a significant impact on...

        References:
        [1] Smith et al. (2018). Climate patterns and biodiversity. Nature, 556, 45-50.
        [2] Johnson and Lee (2019). Tropical ecosystem responses to climate variables. Science, 320, 1200-1205.
        """

    def test_extract_scientific_metadata(self):
        """Test scientific paper metadata extraction."""
        metadata = self.extractor.extract_scientific_metadata(self.sample_paper)

        # Check that the function returns a dictionary
        self.assertIsInstance(metadata, dict)

        # Check that required fields are present
        self.assertIn('has_abstract', metadata)
        self.assertIn('abstract', metadata)
        self.assertIn('has_methods', metadata)
        self.assertIn('methods_section', metadata)
        self.assertIn('has_results', metadata)
        self.assertIn('results_section', metadata)
        self.assertIn('citation_count', metadata)
        self.assertIn('citations', metadata)
        self.assertIn('figure_count', metadata)
        self.assertIn('table_count', metadata)

        # Check that the extractor correctly identified sections
        self.assertTrue(metadata['has_abstract'])
        self.assertTrue(metadata['has_methods'])
        self.assertTrue(metadata['has_results'])

        # Check citation count
        self.assertEqual(metadata['citation_count'], 2)

        # Check figure and table counts
        self.assertEqual(metadata['figure_count'], 1)
        self.assertEqual(metadata['table_count'], 1)


class TestMetadataEnhancedRetrieval(unittest.TestCase):
    """Test the MetadataEnhancedRetrieval implementation."""

    def setUp(self):
        self.retrieval = MetadataEnhancedRetrieval()

        # Sample documents with metadata
        self.documents = [
            {
                'id': 'doc1',
                'content': 'This document discusses artificial intelligence and machine learning applications.',
                'metadata': {
                    'author': 'John Smith',
                    'date': '2023-01-15',
                    'topics': ['AI', 'Machine Learning'],
                    'type': 'article',
                    'word_count': 150
                }
            },
            {
                'id': 'doc2',
                'content': 'Deep learning models have revolutionized computer vision and natural language processing.',
                'metadata': {
                    'author': 'Jane Doe',
                    'date': '2023-02-20',
                    'topics': ['Deep Learning', 'Computer Vision', 'NLP'],
                    'type': 'research',
                    'word_count': 200
                }
            },
            {
                'id': 'doc3',
                'content': 'Climate change is affecting biodiversity in tropical ecosystems.',
                'metadata': {
                    'author': 'John Smith',
                    'date': '2022-11-10',
                    'topics': ['Climate Change', 'Biodiversity'],
                    'type': 'article',
                    'word_count': 180
                }
            }
        ]

    def test_filter_documents(self):
        """Test metadata-based document filtering."""
        # Filter by author
        results = self.retrieval.filter_documents(
            self.documents,
            {'metadata.author': 'John Smith'}
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], 'doc1')
        self.assertEqual(results[1]['id'], 'doc3')

        # Filter by type and minimum word count
        results = self.retrieval.filter_documents(
            self.documents,
            {
                'metadata.type': 'article',
                'metadata.word_count': {'operator': 'gte', 'value': 180}
            }
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'doc3')

        # Filter by topics (list membership)
        results = self.retrieval.filter_documents(
            self.documents,
            {'metadata.topics': 'AI'}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'doc1')

    def test_hybrid_search(self):
        """Test hybrid search combining semantic search with metadata filtering."""
        # This test is simplified since we don't have a real embedding model
        query = "machine learning applications"

        # Basic hybrid search
        results = self.retrieval.hybrid_search(
            query,
            self.documents
        )
        self.assertIsInstance(results, list)

        # Hybrid search with filters
        results = self.retrieval.hybrid_search(
            query,
            self.documents,
            filters={'metadata.type': 'article'}
        )
        self.assertIsInstance(results, list)

        # If the implementation is complete, all results should be articles
        if results:
            for result in results:
                self.assertEqual(result['document']['metadata']['type'], 'article')

    def test_boost_by_metadata(self):
        """Test boosting relevance scores based on metadata."""
        query = "artificial intelligence"

        # Boost recent documents
        results = self.retrieval.boost_by_metadata(
            query,
            self.documents,
            boost_fields={'metadata.date': 1.5}  # Boost recent documents
        )
        self.assertIsInstance(results, list)

        # If the implementation is complete, doc2 should be ranked higher than doc1
        # despite doc1 having more relevant content, because doc2 is more recent
        if len(results) >= 2:
            # Check that the first result is doc2
            self.assertEqual(results[0]['document']['id'], 'doc2')


class TestMetadataVisualizer(unittest.TestCase):
    """Test the MetadataVisualizer implementation."""

    def setUp(self):
        self.visualizer = MetadataVisualizer()

        # Sample documents with metadata
        self.documents = [
            {
                'id': 'doc1',
                'metadata': {
                    'author': 'John Smith',
                    'date': '2023-01-15',
                    'topics': ['AI', 'Machine Learning'],
                    'type': 'article',
                    'word_count': 150,
                    'quality_score': 0.8
                }
            },
            {
                'id': 'doc2',
                'metadata': {
                    'author': 'Jane Doe',
                    'date': '2023-02-20',
                    'topics': ['Deep Learning', 'Computer Vision', 'NLP'],
                    'type': 'research',
                    'word_count': 200,
                    'quality_score': 0.9
                }
            },
            {
                'id': 'doc3',
                'metadata': {
                    'author': 'John Smith',
                    'topics': ['Climate Change', 'Biodiversity'],
                    'type': 'article',
                    'word_count': 180
                }
            }
        ]

    def test_analyze_metadata_coverage(self):
        """Test metadata coverage analysis."""
        coverage = self.visualizer.analyze_metadata_coverage(self.documents)

        # Check that the function returns a dictionary
        self.assertIsInstance(coverage, dict)

        # Check coverage calculations
        self.assertEqual(coverage.get('author', 0), 100.0)  # All documents have author
        self.assertEqual(coverage.get('date', 0), 66.67)    # 2/3 documents have date
        self.assertEqual(coverage.get('quality_score', 0), 66.67)  # 2/3 documents have quality_score

    def test_analyze_value_distribution(self):
        """Test metadata value distribution analysis."""
        author_distribution = self.visualizer.analyze_value_distribution(
            self.documents, 'metadata.author'
        )

        # Check that the function returns a dictionary
        self.assertIsInstance(author_distribution, dict)

        # Check distribution calculations
        self.assertEqual(author_distribution.get('John Smith', 0), 2)
        self.assertEqual(author_distribution.get('Jane Doe', 0), 1)

    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        stats = self.visualizer.generate_summary_statistics(self.documents)

        # Check that the function returns a dictionary
        self.assertIsInstance(stats, dict)

        # Check statistics calculations
        self.assertIn('word_count', stats)
        if 'word_count' in stats:
            self.assertEqual(stats['word_count'].get('min', 0), 150)
            self.assertEqual(stats['word_count'].get('max', 0), 200)
            self.assertAlmostEqual(stats['word_count'].get('mean', 0), 176.67, places=2)

    def test_identify_metadata_gaps(self):
        """Test metadata gap identification."""
        gaps = self.visualizer.identify_metadata_gaps(self.documents)

        # Check that the function returns a dictionary
        self.assertIsInstance(gaps, dict)

        # Check gap identification
        self.assertIn('date', gaps)
        self.assertIn('quality_score', gaps)
        if 'date' in gaps:
            self.assertEqual(len(gaps['date']), 1)
            self.assertEqual(gaps['date'][0], 'doc3')


class TestDocumentQualityScorer(unittest.TestCase):
    """Test the DocumentQualityScorer implementation."""

    def setUp(self):
        self.scorer = DocumentQualityScorer()

        # Sample document with metadata
        self.document = {
            'id': 'doc1',
            'content': 'This is a sample document with reasonable length and structure.',
            'metadata': {
                'author': 'John Smith',
                'date': '2023-01-15',
                'topics': ['AI', 'Machine Learning'],
                'type': 'article',
                'word_count': 150,
                'source': 'Reputable Journal'
            }
        }

        # Sample document with incomplete metadata
        self.incomplete_document = {
            'id': 'doc2',
            'content': 'Short text.',
            'metadata': {
                'type': 'article'
            }
        }

        # Required metadata fields
        self.required_fields = ['author', 'date', 'type', 'topics']

        # Source credibility ratings
        self.credibility_ratings = {
            'Reputable Journal': 0.9,
            'Unknown Source': 0.5
        }

    def test_score_metadata_completeness(self):
        """Test metadata completeness scoring."""
        # Complete document
        score1 = self.scorer.score_metadata_completeness(
            self.document,
            self.required_fields
        )

        # Incomplete document
        score2 = self.scorer.score_metadata_completeness(
            self.incomplete_document,
            self.required_fields
        )

        # Check that scores are between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 1)

        # Complete document should have higher score
        self.assertGreater(score1, score2)

        # Complete document should have perfect score
        self.assertEqual(score1, 1.0)

        # Incomplete document should have 1/4 score (only 'type' is present)
        self.assertEqual(score2, 0.25)

    def test_score_content_quality(self):
        """Test content quality scoring."""
        # Document with reasonable content
        score1 = self.scorer.score_content_quality(self.document)

        # Document with minimal content
        score2 = self.scorer.score_content_quality(self.incomplete_document)

        # Check that scores are between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 1)

        # Better content should have higher score
        self.assertGreater(score1, score2)

    def test_score_source_credibility(self):
        """Test source credibility scoring."""
        # Document with credible source
        score1 = self.scorer.score_source_credibility(
            self.document,
            self.credibility_ratings
        )

        # Document with unknown source
        score2 = self.scorer.score_source_credibility(
            self.incomplete_document,
            self.credibility_ratings
        )

        # Check that scores are between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 1)

        # Credible source should have higher score
        self.assertGreater(score1, score2)

    def test_score_freshness(self):
        """Test freshness scoring."""
        # Recent document
        recent_doc = self.document.copy()
        recent_doc['metadata'] = self.document['metadata'].copy()
        recent_doc['metadata']['date'] = datetime.now().strftime('%Y-%m-%d')

        # Older document
        older_doc = self.document.copy()
        older_doc['metadata'] = self.document['metadata'].copy()
        older_doc['metadata']['date'] = '2020-01-01'

        score1 = self.scorer.score_freshness(recent_doc)
        score2 = self.scorer.score_freshness(older_doc)

        # Check that scores are between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 1)

        # Recent document should have higher freshness score
        self.assertGreater(score1, score2)

    def test_calculate_overall_quality_score(self):
        """Test overall quality scoring."""
        # Complete, high-quality document
        score1 = self.scorer.calculate_overall_quality_score(
            self.document,
            weights={
                'completeness': 0.3,
                'content_quality': 0.3,
                'credibility': 0.2,
                'freshness': 0.2
            }
        )

        # Incomplete, low-quality document
        score2 = self.scorer.calculate_overall_quality_score(
            self.incomplete_document,
            weights={
                'completeness': 0.3,
                'content_quality': 0.3,
                'credibility': 0.2,
                'freshness': 0.2
            }
        )

        # Check that scores are between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 1)

        # Better document should have higher overall score
        self.assertGreater(score1, score2)


@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic not available")
class TestEducationalContentMetadata(unittest.TestCase):
    """Test the EducationalContentMetadata schema."""

    def test_valid_metadata(self):
        """Test valid educational content metadata."""
        try:
            metadata = EducationalContentMetadata(
                content_id="lesson-101",
                title="Introduction to Python",
                content_type="lesson",
                difficulty_level="beginner",
                subject_area="Computer Science",  # Added required subject_area field
                learning_objectives=["Understand basic Python syntax", "Write simple programs"],
                prerequisites=["Basic programming concepts"],
                target_audience=["High school students", "College freshmen"],
                time_required=45,  # minutes
                standards=["CS1.1", "CS1.2"]
            )

            # Check that the schema validates correctly
            self.assertEqual(metadata.content_id, "lesson-101")
            self.assertEqual(metadata.title, "Introduction to Python")
            self.assertEqual(metadata.content_type, "lesson")
            self.assertEqual(metadata.difficulty_level, "beginner")
        except Exception as e:
            self.fail(f"Valid metadata raised an exception: {e}")

    def test_invalid_metadata(self):
        """Test invalid educational content metadata."""
        # This should raise a validation error
        with self.assertRaises(Exception):
            EducationalContentMetadata(
                content_id="lesson-101",
                title="Introduction to Python",
                content_type="invalid-type",  # Invalid content type
                difficulty_level="super-hard",  # Invalid difficulty level
                subject_area="Computer Science",  # Added required subject_area field
                learning_objectives=["Understand basic Python syntax"],
                prerequisites=[],
                target_audience=["High school students"],
                time_required=-10,  # Invalid negative time
                standards=["CS1.1"]
            )


if __name__ == '__main__':
    unittest.main()
