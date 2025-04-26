"""
Test suite for Exercise 4.5.5: Quality Validator

This module contains tests for the QualityValidator class that evaluates
response quality across multiple dimensions.
"""


import unittest
from typing import Dict, Any
import sys
import os

# Import the QualityValidator module
# Assuming the module is in the same directory as this test file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exercise4_5_5_quality_validator import (
    QualityValidator, QualityDimension, QualityLevel, 
    QualityMetrics, QualityIssue
)


class TestQualityValidator(unittest.TestCase):
    """Test cases for the QualityValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = QualityValidator()
        
        # Sample texts for testing
        self.excellent_text = """
        # Understanding Pydantic Validation
        
        Pydantic is a data validation library that uses Python type annotations to validate data.
        It's both powerful and easy to use, making it perfect for many applications.
        
        ## How It Works
        
        When you define a Pydantic model, you specify the expected types and constraints for your data.
        For example:
        
        ```python
        from pydantic import BaseModel, Field
        
        class User(BaseModel):
            name: str = Field(..., min_length=1)
            age: int = Field(..., ge=0)
        ```
        
        This ensures your User objects always have:
        - A name with at least one character
        - A non-negative age value
        
        ## Benefits
        
        Using Pydantic gives you several advantages:
        - Automatic validation of input data
        - Clear error messages when validation fails
        - IDE support for type hinting
        - JSON schema generation
        
        ## Error Handling
        
        When validation fails, Pydantic raises a ValidationError that contains detailed information about what went wrong.
        You can catch these errors and handle them appropriately in your application.
        
        Have you tried using Pydantic in your projects? If not, it's worth considering for your data validation needs!
        """
        
        self.poor_text = """
        Pydantic validation. It's a library used for validation. The library can be utilized to validate data in applications. The validation happens automatically based on type annotations that you define in your models. The models inherit from BaseModel. The BaseModel class is provided by pydantic. You need to import it. You can specify fields in models. Fields have types. The types determine validation rules. If validation fails, errors are raised. You should handle errors properly. Error handling is important. Users should see helpful error messages. Messages help them fix issues. Documentation is available online. The documentation explains how to use pydantic. You should read it. Type hints are used in pydantic. Type hints are a feature of Python. Python introduced type hints in version 3.5. Type hints improve code. They make code more readable. They help with validation in pydantic.
        """
    
    def test_excellent_text_evaluation(self):
        """Test evaluation of high-quality text."""
        metrics = self.validator.validate(self.excellent_text)
        
        # Check overall quality level
        self.assertIn(metrics.overall_quality_level, [QualityLevel.EXCELLENT, QualityLevel.GOOD])
        
        # Check individual dimension scores
        self.assertGreaterEqual(metrics.clarity_score, 0.8)
        self.assertGreaterEqual(metrics.coherence_score, 0.8)
        self.assertGreaterEqual(metrics.engagement_score, 0.7)
        
        # Excellent text should have few or no severe issues
        severe_issues = [issue for issue in metrics.issues if issue.severity > 0.7]
        self.assertLessEqual(len(severe_issues), 1)
    
    def test_poor_text_evaluation(self):
        """Test evaluation of low-quality text."""
        metrics = self.validator.validate(self.poor_text)
        
        # Check overall quality level
        self.assertIn(metrics.overall_quality_level, [QualityLevel.POOR, QualityLevel.ADEQUATE])
        
        # Check individual dimension scores
        self.assertLessEqual(metrics.coherence_score, 0.7)
        self.assertLessEqual(metrics.conciseness_score, 0.7)
        
        # Poor text should have multiple issues
        self.assertGreaterEqual(len(metrics.issues), 3)
    
    def test_dimension_specific_evaluation(self):
        """Test evaluation of specific quality dimensions."""
        # Text with poor clarity but good organization
        complex_text = """
        The aforementioned implementation of algorithmic methodologies necessitates the utilization
        of substantial computational resources due to the intrinsic complexity of the mathematical
        operations involved in the calculation of results. Furthermore, it is imperative to 
        acknowledge that the aforementioned complexity imposes significant constraints on the 
        scalability of said implementation in contexts characterized by limited availability of the
        aforementioned computational resources.
        
        # First Point
        
        - Item one
        - Item two
        
        # Second Point
        
        - Another item
        - Final item
        """
        
        metrics = self.validator.validate(complex_text)
        
        # Should score low on clarity
        self.assertLessEqual(metrics.clarity_score, 0.7)
        
        # But might score better on structure/coherence
        self.assertGreaterEqual(metrics.coherence_score, 0.6)
        
        # Should have clarity issues
        clarity_issues = metrics.get_issues_by_dimension(QualityDimension.CLARITY)
        self.assertGreaterEqual(len(clarity_issues), 1)
    
    def test_improvement_suggestions(self):
        """Test that improvement suggestions are generated correctly."""
        metrics = self.validator.validate(self.poor_text)
        suggestions = self.validator.get_improvement_suggestions(metrics)
        
        # Should provide some suggestions
        self.assertGreaterEqual(len(suggestions), 1)
        
        # Suggestions should be strings
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 10)  # Should be meaningful suggestions
    
    def test_context_aware_validation(self):
        """Test that the validator considers context when available."""
        text = """
        Pydantic is a Python library used for data validation.
        It's quite useful for many applications.
        """
        
        # Context with expected topics
        context = {
            "query": "How does Pydantic validation work with error handling?",
            "expected_topics": ["validation", "error handling", "type hints"]
        }
        
        metrics = self.validator.validate(text, context)
        
        # Should identify missing expected topics
        helpfulness_issues = metrics.get_issues_by_dimension(QualityDimension.HELPFULNESS)
        has_topic_issue = any("topics" in issue.description.lower() for issue in helpfulness_issues)
        self.assertTrue(has_topic_issue)


if __name__ == "__main__":
    unittest.main()