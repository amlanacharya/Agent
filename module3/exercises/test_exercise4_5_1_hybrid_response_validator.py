"""
Test file for Exercise 4.5.1: Hybrid Response Validator
"""

import unittest
from exercise4_5_1_hybrid_response_validator import (
    ResponseType, TextQuality, StructuredDataQuality, HybridResponseValidator
)


class TestHybridResponseValidator(unittest.TestCase):
    """Test cases for hybrid response validator."""

    def test_text_only_validation(self):
        """Test validation of a text-only response."""
        text = "The weather in New York is currently sunny with a temperature of 75°F."
        validator = HybridResponseValidator.validate_text_response(text)
        
        # Check response type
        self.assertEqual(validator.response_type, ResponseType.TEXT_ONLY)
        
        # Check that text content is present
        self.assertEqual(validator.text_content, text)
        
        # Check that structured data is not present
        self.assertIsNone(validator.structured_data)
        
        # Check that text quality metrics are present
        self.assertIsNotNone(validator.text_quality)
        self.assertIsInstance(validator.text_quality, TextQuality)
        
        # Check that data quality metrics are not present
        self.assertIsNone(validator.data_quality)
        
        # Check overall quality score
        self.assertGreaterEqual(validator.overall_quality_score, 0.0)
        self.assertLessEqual(validator.overall_quality_score, 1.0)

    def test_structured_only_validation(self):
        """Test validation of a structured-only response."""
        data = {
            "location": "New York",
            "temperature": 75,
            "condition": "sunny",
            "precipitation_chance": 0.1
        }
        
        schema = {
            "type": "object",
            "required": ["location", "temperature", "condition"],
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "precipitation_chance": {"type": "number"}
            }
        }
        
        validator = HybridResponseValidator.validate_structured_response(data, schema)
        
        # Check response type
        self.assertEqual(validator.response_type, ResponseType.STRUCTURED_ONLY)
        
        # Check that text content is not present
        self.assertIsNone(validator.text_content)
        
        # Check that structured data is present
        self.assertEqual(validator.structured_data, data)
        
        # Check that text quality metrics are not present
        self.assertIsNone(validator.text_quality)
        
        # Check that data quality metrics are present
        self.assertIsNotNone(validator.data_quality)
        self.assertIsInstance(validator.data_quality, StructuredDataQuality)
        
        # Check overall quality score
        self.assertGreaterEqual(validator.overall_quality_score, 0.0)
        self.assertLessEqual(validator.overall_quality_score, 1.0)

    def test_hybrid_validation(self):
        """Test validation of a hybrid response."""
        text = "The weather in New York is currently sunny with a temperature of 75°F."
        data = {
            "location": "New York",
            "temperature": 75,
            "condition": "sunny",
            "precipitation_chance": 0.1
        }
        
        schema = {
            "type": "object",
            "required": ["location", "temperature", "condition"],
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "precipitation_chance": {"type": "number"}
            }
        }
        
        validator = HybridResponseValidator.validate_hybrid_response(text, data, schema)
        
        # Check response type
        self.assertEqual(validator.response_type, ResponseType.HYBRID)
        
        # Check that text content is present
        self.assertEqual(validator.text_content, text)
        
        # Check that structured data is present
        self.assertEqual(validator.structured_data, data)
        
        # Check that text quality metrics are present
        self.assertIsNotNone(validator.text_quality)
        self.assertIsInstance(validator.text_quality, TextQuality)
        
        # Check that data quality metrics are present
        self.assertIsNotNone(validator.data_quality)
        self.assertIsInstance(validator.data_quality, StructuredDataQuality)
        
        # Check overall quality score
        self.assertGreaterEqual(validator.overall_quality_score, 0.0)
        self.assertLessEqual(validator.overall_quality_score, 1.0)

    def test_text_quality_metrics(self):
        """Test text quality metrics calculation."""
        # Test with good text
        good_text = "The weather in New York is currently sunny with a temperature of 75°F."
        good_quality = HybridResponseValidator.validate_text(good_text)
        
        self.assertGreaterEqual(good_quality.clarity_score, 0.0)
        self.assertLessEqual(good_quality.clarity_score, 1.0)
        self.assertGreaterEqual(good_quality.relevance_score, 0.0)
        self.assertLessEqual(good_quality.relevance_score, 1.0)
        self.assertGreaterEqual(good_quality.completeness_score, 0.0)
        self.assertLessEqual(good_quality.completeness_score, 1.0)
        self.assertIsInstance(good_quality.tone_appropriate, bool)
        self.assertIsInstance(good_quality.contains_errors, bool)
        
        # Test with text containing errors
        error_text = "The weather in New York is teh best today. It is teh sunniest day."
        error_quality = HybridResponseValidator.validate_text(error_text)
        
        self.assertTrue(error_quality.contains_errors)
        self.assertIsNotNone(error_quality.error_details)
        self.assertGreater(len(error_quality.error_details), 0)
        
        # Test with inappropriate tone
        tone_text = "The weather is damn hot today. It's stupid how hot it is."
        tone_quality = HybridResponseValidator.validate_text(tone_text)
        
        self.assertFalse(tone_quality.tone_appropriate)

    def test_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        # Test with valid data
        valid_data = {
            "location": "New York",
            "temperature": 75,
            "condition": "sunny",
            "precipitation_chance": 0.1
        }
        
        schema = {
            "type": "object",
            "required": ["location", "temperature", "condition"],
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "precipitation_chance": {"type": "number"}
            }
        }
        
        valid_quality = HybridResponseValidator.validate_structured_data(valid_data, schema)
        
        self.assertTrue(valid_quality.schema_valid)
        self.assertEqual(valid_quality.completeness_score, 1.0)  # All required fields present
        self.assertGreaterEqual(valid_quality.accuracy_score, 0.0)
        self.assertLessEqual(valid_quality.accuracy_score, 1.0)
        self.assertGreaterEqual(valid_quality.consistency_score, 0.0)
        self.assertLessEqual(valid_quality.consistency_score, 1.0)
        self.assertEqual(len(valid_quality.validation_errors), 0)
        
        # Test with invalid data (missing required field)
        invalid_data = {
            "location": "New York",
            "temperature": 75,
            # Missing "condition"
            "precipitation_chance": 0.1
        }
        
        invalid_quality = HybridResponseValidator.validate_structured_data(invalid_data, schema)
        
        self.assertFalse(invalid_quality.schema_valid)
        self.assertLess(invalid_quality.completeness_score, 1.0)  # Missing required field
        self.assertGreater(len(invalid_quality.validation_errors), 0)
        
        # Test with inconsistent data
        inconsistent_data = {
            "location": "New York",
            "temperature": 75,
            "condition": "sunny",
            "precipitation_chance": 0.1,
            "min_value": 80,
            "max_value": 70  # Inconsistent: min > max
        }
        
        inconsistent_quality = HybridResponseValidator.validate_structured_data(inconsistent_data, schema)
        
        self.assertLess(inconsistent_quality.consistency_score, 1.0)
        self.assertGreater(len(inconsistent_quality.validation_errors), 0)

    def test_validation_errors(self):
        """Test validation error handling."""
        # Test missing text content for text-only response
        with self.assertRaises(ValueError):
            HybridResponseValidator(
                response_type=ResponseType.TEXT_ONLY,
                text_content=None,
                structured_data=None,
                text_quality=TextQuality(
                    clarity_score=0.9,
                    relevance_score=0.9,
                    completeness_score=0.9,
                    tone_appropriate=True,
                    contains_errors=False
                ),
                data_quality=None
            )
        
        # Test missing structured data for structured-only response
        with self.assertRaises(ValueError):
            HybridResponseValidator(
                response_type=ResponseType.STRUCTURED_ONLY,
                text_content=None,
                structured_data=None,
                text_quality=None,
                data_quality=StructuredDataQuality(
                    schema_valid=True,
                    completeness_score=1.0,
                    accuracy_score=0.9,
                    consistency_score=1.0
                )
            )
        
        # Test missing text content for hybrid response
        with self.assertRaises(ValueError):
            HybridResponseValidator(
                response_type=ResponseType.HYBRID,
                text_content=None,
                structured_data={"location": "New York"},
                text_quality=TextQuality(
                    clarity_score=0.9,
                    relevance_score=0.9,
                    completeness_score=0.9,
                    tone_appropriate=True,
                    contains_errors=False
                ),
                data_quality=StructuredDataQuality(
                    schema_valid=True,
                    completeness_score=1.0,
                    accuracy_score=0.9,
                    consistency_score=1.0
                )
            )
        
        # Test missing structured data for hybrid response
        with self.assertRaises(ValueError):
            HybridResponseValidator(
                response_type=ResponseType.HYBRID,
                text_content="The weather is sunny.",
                structured_data=None,
                text_quality=TextQuality(
                    clarity_score=0.9,
                    relevance_score=0.9,
                    completeness_score=0.9,
                    tone_appropriate=True,
                    contains_errors=False
                ),
                data_quality=StructuredDataQuality(
                    schema_valid=True,
                    completeness_score=1.0,
                    accuracy_score=0.9,
                    consistency_score=1.0
                )
            )


if __name__ == "__main__":
    unittest.main()
