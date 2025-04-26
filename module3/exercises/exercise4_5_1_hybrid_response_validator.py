"""
Exercise 4.5.1: Hybrid Response Validator

This exercise implements a Pydantic model for validating agent responses that include
both text and structured data components.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union, Literal, Tuple
from enum import Enum
import re
from datetime import datetime


class ResponseType(str, Enum):
    """Types of agent responses."""
    TEXT_ONLY = "text_only"
    STRUCTURED_ONLY = "structured_only"
    HYBRID = "hybrid"


class TextQuality(BaseModel):
    """Model for text quality metrics."""
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="Clarity score (0-1)")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness score (0-1)")
    tone_appropriate: bool = Field(..., description="Whether the tone is appropriate")
    contains_errors: bool = Field(..., description="Whether the text contains errors")
    error_details: Optional[List[str]] = Field(None, description="Details of any errors")

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        base_score = (self.clarity_score + self.relevance_score + self.completeness_score) / 3

        # Penalize for errors and inappropriate tone
        if self.contains_errors:
            base_score *= 0.8

        if not self.tone_appropriate:
            base_score *= 0.9

        return base_score


class StructuredDataQuality(BaseModel):
    """Model for structured data quality metrics."""
    schema_valid: bool = Field(..., description="Whether the data conforms to the expected schema")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness score (0-1)")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Accuracy score (0-1)")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Internal consistency score (0-1)")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        if not self.schema_valid:
            return 0.0

        return (self.completeness_score + self.accuracy_score + self.consistency_score) / 3


class HybridResponseValidator(BaseModel):
    """
    Validator for agent responses that include both text and structured data.

    This validator can handle text-only responses, structured-only responses,
    and hybrid responses that contain both text and structured data.
    """
    response_type: ResponseType = Field(..., description="Type of response")
    text_content: Optional[str] = Field(None, description="Text content of the response")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured data content")
    expected_schema: Optional[Dict[str, Any]] = Field(None, description="Expected schema for structured data")
    text_quality: Optional[TextQuality] = Field(None, description="Quality metrics for text content")
    data_quality: Optional[StructuredDataQuality] = Field(None, description="Quality metrics for structured data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")

    @model_validator(mode='after')
    def validate_response_components(self):
        """Validate that the response components match the response type."""
        if self.response_type == ResponseType.TEXT_ONLY:
            if not self.text_content:
                raise ValueError("Text content is required for text-only responses")
            if self.structured_data:
                raise ValueError("Structured data should not be present for text-only responses")

        elif self.response_type == ResponseType.STRUCTURED_ONLY:
            if not self.structured_data:
                raise ValueError("Structured data is required for structured-only responses")
            if self.text_content:
                raise ValueError("Text content should not be present for structured-only responses")

        elif self.response_type == ResponseType.HYBRID:
            if not self.text_content:
                raise ValueError("Text content is required for hybrid responses")
            if not self.structured_data:
                raise ValueError("Structured data is required for hybrid responses")

        return self

    @model_validator(mode='after')
    def validate_quality_metrics(self):
        """Validate that the quality metrics match the response type."""
        if (self.response_type == ResponseType.TEXT_ONLY or self.response_type == ResponseType.HYBRID) and not self.text_quality:
            raise ValueError("Text quality metrics are required for responses with text content")

        if (self.response_type == ResponseType.STRUCTURED_ONLY or self.response_type == ResponseType.HYBRID) and not self.data_quality:
            raise ValueError("Data quality metrics are required for responses with structured data")

        return self

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score for the response."""
        if self.response_type == ResponseType.TEXT_ONLY:
            return self.text_quality.overall_score

        elif self.response_type == ResponseType.STRUCTURED_ONLY:
            return self.data_quality.overall_score

        elif self.response_type == ResponseType.HYBRID:
            # Weight text and data quality equally for hybrid responses
            return (self.text_quality.overall_score + self.data_quality.overall_score) / 2

        return 0.0

    @classmethod
    def validate_text(cls, text: str) -> TextQuality:
        """
        Validate text content and generate quality metrics.

        Args:
            text: Text content to validate

        Returns:
            TextQuality instance with quality metrics
        """
        # Check for clarity (simple heuristics)
        avg_sentence_length = cls._calculate_avg_sentence_length(text)
        clarity_score = 1.0
        if avg_sentence_length > 25:
            # Penalize very long sentences
            clarity_score -= min(0.5, (avg_sentence_length - 25) * 0.02)

        # Check for relevance (placeholder - in a real system this would use NLP)
        relevance_score = 0.9  # Placeholder

        # Check for completeness (placeholder)
        completeness_score = 0.9  # Placeholder

        # Check for appropriate tone
        tone_appropriate = not cls._contains_inappropriate_tone(text)

        # Check for errors
        errors = cls._find_text_errors(text)
        contains_errors = len(errors) > 0

        return TextQuality(
            clarity_score=clarity_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            tone_appropriate=tone_appropriate,
            contains_errors=contains_errors,
            error_details=errors if contains_errors else None
        )

    @classmethod
    def validate_structured_data(cls, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> StructuredDataQuality:
        """
        Validate structured data and generate quality metrics.

        Args:
            data: Structured data to validate
            schema: Expected schema for the data (optional)

        Returns:
            StructuredDataQuality instance with quality metrics
        """
        validation_errors = []
        schema_valid = True

        # Validate against schema if provided
        if schema:
            schema_valid, schema_errors = cls._validate_against_schema(data, schema)
            validation_errors.extend(schema_errors)

        # Check completeness
        completeness_score = cls._calculate_completeness(data, schema)

        # Check accuracy (placeholder - in a real system this would use domain knowledge)
        accuracy_score = 0.9  # Placeholder

        # Check internal consistency
        consistency_score, consistency_errors = cls._check_consistency(data)
        validation_errors.extend(consistency_errors)

        return StructuredDataQuality(
            schema_valid=schema_valid,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            validation_errors=validation_errors
        )

    @classmethod
    def validate_hybrid_response(cls, text: str, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> 'HybridResponseValidator':
        """
        Validate a hybrid response with both text and structured data.

        Args:
            text: Text content of the response
            data: Structured data content
            schema: Expected schema for the data (optional)

        Returns:
            HybridResponseValidator instance with validation results
        """
        text_quality = cls.validate_text(text)
        data_quality = cls.validate_structured_data(data, schema)

        return cls(
            response_type=ResponseType.HYBRID,
            text_content=text,
            structured_data=data,
            expected_schema=schema,
            text_quality=text_quality,
            data_quality=data_quality
        )

    @classmethod
    def validate_text_response(cls, text: str) -> 'HybridResponseValidator':
        """
        Validate a text-only response.

        Args:
            text: Text content of the response

        Returns:
            HybridResponseValidator instance with validation results
        """
        text_quality = cls.validate_text(text)

        return cls(
            response_type=ResponseType.TEXT_ONLY,
            text_content=text,
            structured_data=None,
            expected_schema=None,
            text_quality=text_quality,
            data_quality=None
        )

    @classmethod
    def validate_structured_response(cls, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> 'HybridResponseValidator':
        """
        Validate a structured-only response.

        Args:
            data: Structured data content
            schema: Expected schema for the data (optional)

        Returns:
            HybridResponseValidator instance with validation results
        """
        data_quality = cls.validate_structured_data(data, schema)

        return cls(
            response_type=ResponseType.STRUCTURED_ONLY,
            text_content=None,
            structured_data=data,
            expected_schema=schema,
            text_quality=None,
            data_quality=data_quality
        )

    @staticmethod
    def _calculate_avg_sentence_length(text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0

        total_words = sum(len(re.findall(r'\b\w+\b', s)) for s in sentences)
        return total_words / len(sentences)

    @staticmethod
    def _contains_inappropriate_tone(text: str) -> bool:
        """Check if text contains inappropriate tone."""
        inappropriate_patterns = [
            r'\b(stupid|idiot|dumb|fool)\b',
            r'\b(hate|despise)\b',
            r'\b(angry|furious|mad)\b',
            r'\b(damn|hell|crap)\b'
        ]

        for pattern in inappropriate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _find_text_errors(text: str) -> List[str]:
        """Find errors in text content."""
        errors = []

        # Check for repeated words
        repeated_words = re.findall(r'\b(\w+)\s+\1\b', text, re.IGNORECASE)
        if repeated_words:
            errors.append(f"Repeated words: {', '.join(set(repeated_words))}")

        # Check for very long sentences (potential readability issues)
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                word_count = len(re.findall(r'\b\w+\b', sentence))
                if word_count > 40:
                    errors.append(f"Very long sentence (#{i+1}, {word_count} words)")

        # Check for common spelling errors (simplified example)
        common_errors = {
            r'\b(teh)\b': 'the',
            r'\b(thier)\b': 'their',
            r'\b(recieve)\b': 'receive',
            r'\b(seperate)\b': 'separate'
        }

        for error_pattern, correction in common_errors.items():
            if re.search(error_pattern, text, re.IGNORECASE):
                errors.append(f"Possible spelling error: '{re.search(error_pattern, text, re.IGNORECASE).group(0)}' (did you mean '{correction}'?)")

        return errors

    @staticmethod
    def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against a schema."""
        errors = []
        valid = True

        # Check required fields
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
                    valid = False

        # Check field types
        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in data:
                    field_type = field_schema.get("type")
                    if field_type:
                        if field_type == "string" and not isinstance(data[field], str):
                            errors.append(f"Field '{field}' should be a string")
                            valid = False
                        elif field_type == "number" and not isinstance(data[field], (int, float)):
                            errors.append(f"Field '{field}' should be a number")
                            valid = False
                        elif field_type == "integer" and not isinstance(data[field], int):
                            errors.append(f"Field '{field}' should be an integer")
                            valid = False
                        elif field_type == "boolean" and not isinstance(data[field], bool):
                            errors.append(f"Field '{field}' should be a boolean")
                            valid = False
                        elif field_type == "array" and not isinstance(data[field], list):
                            errors.append(f"Field '{field}' should be an array")
                            valid = False
                        elif field_type == "object" and not isinstance(data[field], dict):
                            errors.append(f"Field '{field}' should be an object")
                            valid = False

        return valid, errors

    @staticmethod
    def _calculate_completeness(data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> float:
        """Calculate completeness score for structured data."""
        if not schema:
            return 1.0  # No schema to check against

        required_fields = schema.get("required", [])
        if not required_fields:
            return 1.0  # No required fields

        present_fields = [field for field in required_fields if field in data]
        return len(present_fields) / len(required_fields)

    @staticmethod
    def _check_consistency(data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check internal consistency of structured data."""
        errors = []

        # Check for date consistency
        date_fields = {}
        for field, value in data.items():
            if field.endswith("_date") and isinstance(value, str):
                try:
                    date_fields[field] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    pass

        # Check that start dates are before end dates
        for start_field in date_fields:
            if start_field.startswith("start_"):
                end_field = "end_" + start_field[6:]
                if end_field in date_fields and date_fields[start_field] > date_fields[end_field]:
                    errors.append(f"Inconsistent dates: {start_field} is after {end_field}")

        # Check for numeric consistency
        if "min_value" in data and "max_value" in data:
            if isinstance(data["min_value"], (int, float)) and isinstance(data["max_value"], (int, float)):
                if data["min_value"] > data["max_value"]:
                    errors.append("Inconsistent values: min_value is greater than max_value")

        # Calculate consistency score
        if errors:
            return 1.0 - (len(errors) * 0.2), errors  # Deduct 0.2 for each consistency error

        return 1.0, errors


# Example usage
if __name__ == "__main__":
    # Example 1: Validate a text-only response
    text_response = "The weather in New York is currently sunny with a temperature of 75°F. There is a 10% chance of rain later today."
    text_validator = HybridResponseValidator.validate_text_response(text_response)

    print("Example 1: Text-only response validation")
    print(f"Response type: {text_validator.response_type}")
    print(f"Text quality - Clarity: {text_validator.text_quality.clarity_score:.2f}")
    print(f"Text quality - Relevance: {text_validator.text_quality.relevance_score:.2f}")
    print(f"Text quality - Completeness: {text_validator.text_quality.completeness_score:.2f}")
    print(f"Text quality - Appropriate tone: {text_validator.text_quality.tone_appropriate}")
    print(f"Text quality - Contains errors: {text_validator.text_quality.contains_errors}")
    if text_validator.text_quality.contains_errors:
        print(f"Text quality - Error details: {text_validator.text_quality.error_details}")
    print(f"Overall quality score: {text_validator.overall_quality_score:.2f}")
    print()

    # Example 2: Validate a structured-only response
    structured_response = {
        "location": "New York",
        "temperature": 75,
        "condition": "sunny",
        "precipitation_chance": 0.1,
        "forecast_date": "2023-12-25T12:00:00"
    }

    schema = {
        "type": "object",
        "required": ["location", "temperature", "condition"],
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "condition": {"type": "string"},
            "precipitation_chance": {"type": "number"},
            "forecast_date": {"type": "string"}
        }
    }

    data_validator = HybridResponseValidator.validate_structured_response(structured_response, schema)

    print("Example 2: Structured-only response validation")
    print(f"Response type: {data_validator.response_type}")
    print(f"Data quality - Schema valid: {data_validator.data_quality.schema_valid}")
    print(f"Data quality - Completeness: {data_validator.data_quality.completeness_score:.2f}")
    print(f"Data quality - Accuracy: {data_validator.data_quality.accuracy_score:.2f}")
    print(f"Data quality - Consistency: {data_validator.data_quality.consistency_score:.2f}")
    if data_validator.data_quality.validation_errors:
        print(f"Data quality - Validation errors: {data_validator.data_quality.validation_errors}")
    print(f"Overall quality score: {data_validator.overall_quality_score:.2f}")
    print()

    # Example 3: Validate a hybrid response
    hybrid_text = "The weather in New York is currently sunny with a temperature of 75°F."
    hybrid_data = {
        "location": "New York",
        "temperature": 75,
        "condition": "sunny",
        "precipitation_chance": 0.1
    }

    hybrid_validator = HybridResponseValidator.validate_hybrid_response(hybrid_text, hybrid_data, schema)

    print("Example 3: Hybrid response validation")
    print(f"Response type: {hybrid_validator.response_type}")
    print(f"Text quality - Overall: {hybrid_validator.text_quality.overall_score:.2f}")
    print(f"Data quality - Overall: {hybrid_validator.data_quality.overall_score:.2f}")
    print(f"Overall quality score: {hybrid_validator.overall_quality_score:.2f}")
