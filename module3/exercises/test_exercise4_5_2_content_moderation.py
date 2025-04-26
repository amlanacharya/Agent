"""
Test file for Exercise 4.5.2: Content Moderation System
"""

import unittest
from exercise4_5_2_content_moderation import (
    ContentCategory, SeverityLevel, ContentFlag, BiasType, BiasFlag,
    FactualErrorType, FactualErrorFlag, ContentModerationResult,
    ContentModerationSystem
)


class TestContentModerationSystem(unittest.TestCase):
    """Test cases for content moderation system."""

    def setUp(self):
        """Set up test fixtures."""
        self.moderation_system = ContentModerationSystem()

    def test_clean_content(self):
        """Test moderation of clean content."""
        clean_text = "The weather in New York is currently sunny with a temperature of 75°F."
        result = self.moderation_system.moderate_content(clean_text)
        
        self.assertTrue(result.is_approved)
        self.assertEqual(result.total_flags, 0)
        self.assertEqual(result.overall_severity, SeverityLevel.NONE)
        self.assertFalse(result.has_flags)
        self.assertEqual(len(result.content_flags), 0)
        self.assertEqual(len(result.bias_flags), 0)
        self.assertEqual(len(result.factual_error_flags), 0)

    def test_profanity_detection(self):
        """Test detection of profanity."""
        # Use a profanity word from the default list
        profanity_text = "The weather is profanity1 hot today."
        result = self.moderation_system.moderate_content(profanity_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to medium severity
        self.assertEqual(result.total_flags, 1)
        self.assertEqual(result.overall_severity, SeverityLevel.MEDIUM)
        self.assertTrue(result.has_flags)
        self.assertEqual(len(result.content_flags), 1)
        
        # Check the flag details
        flag = result.content_flags[0]
        self.assertEqual(flag.category, ContentCategory.PROFANITY)
        self.assertEqual(flag.severity, SeverityLevel.MEDIUM)
        self.assertEqual(flag.snippet, "profanity1")

    def test_hate_speech_detection(self):
        """Test detection of hate speech."""
        hate_speech_text = "I hate people based on their religion."
        result = self.moderation_system.moderate_content(hate_speech_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to high severity
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.content_flags), 1)
        
        # Check that at least one flag is for hate speech
        hate_speech_flags = [flag for flag in result.content_flags if flag.category == ContentCategory.HATE_SPEECH]
        self.assertGreaterEqual(len(hate_speech_flags), 1)
        
        # Check the flag details
        flag = hate_speech_flags[0]
        self.assertEqual(flag.severity, SeverityLevel.HIGH)

    def test_violence_detection(self):
        """Test detection of violent content."""
        violence_text = "I want to kill people who disagree with me."
        result = self.moderation_system.moderate_content(violence_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to high severity
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.content_flags), 1)
        
        # Check that at least one flag is for violence
        violence_flags = [flag for flag in result.content_flags if flag.category == ContentCategory.VIOLENCE]
        self.assertGreaterEqual(len(violence_flags), 1)
        
        # Check the flag details
        flag = violence_flags[0]
        self.assertEqual(flag.severity, SeverityLevel.HIGH)

    def test_personal_info_detection(self):
        """Test detection of personal information."""
        personal_info_text = "My social security number is 123-45-6789."
        result = self.moderation_system.moderate_content(personal_info_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to critical severity
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.content_flags), 1)
        
        # Check that at least one flag is for personal info
        personal_info_flags = [flag for flag in result.content_flags if flag.category == ContentCategory.PERSONAL_INFO]
        self.assertGreaterEqual(len(personal_info_flags), 1)
        
        # Check the flag details
        flag = personal_info_flags[0]
        self.assertEqual(flag.severity, SeverityLevel.CRITICAL)
        self.assertIn("123-45-6789", flag.snippet)

    def test_bias_detection(self):
        """Test detection of bias."""
        bias_text = "All women can't understand complex technical concepts."
        result = self.moderation_system.moderate_content(bias_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to medium severity
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.bias_flags), 1)
        
        # Check the flag details
        flag = result.bias_flags[0]
        self.assertEqual(flag.bias_type, BiasType.GENDER)
        self.assertEqual(flag.severity, SeverityLevel.MEDIUM)

    def test_factual_error_detection(self):
        """Test detection of factual errors."""
        factual_error_text = "The earth is flat and climate change is a hoax."
        result = self.moderation_system.moderate_content(factual_error_text)
        
        self.assertFalse(result.is_approved)  # Should be rejected due to high severity
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.factual_error_flags), 2)  # Should detect both errors
        
        # Check that the flags are for incorrect facts
        incorrect_fact_flags = [
            flag for flag in result.factual_error_flags 
            if flag.error_type == FactualErrorType.INCORRECT_FACT
        ]
        self.assertGreaterEqual(len(incorrect_fact_flags), 2)
        
        # Check the flag details
        earth_flat_flags = [flag for flag in incorrect_fact_flags if "earth is flat" in flag.snippet.lower()]
        climate_hoax_flags = [flag for flag in incorrect_fact_flags if "climate change is a hoax" in flag.snippet.lower()]
        
        self.assertGreaterEqual(len(earth_flat_flags), 1)
        self.assertGreaterEqual(len(climate_hoax_flags), 1)
        
        # Check that corrections are provided
        self.assertIsNotNone(earth_flat_flags[0].correction)
        self.assertIsNotNone(climate_hoax_flags[0].correction)

    def test_unverified_claim_detection(self):
        """Test detection of unverified claims."""
        unverified_claim_text = "Studies show that drinking water cures all diseases. Everyone knows this is true."
        result = self.moderation_system.moderate_content(unverified_claim_text)
        
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(len(result.factual_error_flags), 1)
        
        # Check that at least one flag is for unverified claims
        unverified_claim_flags = [
            flag for flag in result.factual_error_flags 
            if flag.error_type == FactualErrorType.UNVERIFIED_CLAIM
        ]
        self.assertGreaterEqual(len(unverified_claim_flags), 1)

    def test_multiple_issues(self):
        """Test content with multiple issues."""
        multiple_issues_text = "The earth is flat. All women are bad at math. I want to kill people who disagree."
        result = self.moderation_system.moderate_content(multiple_issues_text)
        
        self.assertFalse(result.is_approved)
        self.assertTrue(result.has_flags)
        self.assertGreaterEqual(result.total_flags, 3)  # Should have at least 3 flags
        
        # Check that we have different types of flags
        self.assertGreaterEqual(len(result.content_flags), 1)  # Violence
        self.assertGreaterEqual(len(result.bias_flags), 1)  # Gender bias
        self.assertGreaterEqual(len(result.factual_error_flags), 1)  # Earth is flat
        
        # Check overall severity (should be the highest of all flags)
        self.assertEqual(result.overall_severity, SeverityLevel.HIGH)

    def test_content_moderation_result_methods(self):
        """Test ContentModerationResult methods."""
        # Create a result with various flags
        result = ContentModerationResult(
            text="Test text",
            content_flags=[
                ContentFlag(
                    category=ContentCategory.PROFANITY,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.9,
                    snippet="profanity1"
                ),
                ContentFlag(
                    category=ContentCategory.VIOLENCE,
                    severity=SeverityLevel.HIGH,
                    confidence=0.8,
                    snippet="kill people"
                )
            ],
            bias_flags=[
                BiasFlag(
                    bias_type=BiasType.GENDER,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.7,
                    snippet="women can't understand"
                )
            ],
            factual_error_flags=[
                FactualErrorFlag(
                    error_type=FactualErrorType.INCORRECT_FACT,
                    severity=SeverityLevel.HIGH,
                    confidence=0.9,
                    snippet="earth is flat",
                    correction="The Earth is an oblate spheroid."
                )
            ],
            is_approved=False,
            rejection_reason="Content contains high severity issues"
        )
        
        # Test get_flags_by_severity
        high_severity_flags = result.get_flags_by_severity(SeverityLevel.HIGH)
        self.assertEqual(len(high_severity_flags), 2)  # Violence and incorrect fact
        
        medium_severity_flags = result.get_flags_by_severity(SeverityLevel.MEDIUM)
        self.assertEqual(len(medium_severity_flags), 2)  # Profanity and gender bias
        
        # Test get_content_flags_by_category
        profanity_flags = result.get_content_flags_by_category(ContentCategory.PROFANITY)
        self.assertEqual(len(profanity_flags), 1)
        
        violence_flags = result.get_content_flags_by_category(ContentCategory.VIOLENCE)
        self.assertEqual(len(violence_flags), 1)
        
        # Test get_bias_flags_by_type
        gender_bias_flags = result.get_bias_flags_by_type(BiasType.GENDER)
        self.assertEqual(len(gender_bias_flags), 1)
        
        # Test get_factual_error_flags_by_type
        incorrect_fact_flags = result.get_factual_error_flags_by_type(FactualErrorType.INCORRECT_FACT)
        self.assertEqual(len(incorrect_fact_flags), 1)
        
        # Test categories_flagged
        self.assertEqual(result.categories_flagged, {ContentCategory.PROFANITY, ContentCategory.VIOLENCE})
        
        # Test bias_types_flagged
        self.assertEqual(result.bias_types_flagged, {BiasType.GENDER})
        
        # Test error_types_flagged
        self.assertEqual(result.error_types_flagged, {FactualErrorType.INCORRECT_FACT})
        
        # Test get_summary
        summary = result.get_summary()
        self.assertEqual(summary["total_flags"], 4)
        self.assertEqual(summary["content_flags"], 2)
        self.assertEqual(summary["bias_flags"], 1)
        self.assertEqual(summary["factual_error_flags"], 1)
        self.assertEqual(summary["overall_severity"], SeverityLevel.HIGH)
        self.assertFalse(summary["is_approved"])


if __name__ == "__main__":
    unittest.main()
