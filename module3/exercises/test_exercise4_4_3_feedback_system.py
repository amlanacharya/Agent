"""
Test file for Exercise 4.4.3: Feedback System for Validation Failures
"""

import unittest
from datetime import datetime, timedelta
from exercise4_4_3_feedback_system import (
    ValidationFailureType, ValidationFailure, ValidationFeedbackSystem
)


class TestFeedbackSystem(unittest.TestCase):
    """Test cases for validation feedback system."""

    def setUp(self):
        """Set up test fixtures."""
        self.feedback_system = ValidationFeedbackSystem()
        
        # Add some test failures
        self.feedback_system.record_failure(
            ValidationFailure(
                failure_id="test-001",
                failure_type=ValidationFailureType.MISSING_ENTITY,
                entity_type="date",
                input_text="Schedule a meeting tomorrow",
                details="Failed to extract date entity from 'tomorrow'",
                severity=3
            )
        )
        
        self.feedback_system.record_failure(
            ValidationFailure(
                failure_id="test-002",
                failure_type=ValidationFailureType.MISSING_ENTITY,
                entity_type="location",
                input_text="What's the weather like?",
                details="Missing required location entity for weather intent",
                severity=2
            )
        )
        
        self.feedback_system.record_failure(
            ValidationFailure(
                failure_id="test-003",
                failure_type=ValidationFailureType.AMBIGUOUS_INTENT,
                intent_type="booking",
                input_text="I need to make a reservation",
                details="Ambiguous between restaurant, hotel, and appointment booking",
                severity=4
            )
        )

    def test_record_failure(self):
        """Test recording a new failure."""
        # Record a new failure
        failure_id = self.feedback_system.record_failure(
            ValidationFailure(
                failure_id="test-004",
                failure_type=ValidationFailureType.INVALID_ENTITY,
                entity_type="email",
                input_text="Contact me at user@example",
                details="Invalid email format",
                severity=2
            )
        )
        
        # Check that the failure was recorded
        self.assertEqual(failure_id, "test-004")
        self.assertEqual(len(self.feedback_system.failures), 4)
        
        # Check that we can retrieve it
        failure = self.feedback_system.get_failure_by_id("test-004")
        self.assertIsNotNone(failure)
        self.assertEqual(failure.entity_type, "email")

    def test_mark_as_resolved(self):
        """Test marking a failure as resolved."""
        # Initially not resolved
        failure = self.feedback_system.get_failure_by_id("test-001")
        self.assertFalse(failure.resolved)
        
        # Mark as resolved
        result = self.feedback_system.mark_as_resolved("test-001")
        self.assertTrue(result)
        
        # Check that it's now resolved
        failure = self.feedback_system.get_failure_by_id("test-001")
        self.assertTrue(failure.resolved)
        
        # Try to mark a non-existent failure
        result = self.feedback_system.mark_as_resolved("non-existent")
        self.assertFalse(result)

    def test_get_failures_by_type(self):
        """Test getting failures by type."""
        # Get missing entity failures
        missing_entity_failures = self.feedback_system.get_failures_by_type(
            ValidationFailureType.MISSING_ENTITY
        )
        self.assertEqual(len(missing_entity_failures), 2)
        
        # Get ambiguous intent failures
        ambiguous_intent_failures = self.feedback_system.get_failures_by_type(
            ValidationFailureType.AMBIGUOUS_INTENT
        )
        self.assertEqual(len(ambiguous_intent_failures), 1)
        
        # Get a type with no failures
        no_failures = self.feedback_system.get_failures_by_type(
            ValidationFailureType.LOW_CONFIDENCE
        )
        self.assertEqual(len(no_failures), 0)

    def test_get_failures_by_entity(self):
        """Test getting failures by entity type."""
        # Get date entity failures
        date_failures = self.feedback_system.get_failures_by_entity("date")
        self.assertEqual(len(date_failures), 1)
        
        # Get location entity failures
        location_failures = self.feedback_system.get_failures_by_entity("location")
        self.assertEqual(len(location_failures), 1)
        
        # Get a type with no failures
        no_failures = self.feedback_system.get_failures_by_entity("time")
        self.assertEqual(len(no_failures), 0)

    def test_get_failures_by_intent(self):
        """Test getting failures by intent type."""
        # Get booking intent failures
        booking_failures = self.feedback_system.get_failures_by_intent("booking")
        self.assertEqual(len(booking_failures), 1)
        
        # Get a type with no failures
        no_failures = self.feedback_system.get_failures_by_intent("weather")
        self.assertEqual(len(no_failures), 0)

    def test_get_unresolved_failures(self):
        """Test getting unresolved failures."""
        # Initially all failures are unresolved
        unresolved = self.feedback_system.get_unresolved_failures()
        self.assertEqual(len(unresolved), 3)
        
        # Mark one as resolved
        self.feedback_system.mark_as_resolved("test-001")
        
        # Now there should be one less unresolved
        unresolved = self.feedback_system.get_unresolved_failures()
        self.assertEqual(len(unresolved), 2)

    def test_get_high_severity_failures(self):
        """Test getting high severity failures."""
        # Get failures with severity >= 4
        high_severity = self.feedback_system.get_high_severity_failures(min_severity=4)
        self.assertEqual(len(high_severity), 1)
        self.assertEqual(high_severity[0].failure_id, "test-003")
        
        # Get failures with severity >= 3
        medium_severity = self.feedback_system.get_high_severity_failures(min_severity=3)
        self.assertEqual(len(medium_severity), 2)

    def test_get_common_failure_types(self):
        """Test getting common failure types."""
        # Get top failure types
        common_types = self.feedback_system.get_common_failure_types(top_n=2)
        
        # MISSING_ENTITY should be the most common (2 occurrences)
        self.assertEqual(common_types[0][0], ValidationFailureType.MISSING_ENTITY)
        self.assertEqual(common_types[0][1], 2)
        
        # AMBIGUOUS_INTENT should be second (1 occurrence)
        self.assertEqual(common_types[1][0], ValidationFailureType.AMBIGUOUS_INTENT)
        self.assertEqual(common_types[1][1], 1)

    def test_get_common_entity_failures(self):
        """Test getting common entity failures."""
        # Get top entity failures
        common_entities = self.feedback_system.get_common_entity_failures(top_n=2)
        
        # Should have date and location with 1 occurrence each
        self.assertEqual(len(common_entities), 2)
        entity_types = [entity for entity, count in common_entities]
        self.assertIn("date", entity_types)
        self.assertIn("location", entity_types)
        
        # All should have count 1
        for _, count in common_entities:
            self.assertEqual(count, 1)

    def test_get_improvement_suggestions(self):
        """Test generating improvement suggestions."""
        # Get improvement suggestions
        suggestions = self.feedback_system.get_improvement_suggestions()
        
        # Should have suggestions for entity extraction
        self.assertIn("entity_extraction", suggestions)
        self.assertTrue(len(suggestions["entity_extraction"]) > 0)
        
        # Should have suggestions for intent classification
        self.assertIn("intent_classification", suggestions)
        self.assertTrue(len(suggestions["intent_classification"]) > 0)
        
        # Should have suggestions for ambiguity resolution
        self.assertIn("ambiguity_resolution", suggestions)
        self.assertTrue(len(suggestions["ambiguity_resolution"]) > 0)

    def test_generate_improvement_report(self):
        """Test generating an improvement report."""
        # Generate report
        report = self.feedback_system.generate_improvement_report()
        
        # Check that it's a non-empty string
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
        
        # Check that it contains key sections
        self.assertIn("# Validation Improvement Report", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Common Failure Types", report)
        self.assertIn("## Improvement Suggestions", report)


if __name__ == "__main__":
    unittest.main()
