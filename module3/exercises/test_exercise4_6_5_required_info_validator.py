"""
Tests for Exercise 4.6.5: Required Information Validator

This module contains tests for the RequiredInfoValidator class and related components.
"""

import unittest
from datetime import datetime
from typing import Dict, Any, List, Optional

from exercise4_6_5_required_info_validator import (
    FieldRequirement, FieldPriority, FieldDefinition, TaskDefinition,
    ValidationResult, FieldStatus, TaskState, RequiredInfoValidator
)


class TestFieldDefinition(unittest.TestCase):
    """Tests for the FieldDefinition class."""
    
    def test_validate_string_field(self):
        """Test validation of string fields."""
        field = FieldDefinition(
            name="test_field",
            description="Test field",
            field_type="string",
            min_length=3,
            max_length=10,
            validation_pattern=r"^[A-Za-z]+$"
        )
        
        # Valid value
        is_valid, error = field.validate_value("Valid")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Too short
        is_valid, error = field.validate_value("AB")
        self.assertFalse(is_valid)
        self.assertIn("at least 3 characters", error)
        
        # Too long
        is_valid, error = field.validate_value("ThisIsTooLong")
        self.assertFalse(is_valid)
        self.assertIn("at most 10 characters", error)
        
        # Invalid pattern
        is_valid, error = field.validate_value("Invalid123")
        self.assertFalse(is_valid)
        self.assertIn("pattern", error)
        
        # None value for required field
        field.requirement = FieldRequirement.REQUIRED
        is_valid, error = field.validate_value(None)
        self.assertFalse(is_valid)
        self.assertIn("required", error)
        
        # None value for optional field
        field.requirement = FieldRequirement.OPTIONAL
        is_valid, error = field.validate_value(None)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_number_field(self):
        """Test validation of number fields."""
        field = FieldDefinition(
            name="test_field",
            description="Test field",
            field_type="number",
            min_value=0,
            max_value=100
        )
        
        # Valid value
        is_valid, error = field.validate_value(50)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Too small
        is_valid, error = field.validate_value(-10)
        self.assertFalse(is_valid)
        self.assertIn("at least 0", error)
        
        # Too large
        is_valid, error = field.validate_value(200)
        self.assertFalse(is_valid)
        self.assertIn("at most 100", error)
        
        # Invalid type
        is_valid, error = field.validate_value("50")
        self.assertFalse(is_valid)
        self.assertIn("Expected number", error)
    
    def test_validate_boolean_field(self):
        """Test validation of boolean fields."""
        field = FieldDefinition(
            name="test_field",
            description="Test field",
            field_type="boolean"
        )
        
        # Valid values
        is_valid, error = field.validate_value(True)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        is_valid, error = field.validate_value(False)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Invalid type
        is_valid, error = field.validate_value("true")
        self.assertFalse(is_valid)
        self.assertIn("Expected boolean", error)
    
    def test_validate_allowed_values(self):
        """Test validation of allowed values."""
        field = FieldDefinition(
            name="test_field",
            description="Test field",
            field_type="string",
            allowed_values=["option1", "option2", "option3"]
        )
        
        # Valid value
        is_valid, error = field.validate_value("option1")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Invalid value
        is_valid, error = field.validate_value("option4")
        self.assertFalse(is_valid)
        self.assertIn("must be one of", error)


class TestTaskDefinition(unittest.TestCase):
    """Tests for the TaskDefinition class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.task_def = TaskDefinition(
            task_type="test_task",
            description="Test task",
            fields=[
                FieldDefinition(
                    name="required_field",
                    description="Required field",
                    requirement=FieldRequirement.REQUIRED
                ),
                FieldDefinition(
                    name="optional_field",
                    description="Optional field",
                    requirement=FieldRequirement.OPTIONAL
                ),
                FieldDefinition(
                    name="conditional_field",
                    description="Conditional field",
                    requirement=FieldRequirement.CONDITIONAL,
                    depends_on={"required_field": "specific_value"}
                )
            ]
        )
    
    def test_get_field(self):
        """Test getting a field by name."""
        field = self.task_def.get_field("required_field")
        self.assertIsNotNone(field)
        self.assertEqual(field.name, "required_field")
        
        field = self.task_def.get_field("nonexistent_field")
        self.assertIsNone(field)
    
    def test_get_required_fields(self):
        """Test getting required fields."""
        required_fields = self.task_def.get_required_fields()
        self.assertEqual(len(required_fields), 1)
        self.assertEqual(required_fields[0].name, "required_field")
    
    def test_get_optional_fields(self):
        """Test getting optional fields."""
        optional_fields = self.task_def.get_optional_fields()
        self.assertEqual(len(optional_fields), 1)
        self.assertEqual(optional_fields[0].name, "optional_field")
    
    def test_get_conditional_fields(self):
        """Test getting conditional fields."""
        conditional_fields = self.task_def.get_conditional_fields()
        self.assertEqual(len(conditional_fields), 1)
        self.assertEqual(conditional_fields[0].name, "conditional_field")


class TestTaskState(unittest.TestCase):
    """Tests for the TaskState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.task_state = TaskState(task_type="test_task")
    
    def test_update_field(self):
        """Test updating a field value."""
        # Update a field
        result = self.task_state.update_field("test_field", "test_value")
        self.assertTrue(result.is_valid)
        
        # Check field status
        self.assertIn("test_field", self.task_state.field_status)
        self.assertTrue(self.task_state.field_status["test_field"].is_filled)
        self.assertEqual(self.task_state.field_status["test_field"].value, "test_value")
        
        # Check last_updated
        self.assertIsNotNone(self.task_state.field_status["test_field"].last_updated)
        self.assertIsNotNone(self.task_state.last_updated)
    
    def test_get_field_value(self):
        """Test getting a field value."""
        # Field not set
        value = self.task_state.get_field_value("test_field")
        self.assertIsNone(value)
        
        # Set field
        self.task_state.update_field("test_field", "test_value")
        
        # Get field value
        value = self.task_state.get_field_value("test_field")
        self.assertEqual(value, "test_value")
    
    def test_get_filled_fields(self):
        """Test getting filled fields."""
        # No fields filled
        filled_fields = self.task_state.get_filled_fields()
        self.assertEqual(len(filled_fields), 0)
        
        # Fill fields
        self.task_state.update_field("field1", "value1")
        self.task_state.update_field("field2", "value2")
        
        # Get filled fields
        filled_fields = self.task_state.get_filled_fields()
        self.assertEqual(len(filled_fields), 2)
        self.assertIn("field1", filled_fields)
        self.assertIn("field2", filled_fields)
    
    def test_get_valid_fields(self):
        """Test getting valid fields."""
        # Fill fields
        self.task_state.update_field("field1", "value1")
        self.task_state.update_field("field2", "value2")
        
        # Mark field as invalid
        self.task_state.field_status["field2"].is_valid = False
        
        # Get valid fields
        valid_fields = self.task_state.get_valid_fields()
        self.assertEqual(len(valid_fields), 1)
        self.assertIn("field1", valid_fields)
        self.assertNotIn("field2", valid_fields)


class TestRequiredInfoValidator(unittest.TestCase):
    """Tests for the RequiredInfoValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create validator
        self.validator = RequiredInfoValidator()
        
        # Register task definitions
        self.validator.register_task_definition(TaskDefinition(
            task_type="weather_query",
            description="Weather information request",
            fields=[
                FieldDefinition(
                    name="location",
                    description="Location",
                    requirement=FieldRequirement.REQUIRED,
                    priority=FieldPriority.HIGH
                ),
                FieldDefinition(
                    name="date",
                    description="Date",
                    requirement=FieldRequirement.OPTIONAL,
                    priority=FieldPriority.MEDIUM
                )
            ]
        ))
        
        self.validator.register_task_definition(TaskDefinition(
            task_type="booking_query",
            description="Booking request",
            fields=[
                FieldDefinition(
                    name="service_type",
                    description="Service type",
                    requirement=FieldRequirement.REQUIRED,
                    priority=FieldPriority.HIGH,
                    allowed_values=["haircut", "massage"]
                ),
                FieldDefinition(
                    name="date",
                    description="Date",
                    requirement=FieldRequirement.REQUIRED,
                    priority=FieldPriority.HIGH
                ),
                FieldDefinition(
                    name="stylist_preference",
                    description="Stylist preference",
                    requirement=FieldRequirement.CONDITIONAL,
                    priority=FieldPriority.MEDIUM,
                    depends_on={"service_type": "haircut"}
                ),
                FieldDefinition(
                    name="massage_type",
                    description="Massage type",
                    requirement=FieldRequirement.CONDITIONAL,
                    priority=FieldPriority.MEDIUM,
                    depends_on={"service_type": "massage"}
                )
            ]
        ))
    
    def test_create_task_state(self):
        """Test creating a task state."""
        # Valid task type
        task_state = self.validator.create_task_state("weather_query")
        self.assertEqual(task_state.task_type, "weather_query")
        
        # Invalid task type
        with self.assertRaises(ValueError):
            self.validator.create_task_state("invalid_task_type")
    
    def test_validate_field(self):
        """Test validating a field value."""
        task_state = self.validator.create_task_state("weather_query")
        
        # Valid value
        result = self.validator.validate_field(task_state, "location", "New York")
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.error_message)
        
        # Invalid field
        with self.assertRaises(ValueError):
            self.validator.validate_field(task_state, "invalid_field", "value")
    
    def test_update_field(self):
        """Test updating a field value."""
        task_state = self.validator.create_task_state("booking_query")
        
        # Valid value
        result = self.validator.update_field(task_state, "service_type", "haircut")
        self.assertTrue(result.is_valid)
        self.assertEqual(task_state.get_field_value("service_type"), "haircut")
        
        # Invalid value
        result = self.validator.update_field(task_state, "service_type", "invalid_service")
        self.assertFalse(result.is_valid)
        self.assertIn("must be one of", result.error_message)
        # Field should not be updated
        self.assertEqual(task_state.get_field_value("service_type"), "haircut")
    
    def test_check_completeness_weather_query(self):
        """Test checking completeness for weather query."""
        task_state = self.validator.create_task_state("weather_query")
        
        # Initial state
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 1)
        self.assertEqual(completeness["missing_required"][0], "location")
        self.assertEqual(completeness["next_field"], "location")
        
        # Update required field
        self.validator.update_field(task_state, "location", "New York")
        
        # Check completeness again
        completeness = self.validator.check_completeness(task_state)
        self.assertTrue(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 0)
        self.assertIsNone(completeness["next_field"])
    
    def test_check_completeness_booking_query(self):
        """Test checking completeness for booking query."""
        task_state = self.validator.create_task_state("booking_query")
        
        # Initial state
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 2)
        self.assertIn("service_type", completeness["missing_required"])
        self.assertIn("date", completeness["missing_required"])
        
        # Update service_type
        self.validator.update_field(task_state, "service_type", "haircut")
        
        # Check completeness again
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 1)
        self.assertEqual(completeness["missing_required"][0], "date")
        self.assertEqual(len(completeness["missing_conditional"]), 1)
        self.assertEqual(completeness["missing_conditional"][0], "stylist_preference")
        
        # Update date
        self.validator.update_field(task_state, "date", "tomorrow")
        
        # Check completeness again
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 0)
        self.assertEqual(len(completeness["missing_conditional"]), 1)
        self.assertEqual(completeness["missing_conditional"][0], "stylist_preference")
        
        # Update stylist_preference
        self.validator.update_field(task_state, "stylist_preference", "John")
        
        # Check completeness again
        completeness = self.validator.check_completeness(task_state)
        self.assertTrue(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_required"]), 0)
        self.assertEqual(len(completeness["missing_conditional"]), 0)
    
    def test_conditional_fields_dependency(self):
        """Test conditional fields dependency."""
        task_state = self.validator.create_task_state("booking_query")
        
        # Set service_type to massage
        self.validator.update_field(task_state, "service_type", "massage")
        self.validator.update_field(task_state, "date", "tomorrow")
        
        # Check completeness
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_conditional"]), 1)
        self.assertEqual(completeness["missing_conditional"][0], "massage_type")
        
        # Change service_type to haircut
        self.validator.update_field(task_state, "service_type", "haircut")
        
        # Check completeness again
        completeness = self.validator.check_completeness(task_state)
        self.assertFalse(completeness["is_complete"])
        self.assertEqual(len(completeness["missing_conditional"]), 1)
        self.assertEqual(completeness["missing_conditional"][0], "stylist_preference")
    
    def test_get_field_prompt(self):
        """Test getting a field prompt."""
        task_state = self.validator.create_task_state("weather_query")
        
        # Get prompt for location
        prompt = self.validator.get_field_prompt(task_state, "location")
        self.assertIn("Please provide Location", prompt)
        
        # Invalid field
        with self.assertRaises(ValueError):
            self.validator.get_field_prompt(task_state, "invalid_field")
        
        # Field with constraints
        booking_task = self.validator.create_task_state("booking_query")
        prompt = self.validator.get_field_prompt(booking_task, "service_type")
        self.assertIn("Please provide Service type", prompt)
        self.assertIn("one of: haircut, massage", prompt)
        
        # Field with error
        result = self.validator.validate_field(booking_task, "service_type", "invalid_service")
        prompt = self.validator.get_field_prompt(booking_task, "service_type")
        self.assertIn("Error:", prompt)


if __name__ == "__main__":
    unittest.main()
