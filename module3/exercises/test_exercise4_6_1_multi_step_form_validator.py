"""
Tests for Exercise 4.6.1: Multi-Step Form Validator
--------------------------------------------------
This module contains tests for the multi-step form validator implementation.
"""

import unittest
from datetime import datetime
from typing import Dict, Any, List

from exercise4_6_1_multi_step_form_validator import (
    FormStepStatus,
    FieldType,
    ValidationRule,
    FormField,
    FormStepDefinition,
    ValidationError,
    FormStepState,
    MultiStepForm
)


class TestMultiStepFormValidator(unittest.TestCase):
    """Test cases for the multi-step form validator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple two-step form for testing
        self.personal_step = FormStepDefinition(
            step_id="personal",
            title="Personal Information",
            fields=[
                FormField(
                    name="name",
                    label="Full Name",
                    field_type=FieldType.TEXT,
                    validation_rules=[
                        ValidationRule.required(),
                        ValidationRule.min_length(2)
                    ]
                ),
                FormField(
                    name="email",
                    label="Email",
                    field_type=FieldType.EMAIL,
                    validation_rules=[
                        ValidationRule.required(),
                        ValidationRule.email()
                    ]
                )
            ],
            next_steps=["contact"]
        )

        self.contact_step = FormStepDefinition(
            step_id="contact",
            title="Contact Information",
            fields=[
                FormField(
                    name="phone",
                    label="Phone Number",
                    field_type=FieldType.TEXT,
                    validation_rules=[
                        ValidationRule.required(),
                        ValidationRule.pattern(r'^\d{10}$', "Phone must be 10 digits")
                    ]
                ),
                FormField(
                    name="address",
                    label="Address",
                    field_type=FieldType.TEXTAREA,
                    required=False
                )
            ],
            previous_step="personal"
        )

        self.form = MultiStepForm(
            title="Test Form",
            steps={
                "personal": self.personal_step,
                "contact": self.contact_step
            },
            start_step_id="personal"
        )

    def test_form_initialization(self):
        """Test that the form initializes correctly."""
        self.assertEqual(self.form.title, "Test Form")
        self.assertEqual(self.form.current_step_id, "personal")
        self.assertEqual(len(self.form.steps), 2)
        self.assertFalse(self.form.is_submitted)

        # Check step states
        self.assertEqual(len(self.form.step_states), 2)
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.IN_PROGRESS)
        self.assertTrue(self.form.step_states["personal"].visited)
        self.assertEqual(self.form.step_states["contact"].status, FormStepStatus.NOT_STARTED)
        self.assertFalse(self.form.step_states["contact"].visited)

    def test_field_validation(self):
        """Test validation of individual fields."""
        name_field = self.personal_step.fields[0]

        # Test required validation
        errors = self.form.validate_field(name_field, None)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], "This field is required")

        # Test min length validation
        errors = self.form.validate_field(name_field, "A")
        self.assertEqual(len(errors), 1)
        self.assertTrue("at least 2 characters" in errors[0])

        # Test valid input
        errors = self.form.validate_field(name_field, "John Doe")
        self.assertEqual(len(errors), 0)

        # Test email validation
        email_field = self.personal_step.fields[1]
        errors = self.form.validate_field(email_field, "not-an-email")
        self.assertEqual(len(errors), 1)
        self.assertTrue("Invalid email" in errors[0])

        errors = self.form.validate_field(email_field, "valid@example.com")
        self.assertEqual(len(errors), 0)

    def test_step_validation(self):
        """Test validation of an entire step."""
        # Test with invalid data
        invalid_data = {
            "name": "J",  # Too short
            "email": "invalid-email"  # Invalid email
        }

        errors = self.form.validate_step("personal", invalid_data)
        self.assertEqual(len(errors), 2)

        # Test with valid data
        valid_data = {
            "name": "John Doe",
            "email": "john@example.com"
        }

        errors = self.form.validate_step("personal", valid_data)
        self.assertEqual(len(errors), 0)

    def test_update_step_data(self):
        """Test updating step data."""
        # Update with invalid data
        result = self.form.update_step_data("personal", {"name": "J"})
        self.assertTrue(result["success"])
        # There are two errors: name is too short and email is missing
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.INVALID)

        # Update with valid data
        result = self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        self.assertTrue(result["success"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.IN_PROGRESS)

    def test_complete_step(self):
        """Test completing a step."""
        # Try to complete with invalid data
        self.form.update_step_data("personal", {"name": "J"}, validate=False)
        result = self.form.complete_step("personal")
        self.assertFalse(result["success"])
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.INVALID)

        # Complete with valid data
        self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        result = self.form.complete_step("personal")
        self.assertTrue(result["success"])
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.COMPLETED)
        self.assertIsNotNone(self.form.step_states["personal"].completed_at)

    def test_navigation(self):
        """Test navigation between steps."""
        # Cannot go to next step without completing current step
        result = self.form.next_step()
        self.assertFalse(result["success"])

        # Complete current step and go to next
        self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        self.form.complete_step("personal")

        result = self.form.next_step()
        self.assertTrue(result["success"])
        self.assertEqual(self.form.current_step_id, "contact")

        # Go back to previous step
        result = self.form.previous_step()
        self.assertTrue(result["success"])
        self.assertEqual(self.form.current_step_id, "personal")

        # Go directly to a step
        result = self.form.go_to_step("contact")
        self.assertTrue(result["success"])
        self.assertEqual(self.form.current_step_id, "contact")

    def test_form_completion(self):
        """Test completing and submitting the form."""
        # Form should not be complete initially
        self.assertFalse(self.form.is_form_complete())

        # Complete first step
        self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        self.form.complete_step("personal")

        # Go to next step
        self.form.next_step()

        # Complete second step
        self.form.update_step_data("contact", {
            "phone": "1234567890",
            "address": "123 Main St"
        })
        self.form.complete_step("contact")

        # Now form should be complete
        self.assertTrue(self.form.is_form_complete())

        # Submit the form
        result = self.form.submit_form()
        self.assertTrue(result["success"])
        self.assertTrue(self.form.is_submitted)

        # Check submission data
        expected_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "address": "123 Main St"
        }
        self.assertEqual(self.form.submission_data, expected_data)

    def test_form_reset(self):
        """Test resetting the form."""
        # Fill out and complete the form
        self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        self.form.complete_step("personal")
        self.form.next_step()

        self.form.update_step_data("contact", {
            "phone": "1234567890",
            "address": "123 Main St"
        })
        self.form.complete_step("contact")

        self.form.submit_form()

        # Reset the form
        result = self.form.reset_form()
        self.assertTrue(result["success"])

        # Check that form is reset
        self.assertEqual(self.form.current_step_id, "personal")
        self.assertFalse(self.form.is_submitted)
        self.assertEqual(self.form.submission_data, {})

        # Check step states
        self.assertEqual(self.form.step_states["personal"].status, FormStepStatus.IN_PROGRESS)
        self.assertTrue(self.form.step_states["personal"].visited)
        self.assertEqual(self.form.step_states["contact"].status, FormStepStatus.NOT_STARTED)
        self.assertFalse(self.form.step_states["contact"].visited)
        self.assertEqual(self.form.step_states["personal"].data, {})
        self.assertEqual(self.form.step_states["contact"].data, {})

    def test_cannot_skip_steps(self):
        """Test that steps cannot be skipped without completing previous steps."""
        # Try to go to contact step without completing personal step
        result = self.form.go_to_step("contact")
        self.assertFalse(result["success"])
        self.assertEqual(self.form.current_step_id, "personal")

    def test_cannot_submit_incomplete_form(self):
        """Test that an incomplete form cannot be submitted."""
        result = self.form.submit_form()
        self.assertFalse(result["success"])
        self.assertFalse(self.form.is_submitted)
        self.assertEqual(len(result["incomplete_steps"]), 2)

    def test_get_form_data(self):
        """Test getting all form data."""
        # Fill out the form
        self.form.update_step_data("personal", {
            "name": "John Doe",
            "email": "john@example.com"
        })

        self.form.update_step_data("contact", {
            "phone": "1234567890",
            "address": "123 Main St"
        })

        # Get form data
        data = self.form.get_form_data()
        expected_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "address": "123 Main St"
        }
        self.assertEqual(data, expected_data)


if __name__ == "__main__":
    unittest.main()
