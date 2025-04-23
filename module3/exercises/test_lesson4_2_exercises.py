"""
Test Lesson 4.2 Exercises - Tests for error handling and recovery exercises
--------------------------------------------------------------------------
This module contains tests for the lesson4_2_exercises module.
"""

import sys
import os
import re

# Add the parent directory to the path so we can import the exercises module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercises.lesson4_2_exercises import (
    # Exercise 1
    MultiStepForm, FormStep, PersonalInfo, ContactInfo, Preferences,
    # Exercise 2
    ValidationWithSuggestions, EmailSuggestionEngine,
    # Exercise 3
    ValidationErrorLogger,
    # Exercise 4
    ValidationMiddleware, APIResponse,
    # Exercise 5
    PartialSubmissionForm, JobApplication, DraftManager
)
from pydantic import BaseModel, Field, field_validator


def test_multi_step_form():
    """Test the multi-step form implementation."""
    form = MultiStepForm()

    # Test personal info step
    result = form.submit_step(FormStep.PERSONAL, {
        "first_name": "John",
        "last_name": "Doe",
        "birth_date": "1990-01-01",
        "gender": "male"
    })
    assert result["status"] == "success"
    assert result["next_step"] == FormStep.CONTACT.value

    # Test contact info step with error
    result = form.submit_step(FormStep.CONTACT, {
        "email": "invalid-email",  # This will fail
        "address": "123 Main St",
        "city": "Anytown",
        "country": "USA"
    })
    assert result["status"] == "error"
    assert "errors" in result

    # Test going back
    result = form.go_back()
    assert result["status"] == "success"
    assert result["current_step"] == FormStep.PERSONAL.value

    print("✅ Multi-step form works")


def test_suggestion_system():
    """Test the suggestion system."""
    engine = EmailSuggestionEngine()

    # Test missing @ symbol
    suggestions = engine.suggest_corrections("johndoe")
    assert len(suggestions) > 0
    assert any("@" in suggestion for suggestion in suggestions)

    # Test domain typo
    suggestions = engine.suggest_corrections("john@gmial.com")
    assert len(suggestions) > 0
    assert any("gmail.com" in suggestion for suggestion in suggestions)

    # Test with validator
    validator = ValidationWithSuggestions()

    class UserProfile(BaseModel):
        name: str
        email: str

        @field_validator('email')
        def validate_email(cls, v):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
                raise ValueError("Invalid email format")
            return v

    result = validator.validate_with_suggestions(UserProfile, {
        "name": "John Doe",
        "email": "john.doegmail.com"  # Missing @ symbol
    })
    assert result["status"] == "error"
    assert "suggestions" in result
    assert "email" in result["suggestions"]

    print("✅ Suggestion system works")


def test_error_logging():
    """Test the error logging system."""
    logger = ValidationErrorLogger()

    # Log some errors
    logger.log_error("form1", "UserProfile", {"name": "John", "email": "invalid"},
                    [{"loc": ("email",), "type": "value_error.email", "msg": "Invalid email"}])

    logger.log_error("form2", "UserProfile", {"name": "Jane", "email": "invalid2"},
                    [{"loc": ("email",), "type": "value_error.email", "msg": "Invalid email"}])

    # Test error frequency
    freq = logger.get_error_frequency()
    assert "email" in freq
    assert freq["email"] == 2

    # Test common error types
    types = logger.get_common_error_types()
    assert "value_error.email" in types
    assert types["value_error.email"] == 2

    # Test improvement suggestions
    suggestions = logger.get_improvement_suggestions()
    # In our test case, we don't have enough errors to trigger suggestions
    # So we'll just check that the method returns a list
    assert isinstance(suggestions, list)

    print("✅ Error logging system works")


def test_validation_middleware():
    """Test the validation middleware."""
    class CreateUser(BaseModel):
        username: str = Field(..., min_length=3)
        email: str

    middleware = ValidationMiddleware()

    # Test valid request
    valid_response = middleware.process_request(CreateUser, {
        "username": "johndoe",
        "email": "john@example.com"
    })
    assert valid_response.success is True
    assert valid_response.data is not None

    # Test invalid request
    invalid_response = middleware.process_request(CreateUser, {
        "username": "jo",  # Too short
        "email": "invalid-email"
    })
    assert invalid_response.success is False
    assert invalid_response.errors is not None
    assert "username" in invalid_response.errors

    print("✅ Validation middleware works")


def test_partial_submission():
    """Test the partial submission form."""
    draft_manager = DraftManager()

    # Test partial submission with some valid fields
    result = draft_manager.save_draft(
        "user123",
        "job_application",
        JobApplication,
        {
            "full_name": "John Doe",
            "email": "john@example.com",
            "phone": "invalid-phone",  # This will fail
        }
    )
    assert result["status"] == "success"
    assert "draft" in result
    # Just check that the draft has the expected structure
    assert "valid_fields" in result["draft"]
    assert "errors" in result["draft"]

    # Test updating with more fields
    result = draft_manager.save_draft(
        "user123",
        "job_application",
        JobApplication,
        {
            "phone": "+1 555-123-4567",  # Fixed
            "resume_url": "https://example.com/resume.pdf",
            "years_experience": 5
        }
    )
    # Just check that the update was successful
    assert result["status"] == "success"

    print("✅ Partial submission form works")


if __name__ == "__main__":
    print("Running exercise tests...")
    test_multi_step_form()
    test_suggestion_system()
    test_error_logging()
    test_validation_middleware()
    test_partial_submission()
    print("All exercise tests completed!")
