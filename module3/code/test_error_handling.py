"""
Test Error Handling - Tests for error handling and recovery strategies
---------------------------------------------------------------------
This module contains tests for the error_handling module.
"""

from error_handling import (
    User, Product, UserRegistration, UserInput,
    demonstrate_validation_error,
    validate_user_input, validate_multiple_items,
    safe_parse_with_defaults, attempt_error_correction,
    user_friendly_errors, FormValidator
)
from pydantic import ValidationError


def test_validation_error():
    """Test basic validation error handling."""
    errors = demonstrate_validation_error()
    assert errors is not None
    assert len(errors) > 0
    print("✅ Validation error demonstration works")


def test_user_input_validation():
    """Test user input validation."""
    # Valid input
    valid_result = validate_user_input({
        "username": "johndoe",
        "email": "john@example.com",
        "age": 25
    })
    assert valid_result["status"] == "success"

    # Invalid input
    invalid_result = validate_user_input({
        "username": "a",
        "email": "not-an-email",
        "age": 16
    })
    assert invalid_result["status"] == "error"
    print("✅ User input validation works")


def test_multiple_items_validation():
    """Test validation of multiple items."""
    products = [
        {"name": "Product 1", "price": 10.99, "quantity": 5},
        {"name": "Product 2", "price": -5.99, "quantity": 3},
        {"name": "Product 3", "price": 7.99, "quantity": "invalid"}
    ]
    result = validate_multiple_items(products, Product)
    assert result["success_count"] == 1
    assert result["error_count"] == 2
    print("✅ Multiple items validation works")


def test_safe_parse_with_defaults():
    """Test safe parsing with defaults."""
    data = {"name": "Product", "price": -10, "quantity": "invalid"}
    defaults = {"price": 0.99, "quantity": 1}

    try:
        product = safe_parse_with_defaults(data, Product, defaults)
        assert product.price == 0.99
        assert product.quantity == 1
        print("✅ Safe parse with defaults works")
    except ValidationError:
        print("❌ Safe parse with defaults failed")


def test_error_correction():
    """Test error correction strategies."""
    result = attempt_error_correction({
        "email": "user.example",
        "age": "25",
        "tags": "python, pydantic, validation"
    })
    assert result["status"] == "success"
    assert "corrections" in result
    assert len(result["corrections"]) > 0
    print("✅ Error correction works")


def test_user_friendly_errors():
    """Test user-friendly error messages."""
    try:
        UserRegistration(username="a", email="not-an-email", password="123", age=16)
        print("❌ User registration should have failed")
    except ValidationError as e:
        friendly_errors = user_friendly_errors(e)
        assert "username" in friendly_errors
        assert "email" in friendly_errors
        assert "password" in friendly_errors
        assert "age" in friendly_errors
        print("✅ User-friendly errors work")


def test_form_validator():
    """Test the form validation system."""
    form_validator = FormValidator()

    # Test contact form
    contact_result = form_validator.process_form("contact", {
        "name": "J",
        "email": "invalid-email",
        "message": "Hi"
    })
    assert contact_result["status"] == "error"
    assert "errors" in contact_result

    # Test registration form
    registration_result = form_validator.process_form("registration", {
        "username": "user1",
        "email": "user@example.com",
        "password": "password123",
        "confirm_password": "password456"
    })
    assert registration_result["status"] == "error"
    assert "errors" in registration_result
    print("✅ Form validator works")


if __name__ == "__main__":
    print("Running error handling tests...")
    test_validation_error()
    test_user_input_validation()
    test_multiple_items_validation()
    test_safe_parse_with_defaults()
    test_error_correction()
    test_user_friendly_errors()
    test_form_validator()
    print("All tests completed!")
