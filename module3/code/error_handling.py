"""
Error Handling and Recovery - Advanced Pydantic Validation Error Handling
------------------------------------------------------------------------
This module demonstrates advanced error handling and recovery strategies for Pydantic validation failures.
"""

from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import List, Dict, Any, Optional, Union
import re
from datetime import date


# Basic Models for Demonstration
# -----------------------------

class User(BaseModel):
    """Basic user model for error handling demonstrations."""
    username: str = Field(..., min_length=3)
    email: str
    age: int = Field(..., ge=18)
    tags: List[str] = []


class Product(BaseModel):
    """Product model for error handling demonstrations."""
    name: str
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)
    tags: List[str] = []


class UserRegistration(BaseModel):
    """User registration model with validation rules."""
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=18)
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v


class UserInput(BaseModel):
    """User input model for error correction demonstrations."""
    email: str
    age: int
    tags: List[str]


# 1. Understanding ValidationError
# ------------------------------

def demonstrate_validation_error():
    """Demonstrate how to access and use ValidationError information."""
    try:
        User(username="a", email="not-an-email", age=16, tags=["one", 2])
    except ValidationError as e:
        print(f"Error: {e}")
        
        # Access structured error data
        print("\nError details:")
        for error in e.errors():
            print(f"- Location: {'.'.join(str(loc) for loc in error['loc'])}")
            print(f"  Type: {error['type']}")
            print(f"  Message: {error['msg']}")
        
        return e.errors()
    
    return None


# 2. Customizing Error Messages
# ---------------------------

class UserWithCustomErrors(BaseModel):
    """User model with custom error messages."""
    username: str = Field(
        ..., 
        min_length=3,
        description="Username must be at least 3 characters long"
    )
    email: str
    age: int = Field(
        ..., 
        ge=18,
        description="User must be at least 18 years old"
    )
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError("Email must contain an @ symbol")
        return v


# 3. Error Handling Strategies
# --------------------------

# 3.1 Try-Except Pattern
def validate_user_input(data: Dict[str, Any]):
    """Basic try-except pattern for validation."""
    try:
        user = User(**data)
        return {"status": "success", "user": user.model_dump()}
    except ValidationError as e:
        return {"status": "error", "message": "Invalid user data", "details": e.errors()}


# 3.2 Error Aggregation
def validate_multiple_items(items: List[Dict[str, Any]], model_class):
    """Validate multiple items and aggregate errors."""
    valid_items = []
    errors = {}
    
    for i, item in enumerate(items):
        try:
            valid_item = model_class(**item)
            valid_items.append(valid_item)
        except ValidationError as e:
            errors[i] = e.errors()
    
    return {
        "valid_items": valid_items,
        "errors": errors,
        "success_count": len(valid_items),
        "error_count": len(errors)
    }


# 4. Advanced Error Recovery Techniques
# -----------------------------------

# 4.1 Default Value Substitution
def safe_parse_with_defaults(data: Dict[str, Any], model_class, default_values: Dict[str, Any]):
    """Try to parse data, substituting defaults for invalid fields."""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get error locations
        error_fields = ['.'.join(str(loc) for loc in error['loc']) for error in e.errors()]
        
        # Create a new dict with defaults for error fields
        fixed_data = {**data}
        for field in error_fields:
            if field in default_values:
                fixed_data[field] = default_values[field]
        
        # Try again with fixed data
        try:
            return model_class(**fixed_data)
        except ValidationError:
            # If still failing, raise the original error
            raise e


# 4.2 Error Correction Strategies
def attempt_error_correction(data: Dict[str, Any]):
    """Try to correct common errors in input data."""
    corrected = {**data}
    corrections_applied = []
    
    # Fix email format
    if 'email' in data and isinstance(data['email'], str) and '@' not in data['email']:
        # Try to guess if it's missing the domain
        if not re.search(r'@[^@]+\.[^@]+$', data['email']):
            corrected['email'] = f"{data['email']}@example.com"
            corrections_applied.append(f"Added default domain to email: {corrected['email']}")
    
    # Convert string age to int
    if 'age' in data and isinstance(data['age'], str):
        try:
            corrected['age'] = int(data['age'])
            corrections_applied.append(f"Converted age from string to int: {corrected['age']}")
        except ValueError:
            pass
    
    # Convert string tags to list
    if 'tags' in data and isinstance(data['tags'], str):
        corrected['tags'] = [tag.strip() for tag in data['tags'].split(',')]
        corrections_applied.append(f"Converted tags from string to list: {corrected['tags']}")
    
    # Try to validate with corrections
    try:
        validated = UserInput(**corrected)
        return {
            "status": "success",
            "data": validated.model_dump(),
            "corrections": corrections_applied
        }
    except ValidationError as e:
        return {
            "status": "error",
            "message": "Validation failed even with corrections",
            "original_data": data,
            "attempted_corrections": corrections_applied,
            "errors": e.errors()
        }


# 5. User-Friendly Error Messages
# -----------------------------

def user_friendly_errors(validation_error: ValidationError) -> Dict[str, List[str]]:
    """Convert validation errors to user-friendly messages."""
    error_messages = {}
    
    # Error message mapping
    friendly_messages = {
        "string_too_short": "This field is too short",
        "string_too_long": "This field is too long",
        "value_error.email": "Please enter a valid email address",
        "greater_than_equal": "This value must be greater than or equal to {limit_value}",
        "less_than_equal": "This value must be less than or equal to {limit_value}",
        "type_error.integer": "Please enter a whole number",
        "type_error.float": "Please enter a number",
        "value_error.missing": "This field is required"
    }
    
    for error in validation_error.errors():
        # Get the field name (first item in location)
        field = error['loc'][0] if error['loc'] else 'general'
        
        # Get the error type
        error_type = error['type']
        
        # Get the error message
        if error_type in friendly_messages:
            message = friendly_messages[error_type]
            
            # Replace placeholders if needed
            if '{limit_value}' in message and 'ctx' in error and 'limit_value' in error['ctx']:
                message = message.format(limit_value=error['ctx']['limit_value'])
        else:
            # Use the original message if no friendly version is available
            message = error['msg']
        
        if field not in error_messages:
            error_messages[field] = []
        error_messages[field].append(message)
    
    return error_messages


# 6. Practical Example: Form Validation System
# -----------------------------------------

class ContactForm(BaseModel):
    """Contact form model."""
    name: str = Field(..., min_length=2)
    email: str
    message: str = Field(..., min_length=10)
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v


class RegistrationForm(BaseModel):
    """Registration form model."""
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str = Field(..., min_length=8)
    confirm_password: str
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('confirm_password')
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError("Passwords do not match")
        return v


class FormValidator:
    """Form validation system with error handling."""
    
    def __init__(self):
        """Initialize with form types."""
        self.form_types = {
            "contact": ContactForm,
            "registration": RegistrationForm
        }
    
    def _format_errors(self, validation_error: ValidationError) -> Dict[str, List[str]]:
        """Format validation errors to be user-friendly."""
        return user_friendly_errors(validation_error)
    
    def process_form(self, form_type: str, data: Dict[str, Any]):
        """Process a form submission with comprehensive error handling."""
        if form_type not in self.form_types:
            return {
                "status": "error",
                "message": f"Unknown form type: {form_type}"
            }
        
        form_class = self.form_types[form_type]
        
        try:
            # Validate form data
            form = form_class(**data)
            
            # Process the valid form (in a real app, this would save to DB, send email, etc.)
            return {
                "status": "success",
                "message": "Form submitted successfully",
                "data": form.model_dump()
            }
        except ValidationError as e:
            # Convert to user-friendly errors
            friendly_errors = self._format_errors(e)
            
            return {
                "status": "error",
                "message": "Please fix the errors in your submission",
                "errors": friendly_errors
            }


# Example Usage
# -----------

if __name__ == "__main__":
    print("1. Understanding ValidationError")
    print("-" * 40)
    demonstrate_validation_error()
    print("\n")
    
    print("2. Error Handling Strategies")
    print("-" * 40)
    # Try-Except Pattern
    user_result = validate_user_input({
        "username": "a",
        "email": "not-an-email",
        "age": 16
    })
    print(f"User validation result: {user_result}")
    
    # Error Aggregation
    products = [
        {"name": "Product 1", "price": 10.99, "quantity": 5},
        {"name": "Product 2", "price": -5.99, "quantity": 3},
        {"name": "Product 3", "price": 7.99, "quantity": "invalid"}
    ]
    validation_result = validate_multiple_items(products, Product)
    print(f"\nMultiple item validation:")
    print(f"Valid items: {len(validation_result['valid_items'])}")
    print(f"Error count: {validation_result['error_count']}")
    print("\n")
    
    print("3. Advanced Error Recovery")
    print("-" * 40)
    # Default Value Substitution
    try:
        data = {"name": "Product", "price": -10, "quantity": "invalid"}
        defaults = {"price": 0.99, "quantity": 1}
        product = safe_parse_with_defaults(data, Product, defaults)
        print(f"Recovered with defaults: {product}")
    except ValidationError as e:
        print(f"Could not recover: {e}")
    
    # Error Correction
    correction_result = attempt_error_correction({
        "email": "user.example",
        "age": "25",
        "tags": "python, pydantic, validation"
    })
    print(f"\nError correction result: {correction_result}")
    print("\n")
    
    print("4. User-Friendly Error Messages")
    print("-" * 40)
    try:
        UserRegistration(username="a", email="not-an-email", password="123", age=16)
    except ValidationError as e:
        friendly_errors = user_friendly_errors(e)
        print("Please fix the following errors:")
        for field, messages in friendly_errors.items():
            print(f"{field.capitalize()}:")
            for msg in messages:
                print(f"  - {msg}")
    print("\n")
    
    print("5. Form Validation System")
    print("-" * 40)
    form_validator = FormValidator()
    
    # Process a contact form
    contact_result = form_validator.process_form("contact", {
        "name": "J",
        "email": "invalid-email",
        "message": "Hi"
    })
    print("Contact Form Result:")
    print(contact_result)
    
    # Process a registration form
    registration_result = form_validator.process_form("registration", {
        "username": "user1",
        "email": "user@example.com",
        "password": "password123",
        "confirm_password": "password456"
    })
    print("\nRegistration Form Result:")
    print(registration_result)
