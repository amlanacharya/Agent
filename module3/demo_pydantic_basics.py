"""
Standalone Demonstration Script for Pydantic Basics
------------------------------------------------
This script demonstrates the basic usage of Pydantic for data validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
import json


class User(BaseModel):
    """Basic user model demonstrating simple field types."""
    id: int
    name: str
    email: str
    age: Optional[int] = None
    tags: List[str] = []


class Product(BaseModel):
    """Product model demonstrating field constraints."""
    id: int
    name: str = Field(..., min_length=3, max_length=50)
    price: float = Field(..., gt=0)
    description: Optional[str] = Field(None, max_length=1000)


class SignupForm(BaseModel):
    """Signup form demonstrating custom validators."""
    username: str
    password: str
    password_confirm: str
    
    @field_validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @field_validator('password_confirm')
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v


def main():
    """Main demonstration function."""
    print("=== Pydantic Basics Demonstration ===")
    
    # Basic model validation
    print("\n1. Basic Model Validation")
    print("--------------------------")
    
    # Valid user
    user = User(id=1, name="John Doe", email="john@example.com")
    print(f"Valid user: {user}")
    
    # Type coercion
    user_with_coercion = User(id="42", name="Jane Doe", email="jane@example.com")
    print(f"User with coerced ID: {user_with_coercion.id} (type: {type(user_with_coercion.id).__name__})")
    
    # Invalid user
    try:
        invalid_user = User(id="not_an_int", name=123, email="invalid_email")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Field constraints
    print("\n2. Field Constraints")
    print("-------------------")
    
    # Valid product
    product = Product(id=1, name="Laptop", price=999.99)
    print(f"Valid product: {product}")
    
    # Invalid product - name too short
    try:
        product = Product(id=2, name="PC", price=1299.99)
    except Exception as e:
        print(f"Validation error (name too short): {e}")
    
    # Invalid product - negative price
    try:
        product = Product(id=3, name="Tablet", price=-199.99)
    except Exception as e:
        print(f"Validation error (negative price): {e}")
    
    # Custom validators
    print("\n3. Custom Validators")
    print("-------------------")
    
    # Valid signup form
    form = SignupForm(
        username="johndoe", 
        password="password123", 
        password_confirm="password123"
    )
    print(f"Valid signup form: {form}")
    
    # Invalid signup form - non-alphanumeric username
    try:
        form = SignupForm(
            username="john.doe", 
            password="password123", 
            password_confirm="password123"
        )
    except Exception as e:
        print(f"Validation error (non-alphanumeric username): {e}")
    
    # Invalid signup form - password mismatch
    try:
        form = SignupForm(
            username="johndoe", 
            password="password123", 
            password_confirm="different"
        )
    except Exception as e:
        print(f"Validation error (password mismatch): {e}")
    
    # Serialization
    print("\n4. Serialization")
    print("----------------")
    
    # Convert to dictionary
    user_dict = user.model_dump()
    print(f"User as dictionary: {user_dict}")
    
    # Convert to JSON
    user_json = user.model_dump_json()
    print(f"User as JSON: {user_json}")
    
    # Parse from JSON
    json_data = '{"id": 3, "name": "Bob Smith", "email": "bob@example.com", "age": 35}'
    parsed_user = User.model_validate_json(json_data)
    print(f"User parsed from JSON: {parsed_user}")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
