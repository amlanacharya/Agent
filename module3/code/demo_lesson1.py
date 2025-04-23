"""
Demonstration Script for Lesson 1
-------------------------------
This script demonstrates the usage of the Pydantic models and validation from Lesson 1.
"""

from module3.code.pydantic_basics import (
    User,
    Product,
    AdvancedUser,
    SignupForm,
    TaskInput,
    process_task_creation
)
from datetime import datetime, timedelta


def demonstrate_basic_models():
    """Demonstrate basic Pydantic models."""
    print("\n=== Basic Models ===")

    # Create a valid user
    user = User(id=1, name="John Doe", email="john@example.com")
    print(f"Valid user: {user}")

    # Demonstrate type coercion
    user_with_coercion = User(id="42", name="Jane Doe", email="jane@example.com")
    print(f"User with coerced ID: {user_with_coercion.id} (type: {type(user_with_coercion.id)})")

    # Try creating an invalid user
    try:
        invalid_user = User(id="not_an_int", name=123, email="invalid_email")
    except Exception as e:
        print(f"Validation error: {e}")


def demonstrate_field_constraints():
    """Demonstrate field constraints."""
    print("\n=== Field Constraints ===")

    # Create a valid product
    product = Product(id=1, name="Laptop", price=999.99)
    print(f"Valid product: {product}")

    # Try creating products with invalid fields
    try:
        # Name too short
        product = Product(id=2, name="PC", price=1299.99)
    except Exception as e:
        print(f"Validation error (name too short): {e}")

    try:
        # Negative price
        product = Product(id=3, name="Tablet", price=-199.99)
    except Exception as e:
        print(f"Validation error (negative price): {e}")


def demonstrate_custom_validators():
    """Demonstrate custom validators."""
    print("\n=== Custom Validators ===")

    # Create a valid signup form
    form = SignupForm(
        username="johndoe",
        password="password123",
        password_confirm="password123"
    )
    print(f"Valid signup form: {form}")

    # Try creating forms with invalid fields
    try:
        # Non-alphanumeric username
        form = SignupForm(
            username="john.doe",
            password="password123",
            password_confirm="password123"
        )
    except Exception as e:
        print(f"Validation error (non-alphanumeric username): {e}")

    try:
        # Password mismatch
        form = SignupForm(
            username="johndoe",
            password="password123",
            password_confirm="different"
        )
    except Exception as e:
        print(f"Validation error (password mismatch): {e}")


def demonstrate_task_creation():
    """Demonstrate task creation with validation."""
    print("\n=== Task Creation ===")

    # Create a valid task
    future_date = datetime.now() + timedelta(days=7)
    valid_task = {
        "title": "Complete project",
        "description": "Finish the Pydantic module",
        "priority": "high",
        "due_date": future_date.isoformat(),
        "tags": ["work", "important"]
    }

    result = process_task_creation(valid_task)
    print(f"Valid task result: {result}")

    # Create invalid tasks
    invalid_task1 = {
        "title": "A",  # Too short
        "priority": "medium"
    }

    result = process_task_creation(invalid_task1)
    print(f"Invalid task result (title too short): {result}")

    invalid_task2 = {
        "title": "Valid Title",
        "priority": "critical"  # Not in enum
    }

    result = process_task_creation(invalid_task2)
    print(f"Invalid task result (invalid priority): {result}")

    # Past due date
    past_date = datetime.now() - timedelta(days=7)
    invalid_task3 = {
        "title": "Valid Title",
        "due_date": past_date.isoformat()
    }

    result = process_task_creation(invalid_task3)
    print(f"Invalid task result (past due date): {result}")


if __name__ == "__main__":
    print("=== Pydantic Basics Demonstration ===")

    demonstrate_basic_models()
    demonstrate_field_constraints()
    demonstrate_custom_validators()
    demonstrate_task_creation()

    print("\nDemonstration complete!")
