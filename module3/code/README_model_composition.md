# Advanced Model Composition with Pydantic

This directory contains implementations of advanced model composition patterns using Pydantic, as covered in Lesson 4.3.

## Files

- `model_composition.py`: Main implementation of various model composition patterns
- `test_model_composition.py`: Tests for the model composition implementations

## Key Concepts

### Inheritance Patterns

- Basic inheritance from parent models
- Multi-level inheritance hierarchies
- Mixin classes for reusable functionality
- Abstract base models for defining interfaces

### Composition Patterns

- Nested models for complex data structures
- Reusing field definitions across models
- Model factories for flexible composition
- Dynamic model generation at runtime

### Transformation Patterns

- Converting between related models (API, DB, etc.)
- Adapter patterns for model transformation
- Form system with dynamic model generation

## Usage Examples

### Basic Inheritance

```python
from model_composition import BaseItem, Product, User

# Create a product
product = Product(id=1, name="Laptop", price=999.99, description="Powerful laptop")
print(product.model_dump_json(indent=2))

# Create a user
user = User(id=2, username="johndoe", email="john@example.com")
print(user.model_dump_json(indent=2))
```

### Dynamic Model Generation

```python
from model_composition import create_dynamic_model
from typing import List

# Define fields configuration
user_fields = {
    "name": {"type": str, "min_length": 2},
    "email": {"type": str, "pattern": r"[^@]+@[^@]+\.[^@]+"},
    "age": {"type": int, "ge": 18, "optional": True},
    "tags": {"type": List[str], "optional": True, "default": []}
}

# Create dynamic model
UserModel = create_dynamic_model("User", user_fields)

# Use the model
user = UserModel(name="John Doe", email="john@example.com", age=30)
print(user.model_dump_json(indent=2))
```

### Form System

```python
from model_composition import FormDefinition, StringField, BooleanField

# Create form definition
form_def = FormDefinition(
    title="Contact Form",
    fields=[
        StringField(
            name="name",
            label="Full Name",
            required=True,
            min_length=2,
            max_length=100,
            help_text="Your full name"
        ),
        StringField(
            name="email",
            label="Email Address",
            required=True,
            pattern=r"[^@]+@[^@]+\.[^@]+",
            help_text="Your email address"
        ),
        BooleanField(
            name="subscribe",
            label="Subscribe to newsletter",
            required=False,
            default=False
        )
    ]
)

# Generate model
ContactForm = form_def.create_model()

# Use the model
form_data = ContactForm(
    name="John Doe",
    email="john@example.com",
    subscribe=True
)
print(form_data.model_dump_json(indent=2))
```

## Running Tests

To run the tests for the model composition implementations:

```bash
python -m unittest test_model_composition.py
```

## Related Exercises

For practical exercises related to model composition, see the exercises directory:

1. `lesson4_3_1_exercises.py`: User model hierarchy with inheritance
2. `lesson4_3_2_exercises.py`: Mixin for tracking model changes
3. `lesson4_3_3_exercises.py`: Dynamic model generator for database schemas
4. `lesson4_3_4_exercises.py`: Adapter system for model conversion
5. `lesson4_3_5_exercises.py`: Form builder with HTML generation
