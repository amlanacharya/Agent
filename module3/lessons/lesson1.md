# 🚀 Module 3: Data Validation with Pydantic - Lesson 1: Pydantic Fundamentals 🔒

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand the core concepts of Pydantic and its role in data validation
- 🧩 Create robust, type-safe data models using Pydantic
- 🔄 Implement custom validation logic for complex requirements
- 📊 Convert between different data formats (JSON, dictionaries)
- 🛠️ Apply Pydantic to validate user inputs for agent systems

---

## 📚 Introduction to Pydantic

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

Pydantic is a powerful data validation and settings management library that uses Python type annotations to enforce type safety at runtime. It provides a clean, intuitive way to define data models and validate inputs against those models.

When building AI agents, we often need to:

1. **Validate user inputs** before processing them
2. **Structure LLM outputs** in a predictable format
3. **Enforce data consistency** across the agent's components
4. **Document data schemas** for API integration
5. **Handle errors gracefully** when data doesn't match expectations

Pydantic excels at all these tasks, making it an essential tool for building reliable agent systems.

![Pydantic Validation Flow](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPrc2ngFZ6BTyww/giphy.gif)

### Key Concepts

#### Pydantic Models

The core of Pydantic is the `BaseModel` class. By inheriting from this class, you can create data models with field validation:

```python
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    tags: List[str] = []

# Usage
user = User(id=1, name="John Doe", email="john@example.com")
print(user.model_dump())
```

#### Data Validation

Pydantic automatically validates data against the model's field types:

```python
# Valid data
user = User(id=1, name="John Doe", email="john@example.com")

# Invalid data - will raise ValidationError
try:
    invalid_user = User(id="not_an_int", name=123, email="invalid_email")
except Exception as e:
    print(f"Validation error: {e}")
```

## 🧩 Core Pydantic Features

### Type Coercion

Pydantic attempts to convert input values to the declared types when possible:

```python
# Pydantic will convert "42" to integer 42
user = User(id="42", name="Jane Doe", email="jane@example.com")
print(user.id)  # Output: 42 (as an integer, not a string)
```

### Field Constraints

You can add constraints to fields using Pydantic's `Field` function:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=3, max_length=50)
    price: float = Field(..., gt=0)
    description: Optional[str] = Field(None, max_length=1000)

# Usage
product = Product(id=1, name="Laptop", price=999.99)
print(product.model_dump())
```

### Model Methods

Pydantic models come with useful methods for data manipulation:

```python
# Create a user
user = User(id=1, name="John Doe", email="john@example.com")

# Convert to dictionary
user_dict = user.model_dump()

# Convert to JSON
user_json = user.model_dump_json()

# Create a copy with updated fields
updated_user = user.model_copy(update={"name": "New Name"})
```

## 🔄 Advanced Validation Patterns

### Config Options

Customize model behavior using the `Config` class:

```python
class User(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        # Allow extra fields that aren't defined in the model
        extra = "ignore"

        # Validate field assignments
        validate_assignment = True

        # Case-insensitive field names
        case_sensitive = False
```

### Required vs. Optional Fields

```python
class UserProfile(BaseModel):
    # Required fields (no default value)
    username: str
    email: str

    # Optional fields (with default values)
    bio: Optional[str] = None
    age: Optional[int] = None
    is_active: bool = True
```

### Custom Validators

You can add custom validation logic using the `@field_validator` decorator:

```python
from pydantic import BaseModel, field_validator

class SignupForm(BaseModel):
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
```

## 📊 Model Relationships and Inheritance

### Model Inheritance

Pydantic supports model inheritance for code reuse:

```python
from datetime import datetime

class BaseItem(BaseModel):
    id: int
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class Product(BaseItem):
    name: str
    price: float
    category: str

class Service(BaseItem):
    name: str
    hourly_rate: float
    description: str
```

### Nested Models

Models can contain other models as fields:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class User(BaseModel):
    id: int
    name: str
    email: str
    address: Address
```

## 🛠️ Putting It All Together: Agent Input Validation

Let's see how Pydantic can be used to validate user inputs for an agent:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum
from datetime import datetime

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    tags: List[str] = []

    @field_validator('due_date')
    def due_date_must_be_future(cls, v):
        if v and v < datetime.now():
            raise ValueError('Due date must be in the future')
        return v

# Usage in an agent
def process_task_creation(user_input: dict):
    try:
        # Validate input against our model
        task = TaskInput(**user_input)

        # If validation passes, proceed with task creation
        return {"status": "success", "task": task.model_dump()}
    except Exception as e:
        # Handle validation errors
        return {"status": "error", "message": str(e)}
```

---

## 💪 Practice Exercises

1. **Create a User Profile Model**:
   - Create a `UserProfile` model with fields for name, email, age, and bio
   - Add appropriate validation rules (e.g., email format, age range)
   - Test with both valid and invalid data

2. **Implement Nested Models**:
   - Extend the model to include a list of skills, where each skill has a name and proficiency level
   - Add validation for the proficiency level (e.g., beginner, intermediate, advanced)
   - Create a sample user with multiple skills

3. **Add Custom Validation**:
   - Add a custom validator that ensures the email field contains an @ symbol
   - Implement a validator that checks if the bio is appropriate (e.g., not too short, no profanity)
   - Test your validators with edge cases

4. **Create a Validation Handler**:
   - Create a function that takes a dictionary of user data
   - Validates it against your model
   - Handles any validation errors gracefully
   - Returns appropriate success/error messages

---

## 🔍 Key Concepts to Remember

1. **Type Annotations**: Pydantic uses Python type hints to define the expected data structure
2. **Validation**: Data is automatically validated against the model's field types and constraints
3. **Coercion**: Pydantic attempts to convert input values to the declared types when possible
4. **Custom Validators**: You can add custom validation logic using field validators
5. **Serialization**: Models can be easily converted to/from JSON and dictionaries

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Advanced schema design patterns
- Model composition techniques
- Strategies for handling evolving schemas
- JSON Schema generation and documentation
- More complex validation scenarios

---

## 📚 Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Pydantic Field Types Reference](https://docs.pydantic.dev/latest/api/fields/)

---

## 🎯 Mini-Project Progress: Data Validation System

In this lesson, we've made progress on our data validation system by:
- Understanding the core concepts of Pydantic models
- Learning how to define and validate data structures
- Implementing custom validation logic
- Creating a foundation for handling user inputs

In the next lesson, we'll continue by:
- Expanding our models to handle more complex data structures
- Implementing schema evolution strategies
- Building a more robust validation system

---

> 💡 **Note on LLM Integration**: This lesson focuses on the fundamentals of Pydantic and does not require integration with real LLMs. The concepts can be applied to both simulated and real LLM-based systems.

---

Happy coding! 🚀
