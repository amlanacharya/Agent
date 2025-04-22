# Lesson 1: Pydantic Fundamentals 🔒

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

## 📋 Overview

Pydantic is a powerful data validation and settings management library that uses Python type annotations to enforce type safety at runtime. In this lesson, we'll explore the core concepts of Pydantic and how it can be used to create robust, type-safe data models for agent systems.

## 🧠 Why Pydantic for Agent Systems?

When building AI agents, we often need to:

1. **Validate user inputs** before processing them
2. **Structure LLM outputs** in a predictable format
3. **Enforce data consistency** across the agent's components
4. **Document data schemas** for API integration
5. **Handle errors gracefully** when data doesn't match expectations

Pydantic excels at all these tasks, making it an essential tool for building reliable agent systems.

![Pydantic Validation Flow](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPrc2ngFZ6BTyww/giphy.gif)

## 🔑 Key Concepts

### 1. Pydantic Models

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
```

### 2. Data Validation

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

### 3. Type Coercion

Pydantic attempts to convert input values to the declared types when possible:

```python
# Pydantic will convert "42" to integer 42
user = User(id="42", name="Jane Doe", email="jane@example.com")
print(user.id)  # Output: 42 (as an integer, not a string)
```

### 4. Field Constraints

You can add constraints to fields using Pydantic's `Field` function:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=3, max_length=50)
    price: float = Field(..., gt=0)
    description: Optional[str] = Field(None, max_length=1000)
```

### 5. Model Methods

Pydantic models come with useful methods for data manipulation:

```python
# Convert to dictionary
user_dict = user.model_dump()

# Convert to JSON
user_json = user.model_dump_json()

# Create a copy with updated fields
updated_user = user.model_copy(update={"name": "New Name"})
```

### 6. Config Options

Customize model behavior using the `Config` class:

```python
class User(BaseModel):
    id: int
    name: str
    email: str
    
    class Config:
        # Allow extra fields that aren't defined in the model
        extra = "ignore"
        
        # Make all fields optional
        # extra = "allow"
        
        # Validate field assignments
        validate_assignment = True
        
        # Case-insensitive field names
        case_sensitive = False
```

## 🛠️ Basic Validation Patterns

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

### Field Types and Constraints

```python
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime

class AdvancedUser(BaseModel):
    id: int = Field(..., gt=0)
    username: str = Field(..., min_length=3, max_length=20)
    email: EmailStr
    password: str = Field(..., min_length=8)
    created_at: datetime = Field(default_factory=datetime.now)
    roles: List[str] = []
    settings: Optional[dict] = None
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

### Model Inheritance

Pydantic supports model inheritance for code reuse:

```python
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

## 🔄 Practical Example: Agent Input Validation

Let's see how Pydantic can be used to validate user inputs for an agent:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum

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

## 💾 Serialization and Deserialization

Pydantic makes it easy to convert between different data formats:

### JSON Serialization

```python
# Convert model to JSON
user_json = user.model_dump_json()

# Parse JSON into model
user_data = '{"id": 1, "name": "John", "email": "john@example.com"}'
parsed_user = User.model_validate_json(user_data)
```

### Dictionary Conversion

```python
# Convert model to dictionary
user_dict = user.model_dump()

# Create model from dictionary
user_from_dict = User.model_validate(user_dict)
```

## 🧪 Exercises

1. Create a `UserProfile` model with fields for name, email, age, and bio. Add appropriate validation rules.

2. Extend the model to include a list of skills, where each skill has a name and proficiency level.

3. Add a custom validator that ensures the email field contains an @ symbol.

4. Create a function that takes a dictionary of user data, validates it against your model, and handles any validation errors gracefully.

## 🔍 Key Takeaways

- Pydantic provides runtime validation of data using Python type annotations
- Models define the structure and constraints of your data
- Validation errors are raised when data doesn't match the expected format
- Custom validators allow for complex validation logic
- Pydantic models can be easily converted to/from JSON and dictionaries

## 📚 Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Pydantic Field Types Reference](https://docs.pydantic.dev/latest/api/fields/)

## 🚀 Next Steps

In the next lesson, we'll explore more advanced Pydantic features, including schema design patterns, model composition, and techniques for handling evolving schemas.
