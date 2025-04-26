# 🚀 Module 3: Data Validation with Pydantic - Lesson 2: Schema Design & Evolution 📋

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand advanced schema design principles for maintainable data models
- 🧩 Implement complex data structures using nested models and generics
- 🔄 Apply strategies for evolving schemas over time
- 📊 Generate and utilize JSON Schema for documentation and validation
- 🛠️ Build a versioned command system for agent interactions

---

## 📚 Introduction to Schema Design & Evolution

<img src="https://github.com/user-attachments/assets/cb3f2aa6-3859-4007-ac07-5cbc2d93e895" width="50%" height="50%"/>

In this lesson, we'll explore advanced schema design patterns using Pydantic and strategies for evolving schemas over time. Well-designed schemas are crucial for building maintainable agent systems that can adapt to changing requirements.

As agent systems grow in complexity, their data models need to evolve while maintaining compatibility with existing data and integrations. Proper schema design and evolution strategies help manage this complexity and ensure your system remains robust over time.

### Key Concepts

#### Schema Design Principles

Good schema design follows several key principles:

1. **Single Responsibility Principle**: Each model should represent a coherent concept
2. **Composition Over Inheritance**: Prefer composing models from smaller components
3. **Progressive Disclosure**: Design schemas to reveal complexity progressively
4. **Explicit Versioning**: Make schema versions explicit when they change

#### Schema Evolution Strategies

When schemas need to change over time, several strategies can help:

1. **Versioning**: Explicitly version your schemas
2. **Optional Fields**: Add new fields as optional for backward compatibility
3. **Deprecation Patterns**: Mark fields as deprecated before removing them
4. **Migration Utilities**: Create utilities to migrate between schema versions

## 🧩 Schema Design Patterns

### Single Responsibility Principle

Each model should have a single responsibility and represent a coherent concept:

```python
# Bad: Mixing concerns
class UserWithPosts(BaseModel):
    user_id: int
    username: str
    email: str
    posts: List[dict]  # Posts mixed with user data

# Good: Separate models for separate concerns
class User(BaseModel):
    id: int
    username: str
    email: str

class Post(BaseModel):
    id: int
    user_id: int
    title: str
    content: str

# Relationship model if needed
class UserWithPosts(BaseModel):
    user: User
    posts: List[Post]
```

### Composition Over Inheritance

Prefer composing models from smaller components rather than deep inheritance hierarchies:

```python
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class User(BaseModel):
    id: int
    username: str
    contact_info: ContactInfo
```

### Progressive Disclosure

Design schemas to reveal complexity progressively:

```python
# Basic user model for simple operations
class UserBasic(BaseModel):
    id: int
    username: str

# Extended user model with more details
class UserDetailed(UserBasic):
    email: str
    full_name: str
    created_at: datetime

# Complete user model with all information
class UserComplete(UserDetailed):
    contact_info: ContactInfo
    preferences: dict
    security_settings: dict
```

## 🔄 Advanced Model Patterns

### Nested Models

Pydantic supports nested models for complex data structures:

```python
class Image(BaseModel):
    url: str
    width: int
    height: int

class Author(BaseModel):
    name: str
    bio: Optional[str] = None
    avatar: Optional[Image] = None

class Article(BaseModel):
    title: str
    content: str
    author: Author
    cover_image: Optional[Image] = None
    tags: List[str] = []
```

### Generic Models

Use generics for reusable model patterns:

```python
from typing import Generic, TypeVar, List

T = TypeVar('T')

class Paginated(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size

# Usage
class User(BaseModel):
    id: int
    name: str

# A paginated list of users
user_page = Paginated[User](
    items=[User(id=1, name="Alice"), User(id=2, name="Bob")],
    total=10,
    page=1,
    page_size=2
)
```

### Discriminated Unions

For better type safety with unions, use discriminated unions:

```python
from typing import Literal, Union

class Dog(BaseModel):
    type: Literal["dog"] = "dog"
    name: str
    breed: str

class Cat(BaseModel):
    type: Literal["cat"] = "cat"
    name: str
    lives_left: int

class Parrot(BaseModel):
    type: Literal["parrot"] = "parrot"
    name: str
    can_speak: bool

Pet = Union[Dog, Cat, Parrot]

# Usage
def process_pet(pet_data: dict):
    pet = Pet(**pet_data)
    if isinstance(pet, Dog):
        print(f"Dog: {pet.name}, breed: {pet.breed}")
    elif isinstance(pet, Cat):
        print(f"Cat: {pet.name}, lives left: {pet.lives_left}")
    elif isinstance(pet, Parrot):
        print(f"Parrot: {pet.name}, can speak: {pet.can_speak}")
```

## 📊 Schema Evolution Strategies

### Versioning

Explicitly version your schemas to manage changes:

```python
class UserV1(BaseModel):
    id: int
    name: str
    email: str

class UserV2(BaseModel):
    id: int
    first_name: str  # Split name into first_name and last_name
    last_name: str
    email: str

    # Migration function from V1
    @classmethod
    def from_v1(cls, user_v1: UserV1):
        name_parts = user_v1.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        return cls(
            id=user_v1.id,
            first_name=first_name,
            last_name=last_name,
            email=user_v1.email
        )
```

### Optional Fields for Backward Compatibility

Add new fields as optional to maintain compatibility:

```python
# Original schema
class Product(BaseModel):
    id: int
    name: str
    price: float

# Updated schema with backward compatibility
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None  # New field, but optional
    category: Optional[str] = None  # New field, but optional
```

### Deprecation Patterns

Mark fields as deprecated before removing them:

```python
from pydantic import Field, BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    username: str

    # Deprecated field
    email: Optional[str] = Field(
        None,
        deprecated=True,
        description="Deprecated: Use contact_info.email instead"
    )

    # New field structure
    contact_info: Optional[dict] = None
```

## 🛠️ Putting It All Together: Agent Command Schema

Let's design a schema for agent commands that can evolve over time:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal
from datetime import datetime

# Base command structure
class Command(BaseModel):
    command_type: str
    timestamp: datetime = Field(default_factory=datetime.now)

# V1 Commands
class CreateTaskV1(Command):
    command_type: Literal["create_task"] = "create_task"
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None

class DeleteTaskV1(Command):
    command_type: Literal["delete_task"] = "delete_task"
    task_id: int

# V2 Commands with enhanced features
class CreateTaskV2(Command):
    command_type: Literal["create_task"] = "create_task"
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"  # New field
    tags: List[str] = []  # New field

class DeleteTaskV2(Command):
    command_type: Literal["delete_task"] = "delete_task"
    task_id: int
    soft_delete: bool = False  # New field

# Command registry
class CommandRegistry:
    def __init__(self):
        self.command_types = {}

    def register(self, version: int, command_type: str, model_class):
        key = f"v{version}_{command_type}"
        self.command_types[key] = model_class

    def get_command_class(self, version: int, command_type: str):
        key = f"v{version}_{command_type}"
        return self.command_types.get(key)

    def parse_command(self, version: int, data: dict):
        command_type = data.get("command_type")
        if not command_type:
            raise ValueError("Missing command_type in data")

        command_class = self.get_command_class(version, command_type)
        if not command_class:
            raise ValueError(f"Unknown command type: {command_type} for version {version}")

        return command_class(**data)

# Usage
registry = CommandRegistry()
registry.register(1, "create_task", CreateTaskV1)
registry.register(1, "delete_task", DeleteTaskV1)
registry.register(2, "create_task", CreateTaskV2)
registry.register(2, "delete_task", DeleteTaskV2)

# Parse a v1 command
v1_data = {
    "command_type": "create_task",
    "title": "Complete project",
    "description": "Finish the project by Friday"
}

v1_command = registry.parse_command(1, v1_data)
print(v1_command)

# Parse a v2 command
v2_data = {
    "command_type": "create_task",
    "title": "Complete project",
    "description": "Finish the project by Friday",
    "priority": "high",
    "tags": ["work", "urgent"]
}

v2_command = registry.parse_command(2, v2_data)
print(v2_command)
```

---

## 💪 Practice Exercises

1. **Design a Blog Post Schema**:
   - Create a schema for a blog post with nested models for author information, comments, and metadata
   - Include validation rules for each component
   - Implement at least one discriminated union (e.g., for different content types)

2. **Create a Versioned User Profile**:
   - Design a UserProfileV1 with basic fields
   - Create a UserProfileV2 with additional fields and modified structure
   - Implement migration functions between versions
   - Test with sample data

3. **Build a Schema Registry**:
   - Implement a registry system that can store and retrieve different schema versions
   - Add migration functions between versions
   - Create a function that can automatically upgrade data from any version to the latest

4. **Generate and Analyze JSON Schema**:
   - Generate JSON Schema for your models
   - Analyze the output and identify how different Pydantic features are represented
   - Create documentation based on the generated schema

---

## 🔍 Key Concepts to Remember

1. **Single Responsibility**: Each model should represent a coherent concept
2. **Composition**: Prefer composing models from smaller components over deep inheritance
3. **Versioning**: Explicitly version your schemas when they change significantly
4. **Backward Compatibility**: Use optional fields and deprecation patterns for smooth transitions
5. **Migration Utilities**: Create tools to convert between schema versions

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Structured output parsing techniques for LLMs
- Extracting structured data from natural language responses
- Handling parsing errors and edge cases
- Building robust parsers for different output formats
- Integrating parsers with agent systems

---

## 📚 Resources

- [Pydantic Models Documentation](https://docs.pydantic.dev/latest/usage/models/)
- [JSON Schema Specification](https://json-schema.org/)
- [API Evolution Best Practices](https://www.mnot.net/blog/2012/12/04/api-evolution)
- [Type Hints with Generics](https://mypy.readthedocs.io/en/stable/generics.html)

---

## 🎯 Mini-Project Progress: Data Validation System

In this lesson, we've made progress on our data validation system by:
- Learning how to design complex, nested data models
- Implementing versioning strategies for evolving schemas
- Creating a command registry system for handling different command versions
- Building migration utilities between schema versions

In the next lesson, we'll continue by:
- Adding structured output parsing capabilities
- Integrating our validation system with LLM outputs
- Handling parsing errors and edge cases

---

> 💡 **Note on LLM Integration**: This lesson focuses on schema design and evolution, which applies to both simulated and real LLM-based systems. The command registry pattern is particularly useful when working with LLMs, as it allows for robust handling of different command formats that might be generated.

---

Happy coding! 🚀
