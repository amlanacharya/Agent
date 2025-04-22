# Lesson 2: Schema Design & Evolution 📋

<img src="https://github.com/user-attachments/assets/cb3f2aa6-3859-4007-ac07-5cbc2d93e895" width="50%" height="50%"/>

## 📋 Overview

In this lesson, we'll explore advanced schema design patterns using Pydantic and strategies for evolving schemas over time. Well-designed schemas are crucial for building maintainable agent systems that can adapt to changing requirements.

## 🏗️ Schema Design Principles

### 1. Single Responsibility Principle

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

### 2. Composition Over Inheritance

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

### 3. Progressive Disclosure

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

## 🧩 Advanced Model Patterns

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

### Union Types

Handle multiple possible types with Union:

```python
from typing import Union

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: str
    caption: Optional[str] = None

class VideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: str
    duration: int  # seconds
    thumbnail: Optional[str] = None

# A message can contain different types of content
class Message(BaseModel):
    id: int
    sender: str
    content: Union[TextContent, ImageContent, VideoContent]
    timestamp: datetime
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

## 🔄 Schema Evolution Strategies

### 1. Versioning

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

### 2. Optional Fields for Backward Compatibility

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

### 3. Deprecation Patterns

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

### 4. Migration Utilities

Create utilities to migrate between schema versions:

```python
class SchemaRegistry:
    """Registry for schema versions and migrations."""
    
    def __init__(self):
        self.schemas = {}
        self.migrations = {}
    
    def register_schema(self, name: str, version: int, schema_class):
        """Register a schema version."""
        key = f"{name}_v{version}"
        self.schemas[key] = schema_class
    
    def register_migration(self, name: str, from_version: int, to_version: int, migration_func):
        """Register a migration between schema versions."""
        key = f"{name}_v{from_version}_to_v{to_version}"
        self.migrations[key] = migration_func
    
    def get_schema(self, name: str, version: int):
        """Get a schema by name and version."""
        key = f"{name}_v{version}"
        return self.schemas.get(key)
    
    def migrate(self, data, name: str, from_version: int, to_version: int):
        """Migrate data from one schema version to another."""
        if from_version == to_version:
            return data
            
        # Direct migration
        key = f"{name}_v{from_version}_to_v{to_version}"
        if key in self.migrations:
            return self.migrations[key](data)
            
        # Step-by-step migration
        current_version = from_version
        current_data = data
        
        while current_version < to_version:
            next_version = current_version + 1
            key = f"{name}_v{current_version}_to_v{next_version}"
            
            if key not in self.migrations:
                raise ValueError(f"No migration path from v{current_version} to v{next_version}")
                
            current_data = self.migrations[key](current_data)
            current_version = next_version
            
        return current_data
```

## 🔍 JSON Schema Generation

Pydantic can generate JSON Schema from your models, which is useful for documentation and API integration:

```python
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    tags: List[str] = []
    
# Generate JSON Schema
schema = User.model_json_schema()
print(schema)
```

Example output:
```json
{
  "title": "User",
  "type": "object",
  "properties": {
    "id": {
      "title": "Id",
      "type": "integer"
    },
    "name": {
      "title": "Name",
      "type": "string"
    },
    "email": {
      "title": "Email",
      "type": "string"
    },
    "tags": {
      "title": "Tags",
      "default": [],
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "id",
    "name",
    "email"
  ]
}
```

## 🧪 Practical Example: Agent Command Schema

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

## 🧪 Exercises

1. Design a schema for a blog post that includes nested models for author information, comments, and metadata.

2. Create a versioned schema for a user profile that evolves from a simple version to a more complex one with additional fields.

3. Implement a migration function that can convert data from an older schema version to a newer one.

4. Generate JSON Schema for your models and analyze the output.

## 🔍 Key Takeaways

- Well-designed schemas follow principles like single responsibility and composition
- Nested models, generics, and unions enable complex data modeling
- Schema evolution requires strategies like versioning and migration utilities
- JSON Schema generation provides documentation and integration capabilities
- Properly designed schemas make agent systems more maintainable and adaptable

## 📚 Additional Resources

- [Pydantic Models Documentation](https://docs.pydantic.dev/latest/usage/models/)
- [JSON Schema Specification](https://json-schema.org/)
- [API Evolution Best Practices](https://www.mnot.net/blog/2012/12/04/api-evolution)

## 🚀 Next Steps

In the next lesson, we'll explore structured output parsing techniques for LLMs, including how to reliably extract structured data from natural language responses.
