"""
Advanced Pydantic Features - Schema Design & Evolution
---------------------------------------------------
This module demonstrates advanced Pydantic features for schema design and evolution.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Generic, TypeVar, Union, Literal
from datetime import datetime
from enum import Enum


# 1. Single Responsibility Principle
# ----------------------------------

class User(BaseModel):
    """User model with basic information."""
    id: int
    username: str
    email: str


class Post(BaseModel):
    """Post model with content information."""
    id: int
    user_id: int
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)


class UserWithPosts(BaseModel):
    """Relationship model combining user and posts."""
    user: User
    posts: List[Post]


# 2. Composition Over Inheritance
# ------------------------------

class Address(BaseModel):
    """Address component for reuse."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str


class ContactInfo(BaseModel):
    """Contact information component for reuse."""
    email: str
    phone: Optional[str] = None
    address: Address


class UserWithContact(BaseModel):
    """User model composed with contact information."""
    id: int
    username: str
    contact_info: ContactInfo


# 3. Progressive Disclosure
# ------------------------

class UserBasic(BaseModel):
    """Basic user model for simple operations."""
    id: int
    username: str


class UserDetailed(UserBasic):
    """Extended user model with more details."""
    email: str
    full_name: str
    created_at: datetime


class UserComplete(UserDetailed):
    """Complete user model with all information."""
    contact_info: ContactInfo
    preferences: Dict[str, Any]
    security_settings: Dict[str, Any]


# 4. Nested Models
# ---------------

class Image(BaseModel):
    """Image model for reuse in other models."""
    url: str
    width: int
    height: int


class Author(BaseModel):
    """Author model with nested image for avatar."""
    name: str
    bio: Optional[str] = None
    avatar: Optional[Image] = None


class Article(BaseModel):
    """Article model with nested author and image."""
    title: str
    content: str
    author: Author
    cover_image: Optional[Image] = None
    tags: List[str] = []


# 5. Generic Models
# ---------------

T = TypeVar('T')


class Paginated(BaseModel, Generic[T]):
    """Generic pagination model that can work with any type."""
    items: List[T]
    total: int
    page: int
    page_size: int
    
    @property
    def total_pages(self) -> int:
        """Calculate total pages based on total items and page size."""
        return (self.total + self.page_size - 1) // self.page_size
    
    @property
    def has_next(self) -> bool:
        """Check if there is a next page."""
        return self.page < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        """Check if there is a previous page."""
        return self.page > 1


# 6. Union Types
# -------------

class TextContent(BaseModel):
    """Text content type for messages."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content type for messages."""
    type: Literal["image"] = "image"
    url: str
    caption: Optional[str] = None


class VideoContent(BaseModel):
    """Video content type for messages."""
    type: Literal["video"] = "video"
    url: str
    duration: int  # seconds
    thumbnail: Optional[str] = None


class Message(BaseModel):
    """Message model with different content types."""
    id: int
    sender: str
    content: Union[TextContent, ImageContent, VideoContent]
    timestamp: datetime = Field(default_factory=datetime.now)


# 7. Discriminated Unions
# ---------------------

class Dog(BaseModel):
    """Dog model with discriminator field."""
    type: Literal["dog"] = "dog"
    name: str
    breed: str


class Cat(BaseModel):
    """Cat model with discriminator field."""
    type: Literal["cat"] = "cat"
    name: str
    lives_left: int


class Parrot(BaseModel):
    """Parrot model with discriminator field."""
    type: Literal["parrot"] = "parrot"
    name: str
    can_speak: bool


Pet = Union[Dog, Cat, Parrot]


def process_pet(pet_data: dict) -> str:
    """
    Process a pet based on its type.
    
    Args:
        pet_data: Dictionary containing pet data
        
    Returns:
        String with pet information
    """
    # Convert the dictionary to the appropriate Pet type
    pet = Pet.model_validate(pet_data)
    
    # Process based on the specific type
    if isinstance(pet, Dog):
        return f"Dog: {pet.name}, breed: {pet.breed}"
    elif isinstance(pet, Cat):
        return f"Cat: {pet.name}, lives left: {pet.lives_left}"
    elif isinstance(pet, Parrot):
        return f"Parrot: {pet.name}, can speak: {pet.can_speak}"
    
    # This should never happen if the Union type is correct
    return "Unknown pet type"


# 8. Schema Evolution Strategies
# ----------------------------

# Versioning
class UserV1(BaseModel):
    """Version 1 of the user schema."""
    id: int
    name: str
    email: str


class UserV2(BaseModel):
    """Version 2 of the user schema with split name."""
    id: int
    first_name: str
    last_name: str
    email: str
    
    @classmethod
    def from_v1(cls, user_v1: UserV1) -> 'UserV2':
        """
        Migrate from UserV1 to UserV2.
        
        Args:
            user_v1: UserV1 instance
            
        Returns:
            UserV2 instance
        """
        name_parts = user_v1.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return cls(
            id=user_v1.id,
            first_name=first_name,
            last_name=last_name,
            email=user_v1.email
        )


# Optional Fields for Backward Compatibility
class ProductV1(BaseModel):
    """Original product schema."""
    id: int
    name: str
    price: float


class ProductV2(BaseModel):
    """Updated product schema with backward compatibility."""
    id: int
    name: str
    price: float
    description: Optional[str] = None  # New field, but optional
    category: Optional[str] = None  # New field, but optional


# Deprecation Patterns
class UserWithDeprecation(BaseModel):
    """User model with deprecated fields."""
    id: int
    username: str
    
    # Deprecated field
    email: Optional[str] = Field(
        None,
        deprecated=True,
        description="Deprecated: Use contact_info.email instead"
    )
    
    # New field structure
    contact_info: Optional[Dict[str, Any]] = None


# 9. Migration Utilities
# --------------------

class SchemaRegistry:
    """Registry for schema versions and migrations."""
    
    def __init__(self):
        self.schemas = {}
        self.migrations = {}
    
    def register_schema(self, name: str, version: int, schema_class):
        """
        Register a schema version.
        
        Args:
            name: Schema name
            version: Schema version
            schema_class: Schema class
        """
        key = f"{name}_v{version}"
        self.schemas[key] = schema_class
    
    def register_migration(self, name: str, from_version: int, to_version: int, migration_func):
        """
        Register a migration between schema versions.
        
        Args:
            name: Schema name
            from_version: Source version
            to_version: Target version
            migration_func: Migration function
        """
        key = f"{name}_v{from_version}_to_v{to_version}"
        self.migrations[key] = migration_func
    
    def get_schema(self, name: str, version: int):
        """
        Get a schema by name and version.
        
        Args:
            name: Schema name
            version: Schema version
            
        Returns:
            Schema class
        """
        key = f"{name}_v{version}"
        return self.schemas.get(key)
    
    def migrate(self, data, name: str, from_version: int, to_version: int):
        """
        Migrate data from one schema version to another.
        
        Args:
            data: Data to migrate
            name: Schema name
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migrated data
        """
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


# 10. JSON Schema Generation
# ------------------------

def generate_json_schema(model_class):
    """
    Generate JSON schema for a model class.
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        JSON schema as a dictionary
    """
    return model_class.model_json_schema()


# 11. Practical Example: Agent Command Schema
# -----------------------------------------

class Command(BaseModel):
    """Base command structure."""
    command_type: str
    timestamp: datetime = Field(default_factory=datetime.now)


class CreateTaskV1(Command):
    """V1 Create Task command."""
    command_type: Literal["create_task"] = "create_task"
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None


class DeleteTaskV1(Command):
    """V1 Delete Task command."""
    command_type: Literal["delete_task"] = "delete_task"
    task_id: int


class CreateTaskV2(Command):
    """V2 Create Task command with enhanced features."""
    command_type: Literal["create_task"] = "create_task"
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"  # New field
    tags: List[str] = []  # New field


class DeleteTaskV2(Command):
    """V2 Delete Task command with enhanced features."""
    command_type: Literal["delete_task"] = "delete_task"
    task_id: int
    soft_delete: bool = False  # New field


class CommandRegistry:
    """Registry for command types and versions."""
    
    def __init__(self):
        self.command_types = {}
    
    def register(self, version: int, command_type: str, model_class):
        """
        Register a command type for a specific version.
        
        Args:
            version: Command version
            command_type: Command type
            model_class: Command model class
        """
        key = f"v{version}_{command_type}"
        self.command_types[key] = model_class
    
    def get_command_class(self, version: int, command_type: str):
        """
        Get a command class by version and type.
        
        Args:
            version: Command version
            command_type: Command type
            
        Returns:
            Command model class
        """
        key = f"v{version}_{command_type}"
        return self.command_types.get(key)
    
    def parse_command(self, version: int, data: dict):
        """
        Parse a command from data.
        
        Args:
            version: Command version
            data: Command data
            
        Returns:
            Command instance
        """
        command_type = data.get("command_type")
        if not command_type:
            raise ValueError("Missing command_type in data")
            
        command_class = self.get_command_class(version, command_type)
        if not command_class:
            raise ValueError(f"Unknown command type: {command_type} for version {version}")
            
        return command_class(**data)


def demonstrate_command_registry():
    """Demonstrate the command registry."""
    # Create registry
    registry = CommandRegistry()
    
    # Register command types
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
    print(f"V1 Command: {v1_command}")
    
    # Parse a v2 command
    v2_data = {
        "command_type": "create_task",
        "title": "Complete project",
        "description": "Finish the project by Friday",
        "priority": "high",
        "tags": ["work", "urgent"]
    }
    
    v2_command = registry.parse_command(2, v2_data)
    print(f"V2 Command: {v2_command}")


if __name__ == "__main__":
    # Demonstrate generic models
    user_page = Paginated[User](
        items=[User(id=1, username="Alice", email="alice@example.com"), 
               User(id=2, username="Bob", email="bob@example.com")],
        total=10,
        page=1,
        page_size=2
    )
    print(f"User page: {user_page}")
    print(f"Total pages: {user_page.total_pages}")
    print(f"Has next page: {user_page.has_next}")
    
    # Demonstrate discriminated unions
    dog_data = {"type": "dog", "name": "Rex", "breed": "German Shepherd"}
    cat_data = {"type": "cat", "name": "Whiskers", "lives_left": 9}
    parrot_data = {"type": "parrot", "name": "Polly", "can_speak": True}
    
    print(process_pet(dog_data))
    print(process_pet(cat_data))
    print(process_pet(parrot_data))
    
    # Demonstrate schema evolution
    user_v1 = UserV1(id=1, name="John Doe", email="john@example.com")
    user_v2 = UserV2.from_v1(user_v1)
    print(f"UserV1: {user_v1}")
    print(f"UserV2 (migrated): {user_v2}")
    
    # Demonstrate JSON schema generation
    user_schema = generate_json_schema(User)
    print(f"User JSON Schema: {user_schema}")
    
    # Demonstrate command registry
    demonstrate_command_registry()
