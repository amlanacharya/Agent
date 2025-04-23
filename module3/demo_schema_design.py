"""
Standalone Demonstration Script for Schema Design & Evolution
---------------------------------------------------------
This script demonstrates advanced Pydantic features for schema design and evolution.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
import json


# 1. Nested Models
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


# 2. Discriminated Unions
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
    """Process a pet based on its type."""
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


# 3. Schema Evolution
# -----------------

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
        """Migrate from UserV1 to UserV2."""
        name_parts = user_v1.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return cls(
            id=user_v1.id,
            first_name=first_name,
            last_name=last_name,
            email=user_v1.email
        )


class UserV3(BaseModel):
    """Version 3 of the user schema with additional fields."""
    id: int
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    is_active: bool = True
    
    @classmethod
    def from_v2(cls, user_v2: UserV2) -> 'UserV3':
        """Migrate from UserV2 to UserV3."""
        return cls(
            id=user_v2.id,
            first_name=user_v2.first_name,
            last_name=user_v2.last_name,
            email=user_v2.email
        )


# 4. Command Registry
# -----------------

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


class CommandRegistry:
    """Registry for command types and versions."""
    
    def __init__(self):
        self.command_types = {}
    
    def register(self, version: int, command_type: str, model_class):
        """Register a command type for a specific version."""
        key = f"v{version}_{command_type}"
        self.command_types[key] = model_class
    
    def get_command_class(self, version: int, command_type: str):
        """Get a command class by version and type."""
        key = f"v{version}_{command_type}"
        return self.command_types.get(key)
    
    def parse_command(self, version: int, data: dict):
        """Parse a command from data."""
        command_type = data.get("command_type")
        if not command_type:
            raise ValueError("Missing command_type in data")
            
        command_class = self.get_command_class(version, command_type)
        if not command_class:
            raise ValueError(f"Unknown command type: {command_type} for version {version}")
            
        return command_class(**data)


def main():
    """Main demonstration function."""
    print("=== Schema Design & Evolution Demonstration ===")
    
    # 1. Nested Models
    print("\n1. Nested Models")
    print("---------------")
    
    # Create image
    avatar = Image(url="https://example.com/avatar.jpg", width=200, height=200)
    cover = Image(url="https://example.com/cover.jpg", width=1200, height=600)
    
    # Create author
    author = Author(
        name="Jane Smith",
        bio="Tech writer and developer",
        avatar=avatar
    )
    
    # Create article
    article = Article(
        title="Advanced Pydantic Features",
        content="This is an article about Pydantic...",
        author=author,
        cover_image=cover,
        tags=["python", "pydantic", "validation"]
    )
    
    print(f"Article: {article.title}")
    print(f"Author: {article.author.name}")
    print(f"Author Bio: {article.author.bio}")
    print(f"Author Avatar: {article.author.avatar.url}")
    print(f"Cover Image: {article.cover_image.url}")
    print(f"Tags: {article.tags}")
    
    # 2. Discriminated Unions
    print("\n2. Discriminated Unions")
    print("---------------------")
    
    # Create pet data
    dog_data = {"type": "dog", "name": "Rex", "breed": "German Shepherd"}
    cat_data = {"type": "cat", "name": "Whiskers", "lives_left": 9}
    parrot_data = {"type": "parrot", "name": "Polly", "can_speak": True}
    
    # Process pets
    print(process_pet(dog_data))
    print(process_pet(cat_data))
    print(process_pet(parrot_data))
    
    # Try invalid pet type
    try:
        invalid_data = {"type": "fish", "name": "Nemo"}
        process_pet(invalid_data)
    except Exception as e:
        print(f"Error processing invalid pet: {e}")
    
    # 3. Schema Evolution
    print("\n3. Schema Evolution")
    print("-----------------")
    
    # Create v1 user
    user_v1 = UserV1(id=1, name="John Doe", email="john@example.com")
    print(f"UserV1: {user_v1}")
    
    # Migrate to v2
    user_v2 = UserV2.from_v1(user_v1)
    print(f"UserV2 (migrated): {user_v2}")
    
    # Migrate to v3
    user_v3 = UserV3.from_v2(user_v2)
    print(f"UserV3 (migrated): {user_v3}")
    
    # 4. Command Registry
    print("\n4. Command Registry")
    print("-----------------")
    
    # Create registry
    registry = CommandRegistry()
    
    # Register command types
    registry.register(1, "create_task", CreateTaskV1)
    registry.register(1, "delete_task", DeleteTaskV1)
    registry.register(2, "create_task", CreateTaskV2)
    
    # Parse v1 command
    v1_data = {
        "command_type": "create_task",
        "title": "Complete project",
        "description": "Finish the project by Friday"
    }
    
    v1_command = registry.parse_command(1, v1_data)
    print(f"V1 Command: {v1_command}")
    
    # Parse v2 command
    v2_data = {
        "command_type": "create_task",
        "title": "Complete project",
        "description": "Finish the project by Friday",
        "priority": "high",
        "tags": ["work", "urgent"]
    }
    
    v2_command = registry.parse_command(2, v2_data)
    print(f"V2 Command: {v2_command}")
    
    # Try unknown command type
    try:
        registry.parse_command(1, {"command_type": "unknown_command"})
    except Exception as e:
        print(f"Error parsing unknown command: {e}")
    
    # 5. JSON Schema Generation
    print("\n5. JSON Schema Generation")
    print("-----------------------")
    
    # Generate JSON schema for Article
    article_schema = Article.model_json_schema()
    print(f"Article JSON Schema (simplified):")
    print(json.dumps(article_schema, indent=2)[:500] + "...\n")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
