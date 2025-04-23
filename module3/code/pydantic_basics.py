"""
Pydantic Basics - Core concepts and examples
-------------------------------------------
This module demonstrates the fundamental concepts of Pydantic for data validation.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


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


class UserWithConfig(BaseModel):
    """User model demonstrating config options."""
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


class UserProfile(BaseModel):
    """User profile demonstrating required vs optional fields."""
    # Required fields (no default value)
    username: str
    email: str
    
    # Optional fields (with default values)
    bio: Optional[str] = None
    age: Optional[int] = None
    is_active: bool = True


class AdvancedUser(BaseModel):
    """Advanced user model with field types and constraints."""
    id: int = Field(..., gt=0)
    username: str = Field(..., min_length=3, max_length=20)
    email: EmailStr
    password: str = Field(..., min_length=8)
    created_at: datetime = Field(default_factory=datetime.now)
    roles: List[str] = []
    settings: Optional[Dict[str, Any]] = None


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


class BaseItem(BaseModel):
    """Base item model for inheritance demonstration."""
    id: int
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class ProductItem(BaseItem):
    """Product model inheriting from BaseItem."""
    name: str
    price: float
    category: str


class ServiceItem(BaseItem):
    """Service model inheriting from BaseItem."""
    name: str
    hourly_rate: float
    description: str


class TaskPriority(str, Enum):
    """Enum for task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskInput(BaseModel):
    """Task input model for agent input validation example."""
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


def process_task_creation(user_input: dict):
    """
    Process task creation with validation.
    
    Args:
        user_input: Dictionary containing task data
        
    Returns:
        Dictionary with status and task data or error message
    """
    try:
        # Validate input against our model
        task = TaskInput(**user_input)
        
        # If validation passes, proceed with task creation
        return {"status": "success", "task": task.model_dump()}
    except Exception as e:
        # Handle validation errors
        return {"status": "error", "message": str(e)}


def demonstrate_serialization():
    """Demonstrate serialization and deserialization."""
    # Create a user
    user = User(id=1, name="John Doe", email="john@example.com")
    
    # Convert to JSON
    user_json = user.model_dump_json()
    print(f"JSON: {user_json}")
    
    # Parse from JSON
    user_data = '{"id": 2, "name": "Jane Doe", "email": "jane@example.com"}'
    parsed_user = User.model_validate_json(user_data)
    print(f"Parsed from JSON: {parsed_user}")
    
    # Convert to dictionary
    user_dict = user.model_dump()
    print(f"Dict: {user_dict}")
    
    # Create from dictionary
    user_from_dict = User.model_validate({"id": 3, "name": "Bob", "email": "bob@example.com"})
    print(f"From dict: {user_from_dict}")


if __name__ == "__main__":
    # Demonstrate basic validation
    try:
        # Valid user
        user = User(id=1, name="John Doe", email="john@example.com")
        print(f"Valid user: {user}")
        
        # Invalid user
        invalid_user = User(id="not_an_int", name=123, email="invalid_email")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Demonstrate type coercion
    user = User(id="42", name="Jane Doe", email="jane@example.com")
    print(f"User with coerced ID: {user.id} (type: {type(user.id)})")
    
    # Demonstrate serialization
    demonstrate_serialization()
    
    # Demonstrate task creation
    valid_task = {
        "title": "Complete project",
        "description": "Finish the Pydantic module",
        "priority": "high",
        "due_date": (datetime.now().replace(microsecond=0) + 
                    datetime.timedelta(days=7)).isoformat(),
        "tags": ["work", "important"]
    }
    
    result = process_task_creation(valid_task)
    print(f"Task creation result: {result}")
    
    # Invalid task
    invalid_task = {
        "title": "A",  # Too short
        "priority": "critical"  # Not in enum
    }
    
    result = process_task_creation(invalid_task)
    print(f"Invalid task result: {result}")
