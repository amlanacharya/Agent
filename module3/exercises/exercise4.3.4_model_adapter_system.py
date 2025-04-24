"""
Exercise 4.3.4: Model Adapter System

This exercise implements an adapter system that can convert between
API request models, database models, and API response models.
"""

from pydantic import BaseModel, Field, field_validator, create_model
from typing import Dict, Any, Type, Optional, List, TypeVar, Generic, Callable, Union, get_type_hints
from datetime import datetime
import re
import uuid


# Type variables for generic typing
T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=BaseModel)


class ModelAdapter(Generic[T, U]):
    """
    Generic adapter for converting between related models.
    
    This adapter can convert from a source model type to a target model type,
    with customizable field mapping and transformation functions.
    """
    
    def __init__(
        self,
        source_model: Type[T],
        target_model: Type[U],
        field_mapping: Dict[str, str] = None,
        transformers: Dict[str, Callable[[Any], Any]] = None,
        exclude_fields: List[str] = None,
        include_unmapped: bool = True
    ):
        """
        Initialize the model adapter.
        
        Args:
            source_model: Source model class
            target_model: Target model class
            field_mapping: Mapping from target field names to source field names
            transformers: Functions to transform values for specific target fields
            exclude_fields: Fields to exclude from the target model
            include_unmapped: Whether to include fields that aren't explicitly mapped
        """
        self.source_model = source_model
        self.target_model = target_model
        self.field_mapping = field_mapping or {}
        self.transformers = transformers or {}
        self.exclude_fields = exclude_fields or []
        self.include_unmapped = include_unmapped
    
    def adapt(self, source: T, **extra_fields) -> U:
        """
        Convert source model to target model.
        
        Args:
            source: Source model instance
            **extra_fields: Additional fields to include in the target model
            
        Returns:
            An instance of the target model
        """
        # Get data from source model
        source_data = source.model_dump()
        
        # Prepare target data
        target_data = {}
        
        # Apply field mapping
        for target_field, source_field in self.field_mapping.items():
            if source_field in source_data and target_field not in self.exclude_fields:
                value = source_data[source_field]
                
                # Apply transformer if available
                if target_field in self.transformers:
                    value = self.transformers[target_field](value)
                
                target_data[target_field] = value
        
        # Add unmapped fields that exist in both models
        if self.include_unmapped:
            for field in source_data:
                if (field not in self.field_mapping.values() and 
                    field in self.target_model.model_fields and
                    field not in self.exclude_fields):
                    
                    value = source_data[field]
                    
                    # Apply transformer if available
                    if field in self.transformers:
                        value = self.transformers[field](value)
                    
                    target_data[field] = value
        
        # Add extra fields
        for field, value in extra_fields.items():
            if field not in self.exclude_fields:
                target_data[field] = value
        
        # Create target model
        return self.target_model(**target_data)
    
    def adapt_many(self, sources: List[T], **extra_fields) -> List[U]:
        """
        Convert multiple source models to target models.
        
        Args:
            sources: List of source model instances
            **extra_fields: Additional fields to include in all target models
            
        Returns:
            List of target model instances
        """
        return [self.adapt(source, **extra_fields) for source in sources]


class AdapterRegistry:
    """
    Registry for managing multiple adapters between different model types.
    
    This registry allows registering adapters for specific model type pairs
    and retrieving them when needed for conversions.
    """
    
    def __init__(self):
        """Initialize the adapter registry."""
        self.adapters = {}
    
    def register(
        self,
        source_model: Type[BaseModel],
        target_model: Type[BaseModel],
        adapter: ModelAdapter
    ):
        """
        Register an adapter for a specific source-target model pair.
        
        Args:
            source_model: Source model class
            target_model: Target model class
            adapter: ModelAdapter instance for this pair
        """
        key = (source_model, target_model)
        self.adapters[key] = adapter
    
    def get_adapter(
        self,
        source_model: Type[BaseModel],
        target_model: Type[BaseModel]
    ) -> Optional[ModelAdapter]:
        """
        Get the adapter for a specific source-target model pair.
        
        Args:
            source_model: Source model class
            target_model: Target model class
            
        Returns:
            ModelAdapter instance if registered, None otherwise
        """
        key = (source_model, target_model)
        return self.adapters.get(key)
    
    def adapt(
        self,
        source: BaseModel,
        target_model: Type[BaseModel],
        **extra_fields
    ) -> BaseModel:
        """
        Convert a source model to a target model using the registered adapter.
        
        Args:
            source: Source model instance
            target_model: Target model class
            **extra_fields: Additional fields to include in the target model
            
        Returns:
            An instance of the target model
            
        Raises:
            ValueError: If no adapter is registered for this pair
        """
        adapter = self.get_adapter(source.__class__, target_model)
        if not adapter:
            raise ValueError(f"No adapter registered for {source.__class__.__name__} to {target_model.__name__}")
        
        return adapter.adapt(source, **extra_fields)
    
    def adapt_many(
        self,
        sources: List[BaseModel],
        target_model: Type[BaseModel],
        **extra_fields
    ) -> List[BaseModel]:
        """
        Convert multiple source models to target models using the registered adapter.
        
        Args:
            sources: List of source model instances
            target_model: Target model class
            **extra_fields: Additional fields to include in all target models
            
        Returns:
            List of target model instances
            
        Raises:
            ValueError: If no adapter is registered for this pair or if sources have different types
        """
        if not sources:
            return []
        
        # Check that all sources have the same type
        source_type = sources[0].__class__
        if not all(isinstance(source, source_type) for source in sources):
            raise ValueError("All sources must have the same type")
        
        adapter = self.get_adapter(source_type, target_model)
        if not adapter:
            raise ValueError(f"No adapter registered for {source_type.__name__} to {target_model.__name__}")
        
        return adapter.adapt_many(sources, **extra_fields)


# Example usage with a user management system

# API Request Models
class CreateUserRequest(BaseModel):
    """API request model for creating a user."""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    
    @field_validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', v):
            raise ValueError("Username must be 3-20 characters and contain only letters, numbers, and underscores")
        return v
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UpdateUserRequest(BaseModel):
    """API request model for updating a user."""
    email: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    
    @field_validator('email')
    def validate_email(cls, v):
        if v is not None and not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('password')
    def validate_password(cls, v):
        if v is not None and len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


# Database Models
class UserDB(BaseModel):
    """Database model for a user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    password_hash: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class UserProfileDB(BaseModel):
    """Database model for a user profile."""
    user_id: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    social_links: Dict[str, str] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)


# API Response Models
class UserResponse(BaseModel):
    """API response model for a user."""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime


class UserDetailResponse(BaseModel):
    """API response model for detailed user information."""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    social_links: Dict[str, str] = {}


class UserListResponse(BaseModel):
    """API response model for a list of users."""
    total: int
    users: List[UserResponse]


# Helper functions
def hash_password(password: str) -> str:
    """
    Hash a password (simulated).
    
    In a real application, you would use a secure hashing algorithm like bcrypt.
    """
    return f"hashed_{password}"


# Create the adapter registry
registry = AdapterRegistry()

# Register adapters

# CreateUserRequest -> UserDB
create_user_adapter = ModelAdapter(
    CreateUserRequest,
    UserDB,
    field_mapping={
        "password_hash": "password"  # Map password from request to password_hash in DB
    },
    transformers={
        "password_hash": hash_password  # Transform password to hashed password
    }
)
registry.register(CreateUserRequest, UserDB, create_user_adapter)

# UserDB -> UserResponse
user_response_adapter = ModelAdapter(
    UserDB,
    UserResponse,
    exclude_fields=["password_hash"]  # Exclude sensitive fields
)
registry.register(UserDB, UserResponse, user_response_adapter)

# UserDB + UserProfileDB -> UserDetailResponse
def create_detail_adapter(user_profile: UserProfileDB) -> ModelAdapter:
    """Create an adapter that combines user and profile data."""
    return ModelAdapter(
        UserDB,
        UserDetailResponse,
        exclude_fields=["password_hash"],
        transformers={},
        # Extra fields from profile will be added in adapt() call
    )


def main():
    """Demonstrate the model adapter system."""
    print("=" * 80)
    print("MODEL ADAPTER SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create a user from API request
    print("\n1. Creating a user from API request")
    create_request = CreateUserRequest(
        username="johndoe",
        email="john@example.com",
        password="securepassword",
        full_name="John Doe"
    )
    print(f"API Request: {create_request.model_dump_json(indent=2)}")
    
    # Convert to database model
    user_db = registry.adapt(create_request, UserDB)
    print(f"\nConverted to DB Model: {user_db.model_dump_json(indent=2)}")
    
    # Create a user profile
    user_profile_db = UserProfileDB(
        user_id=user_db.id,
        bio="Software developer and tech enthusiast",
        location="New York, NY",
        website="https://johndoe.example.com",
        social_links={
            "twitter": "https://twitter.com/johndoe",
            "github": "https://github.com/johndoe"
        }
    )
    print(f"\nUser Profile DB: {user_profile_db.model_dump_json(indent=2)}")
    
    # Convert to API response
    user_response = registry.adapt(user_db, UserResponse)
    print(f"\n2. Basic API Response: {user_response.model_dump_json(indent=2)}")
    
    # Create detailed response by combining user and profile
    # We need to create a custom adapter for this specific case
    detail_adapter = create_detail_adapter(user_profile_db)
    user_detail = detail_adapter.adapt(
        user_db,
        # Add profile fields
        bio=user_profile_db.bio,
        avatar_url=user_profile_db.avatar_url,
        location=user_profile_db.location,
        website=user_profile_db.website,
        social_links=user_profile_db.social_links
    )
    print(f"\n3. Detailed API Response: {user_detail.model_dump_json(indent=2)}")
    
    # Create multiple users
    print("\n4. Handling multiple users")
    users_db = [
        UserDB(
            id=str(uuid.uuid4()),
            username=f"user{i}",
            email=f"user{i}@example.com",
            password_hash=f"hashed_password{i}",
            full_name=f"User {i}",
            created_at=datetime.now()
        )
        for i in range(1, 4)
    ]
    
    # Convert multiple users to API responses
    user_responses = registry.adapt_many(users_db, UserResponse)
    
    # Create a list response
    list_response = UserListResponse(
        total=len(user_responses),
        users=user_responses
    )
    print(f"User List Response: {list_response.model_dump_json(indent=2)}")
    
    # Update a user
    print("\n5. Updating a user")
    update_request = UpdateUserRequest(
        email="john.doe@example.com",
        full_name="John D. Doe"
    )
    print(f"Update Request: {update_request.model_dump_json(indent=2)}")
    
    # Apply updates to user_db (in a real app, you'd have an adapter for this)
    for field, value in update_request.model_dump(exclude_unset=True).items():
        if field == "password":
            user_db.password_hash = hash_password(value)
        else:
            setattr(user_db, field, value)
    
    user_db.updated_at = datetime.now()
    print(f"Updated User DB: {user_db.model_dump_json(indent=2)}")
    
    # Convert updated user to API response
    updated_response = registry.adapt(user_db, UserResponse)
    print(f"Updated API Response: {updated_response.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
