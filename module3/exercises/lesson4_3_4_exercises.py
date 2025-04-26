"""
Lesson 4.3.4: Model Adapter System

This exercise implements an adapter system that can convert between API request models,
database models, and API response models.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Type, List, Optional, Callable, TypeVar, Generic
from datetime import datetime
import uuid


# Type variables for source and target models
S = TypeVar('S', bound=BaseModel)
T = TypeVar('T', bound=BaseModel)


class ModelAdapter(Generic[S, T]):
    """
    Adapter for converting between different Pydantic models.
    
    This adapter can convert from a source model type to a target model type,
    with optional field mapping and transformations.
    """
    
    def __init__(
        self,
        source_model: Type[S],
        target_model: Type[T],
        field_mapping: Dict[str, str] = None,
        transformers: Dict[str, Callable] = None,
        exclude_fields: List[str] = None,
        include_unmapped: bool = True
    ):
        """
        Initialize the adapter.
        
        Args:
            source_model: The source model type
            target_model: The target model type
            field_mapping: Optional mapping from target field to source field
            transformers: Optional functions to transform field values
            exclude_fields: Fields to exclude from the target model
            include_unmapped: Whether to include fields not in the mapping
        """
        self.source_model = source_model
        self.target_model = target_model
        self.field_mapping = field_mapping or {}
        self.transformers = transformers or {}
        self.exclude_fields = exclude_fields or []
        self.include_unmapped = include_unmapped
    
    def adapt(self, source: S) -> T:
        """
        Convert a source model instance to a target model instance.
        
        Args:
            source: The source model instance
            
        Returns:
            A new instance of the target model
        """
        # Convert source to dict
        source_data = source.model_dump()
        
        # Prepare target data
        target_data = {}
        
        # Get target model fields
        target_fields = self.target_model.model_fields
        
        # Process each target field
        for field_name in target_fields:
            # Skip excluded fields
            if field_name in self.exclude_fields:
                continue
            
            # Get the corresponding source field
            source_field = self.field_mapping.get(field_name, field_name)
            
            # Skip if source field doesn't exist and we're not including unmapped fields
            if source_field not in source_data and not self.include_unmapped:
                continue
            
            # Get the value from source data
            if source_field in source_data:
                value = source_data[source_field]
                
                # Apply transformer if available
                if field_name in self.transformers:
                    transformer = self.transformers[field_name]
                    value = transformer(value)
                
                target_data[field_name] = value
        
        # Create and return target instance
        return self.target_model(**target_data)
    
    def adapt_many(self, sources: List[S]) -> List[T]:
        """
        Convert multiple source model instances to target model instances.
        
        Args:
            sources: List of source model instances
            
        Returns:
            List of target model instances
        """
        return [self.adapt(source) for source in sources]


class AdapterRegistry:
    """
    Registry for model adapters.
    
    This registry manages adapters for converting between different model types.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.adapters = {}
    
    def register(self, source_model: Type[BaseModel], target_model: Type[BaseModel], adapter: ModelAdapter):
        """
        Register an adapter for a source-target model pair.
        
        Args:
            source_model: The source model type
            target_model: The target model type
            adapter: The adapter instance
        """
        key = (source_model, target_model)
        self.adapters[key] = adapter
    
    def get_adapter(self, source_model: Type[BaseModel], target_model: Type[BaseModel]) -> Optional[ModelAdapter]:
        """
        Get the adapter for a source-target model pair.
        
        Args:
            source_model: The source model type
            target_model: The target model type
            
        Returns:
            The adapter instance, or None if not found
        """
        key = (source_model, target_model)
        return self.adapters.get(key)
    
    def adapt(self, source: BaseModel, target_model: Type[BaseModel]) -> BaseModel:
        """
        Convert a source model instance to a target model instance.
        
        Args:
            source: The source model instance
            target_model: The target model type
            
        Returns:
            A new instance of the target model
            
        Raises:
            ValueError: If no adapter is registered for the source-target pair
        """
        adapter = self.get_adapter(type(source), target_model)
        if not adapter:
            raise ValueError(f"No adapter registered for {type(source).__name__} to {target_model.__name__}")
        
        return adapter.adapt(source)
    
    def adapt_many(self, sources: List[BaseModel], target_model: Type[BaseModel]) -> List[BaseModel]:
        """
        Convert multiple source model instances to target model instances.
        
        Args:
            sources: List of source model instances
            target_model: The target model type
            
        Returns:
            List of target model instances
            
        Raises:
            ValueError: If no adapter is registered for the source-target pair
        """
        if not sources:
            return []
        
        adapter = self.get_adapter(type(sources[0]), target_model)
        if not adapter:
            raise ValueError(f"No adapter registered for {type(sources[0]).__name__} to {target_model.__name__}")
        
        return adapter.adapt_many(sources)


# Example models for demonstration

# API Request Models
class CreateUserRequest(BaseModel):
    """API request model for creating a user."""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UpdateUserRequest(BaseModel):
    """API request model for updating a user."""
    email: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


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
    updated_at: datetime = Field(default_factory=datetime.now)


class UserProfileDB(BaseModel):
    """Database model for a user profile."""
    user_id: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    social_links: Dict[str, str] = {}
    preferences: Dict[str, Any] = {}


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

# UpdateUserRequest -> UserDB
update_user_adapter = ModelAdapter(
    UpdateUserRequest,
    UserDB,
    transformers={
        "password_hash": hash_password  # Transform password to hashed password
    },
    include_unmapped=False  # Only include fields that are in the update request
)
registry.register(UpdateUserRequest, UserDB, update_user_adapter)

# UserDB -> UserResponse
user_response_adapter = ModelAdapter(
    UserDB,
    UserResponse,
    exclude_fields=["password_hash"]  # Don't include password hash in response
)
registry.register(UserDB, UserResponse, user_response_adapter)

# UserDB + UserProfileDB -> UserDetailResponse
def combine_user_and_profile(user_db: UserDB, profile_db: UserProfileDB) -> UserDetailResponse:
    """Combine user and profile data into a detailed response."""
    user_data = user_db.model_dump(exclude={"password_hash"})
    profile_data = profile_db.model_dump(exclude={"user_id"})
    
    return UserDetailResponse(**user_data, **profile_data)


def main():
    """Demonstrate the model adapter system."""
    print("Model Adapter System Demonstration")
    print("=" * 40)
    
    # Create a user request
    create_request = CreateUserRequest(
        username="johndoe",
        email="john@example.com",
        password="securepassword",
        full_name="John Doe"
    )
    print("\nCreate User Request:")
    print(create_request.model_dump_json(indent=2))
    
    # Convert to database model
    user_db = registry.adapt(create_request, UserDB)
    print("\nConverted to User DB Model:")
    print(user_db.model_dump_json(indent=2))
    
    # Create a user profile
    profile_db = UserProfileDB(
        user_id=user_db.id,
        bio="Software developer and tech enthusiast",
        location="San Francisco, CA",
        website="https://johndoe.example.com",
        social_links={
            "twitter": "https://twitter.com/johndoe",
            "github": "https://github.com/johndoe"
        }
    )
    print("\nUser Profile DB Model:")
    print(profile_db.model_dump_json(indent=2))
    
    # Convert to API response
    user_response = registry.adapt(user_db, UserResponse)
    print("\nConverted to User Response:")
    print(user_response.model_dump_json(indent=2))
    
    # Combine user and profile for detailed response
    user_detail = combine_user_and_profile(user_db, profile_db)
    print("\nCombined User Detail Response:")
    print(user_detail.model_dump_json(indent=2))
    
    # Update user request
    update_request = UpdateUserRequest(
        email="john.doe@example.com",
        full_name="John D. Doe"
    )
    print("\nUpdate User Request:")
    print(update_request.model_dump_json(indent=2))
    
    # Apply update to user DB model
    # In a real application, you would fetch the user from the database first
    updated_user_db = user_db.model_copy()
    
    # Only update fields that are present in the update request
    update_data = {}
    for field, value in update_request.model_dump(exclude_unset=True).items():
        if field == "password":
            update_data["password_hash"] = hash_password(value)
        else:
            update_data[field] = value
    
    for field, value in update_data.items():
        setattr(updated_user_db, field, value)
    
    updated_user_db.updated_at = datetime.now()
    
    print("\nUpdated User DB Model:")
    print(updated_user_db.model_dump_json(indent=2))
    
    # Convert updated user to response
    updated_response = registry.adapt(updated_user_db, UserResponse)
    print("\nUpdated User Response:")
    print(updated_response.model_dump_json(indent=2))
    
    # Batch conversion example
    users_db = [user_db, updated_user_db]
    user_responses = [registry.adapt(u, UserResponse) for u in users_db]
    
    print("\nBatch Conversion Result:")
    for i, response in enumerate(user_responses):
        print(f"\nUser {i+1}:")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
