"""
Lesson 2 Exercise Solutions
-------------------------
This module contains solutions for the exercises in Lesson 2: Schema Design & Evolution.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum


# Exercise 1: Design a schema for a blog post that includes nested models
# for author information, comments, and metadata.

class UserRole(str, Enum):
    """User roles for blog system."""
    ADMIN = "admin"
    AUTHOR = "author"
    EDITOR = "editor"
    READER = "reader"


class UserProfile(BaseModel):
    """User profile information."""
    display_name: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    website: Optional[str] = None
    social_links: Dict[str, str] = {}


class Author(BaseModel):
    """Author information."""
    id: int
    username: str
    email: str
    role: UserRole = UserRole.AUTHOR
    profile: UserProfile
    joined_date: datetime = Field(default_factory=datetime.now)


class CommentAuthor(BaseModel):
    """Simplified author information for comments."""
    id: int
    username: str
    avatar_url: Optional[str] = None


class Comment(BaseModel):
    """Comment on a blog post."""
    id: int
    author: CommentAuthor
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    parent_id: Optional[int] = None  # For nested comments

    @field_validator('content')
    def content_not_empty(cls, v):
        """Validate that content is not empty."""
        if not v.strip():
            raise ValueError("Comment content cannot be empty")
        return v


class Tag(BaseModel):
    """Tag for categorizing blog posts."""
    id: int
    name: str
    slug: str


class PostStatus(str, Enum):
    """Status of a blog post."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class PostMetadata(BaseModel):
    """Metadata for a blog post."""
    featured_image: Optional[str] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    read_time_minutes: Optional[int] = None
    views_count: int = 0
    likes_count: int = 0
    is_featured: bool = False


class BlogPost(BaseModel):
    """Complete blog post model."""
    id: int
    title: str = Field(..., min_length=3, max_length=200)
    slug: str = Field(..., min_length=3, max_length=200)
    content: str
    excerpt: Optional[str] = None
    author: Author
    status: PostStatus = PostStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    tags: List[Tag] = []
    comments: List[Comment] = []
    metadata: PostMetadata = Field(default_factory=PostMetadata)

    def model_post_init(self, __context):
        """Generate excerpt from content if not provided."""
        if self.excerpt is None:
            # Generate a simple excerpt (first 150 chars)
            self.excerpt = self.content[:150] + "..." if len(self.content) > 150 else self.content


# Exercise 2: Create a versioned schema for a user profile that evolves
# from a simple version to a more complex one with additional fields.

class UserProfileV1(BaseModel):
    """Version 1 of user profile schema."""
    id: int
    username: str
    email: str
    name: str
    bio: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class Address(BaseModel):
    """Address information for user profile."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class SocialMedia(BaseModel):
    """Social media links for user profile."""
    twitter: Optional[str] = None
    facebook: Optional[str] = None
    instagram: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class UserProfileV2(BaseModel):
    """Version 2 of user profile schema."""
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    address: Optional[Address] = None
    phone_number: Optional[str] = None
    social_media: SocialMedia = Field(default_factory=SocialMedia)
    preferences: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None

    @classmethod
    def from_v1(cls, profile_v1: UserProfileV1):
        """
        Migrate from UserProfileV1 to UserProfileV2.

        Args:
            profile_v1: UserProfileV1 instance

        Returns:
            UserProfileV2 instance
        """
        # Split name into first_name and last_name
        name_parts = profile_v1.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        return cls(
            id=profile_v1.id,
            username=profile_v1.username,
            email=profile_v1.email,
            first_name=first_name,
            last_name=last_name,
            bio=profile_v1.bio,
            created_at=profile_v1.created_at
        )


class UserSettings(BaseModel):
    """User settings for preferences."""
    theme: str = "light"
    email_notifications: bool = True
    two_factor_auth: bool = False
    language: str = "en"
    timezone: str = "UTC"


class UserProfileV3(BaseModel):
    """Version 3 of user profile schema."""
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    address: Optional[Address] = None
    phone_number: Optional[str] = None
    social_media: SocialMedia = Field(default_factory=SocialMedia)
    settings: UserSettings = Field(default_factory=UserSettings)
    preferences: Dict[str, Any] = {}  # Deprecated in favor of settings
    roles: List[UserRole] = [UserRole.READER]
    is_verified: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    @classmethod
    def from_v2(cls, profile_v2: UserProfileV2):
        """
        Migrate from UserProfileV2 to UserProfileV3.

        Args:
            profile_v2: UserProfileV2 instance

        Returns:
            UserProfileV3 instance
        """
        # Create settings from preferences
        settings = UserSettings()

        # Extract known settings from preferences
        if "theme" in profile_v2.preferences:
            settings.theme = profile_v2.preferences["theme"]
        if "email_notifications" in profile_v2.preferences:
            settings.email_notifications = profile_v2.preferences["email_notifications"]
        if "language" in profile_v2.preferences:
            settings.language = profile_v2.preferences["language"]

        return cls(
            id=profile_v2.id,
            username=profile_v2.username,
            email=profile_v2.email,
            first_name=profile_v2.first_name,
            last_name=profile_v2.last_name,
            bio=profile_v2.bio,
            avatar_url=profile_v2.avatar_url,
            address=profile_v2.address,
            phone_number=profile_v2.phone_number,
            social_media=profile_v2.social_media,
            settings=settings,
            preferences=profile_v2.preferences,  # Keep for backward compatibility
            created_at=profile_v2.created_at,
            updated_at=profile_v2.updated_at
        )


# Exercise 3: Implement a migration function that can convert data
# from an older schema version to a newer one.

class SchemaEvolution:
    """Utility for schema evolution and migration."""

    @staticmethod
    def migrate_user_profile_v1_to_v2(data: dict) -> dict:
        """
        Migrate user profile data from v1 to v2 format.

        Args:
            data: Dictionary with v1 user profile data

        Returns:
            Dictionary with v2 user profile data
        """
        # Validate input data against v1 schema
        profile_v1 = UserProfileV1(**data)

        # Split name into first_name and last_name
        name_parts = profile_v1.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Create v2 data
        profile_v2_data = {
            "id": profile_v1.id,
            "username": profile_v1.username,
            "email": profile_v1.email,
            "first_name": first_name,
            "last_name": last_name,
            "bio": profile_v1.bio,
            "created_at": profile_v1.created_at,
            "avatar_url": None,
            "address": None,
            "phone_number": None,
            "social_media": {},
            "preferences": {}
        }

        return profile_v2_data

    @staticmethod
    def migrate_user_profile_v2_to_v3(data: dict) -> dict:
        """
        Migrate user profile data from v2 to v3 format.

        Args:
            data: Dictionary with v2 user profile data

        Returns:
            Dictionary with v3 user profile data
        """
        # Validate input data against v2 schema
        profile_v2 = UserProfileV2(**data)

        # Create settings from preferences
        settings = {
            "theme": "light",
            "email_notifications": True,
            "two_factor_auth": False,
            "language": "en",
            "timezone": "UTC"
        }

        # Extract known settings from preferences
        if "theme" in profile_v2.preferences:
            settings["theme"] = profile_v2.preferences["theme"]
        if "email_notifications" in profile_v2.preferences:
            settings["email_notifications"] = profile_v2.preferences["email_notifications"]
        if "language" in profile_v2.preferences:
            settings["language"] = profile_v2.preferences["language"]

        # Create v3 data
        profile_v3_data = {
            "id": profile_v2.id,
            "username": profile_v2.username,
            "email": profile_v2.email,
            "first_name": profile_v2.first_name,
            "last_name": profile_v2.last_name,
            "bio": profile_v2.bio,
            "avatar_url": profile_v2.avatar_url,
            "address": profile_v2.address.model_dump() if profile_v2.address else None,
            "phone_number": profile_v2.phone_number,
            "social_media": profile_v2.social_media.model_dump(),
            "settings": settings,
            "preferences": profile_v2.preferences,
            "roles": ["reader"],
            "is_verified": False,
            "created_at": profile_v2.created_at,
            "updated_at": profile_v2.updated_at
        }

        return profile_v3_data

    @staticmethod
    def migrate_user_profile(data: dict, from_version: int, to_version: int) -> dict:
        """
        Migrate user profile data between versions.

        Args:
            data: Dictionary with user profile data
            from_version: Source version (1, 2, or 3)
            to_version: Target version (1, 2, or 3)

        Returns:
            Dictionary with migrated user profile data
        """
        if from_version == to_version:
            return data

        if from_version > to_version:
            raise ValueError("Cannot downgrade schema version")

        if from_version == 1 and to_version == 2:
            return SchemaEvolution.migrate_user_profile_v1_to_v2(data)

        if from_version == 2 and to_version == 3:
            return SchemaEvolution.migrate_user_profile_v2_to_v3(data)

        if from_version == 1 and to_version == 3:
            # Two-step migration
            v2_data = SchemaEvolution.migrate_user_profile_v1_to_v2(data)
            return SchemaEvolution.migrate_user_profile_v2_to_v3(v2_data)

        raise ValueError(f"Unsupported migration from v{from_version} to v{to_version}")


# Exercise 4: Generate JSON Schema for your models and analyze the output.

class ConfigModel(BaseModel):
    """Model with custom configuration for JSON Schema generation."""

    id: int
    name: str
    description: Optional[str] = Field(
        None,
        description="A detailed description of the item",
        examples=["This is a sample item with various properties"]
    )
    tags: List[str] = Field(
        [],
        description="Tags for categorizing the item",
        examples=[["sample", "example", "test"]]
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the item was created"
    )

    class Config:
        """Configuration for the model."""
        title = "Configured Item"
        description = "A model with custom JSON Schema configuration"
        json_schema_extra = {
            "examples": [
                {
                    "id": 1,
                    "name": "Sample Item",
                    "description": "This is a sample item",
                    "tags": ["sample", "example"],
                    "created_at": "2023-01-01T00:00:00Z"
                }
            ]
        }


class NestedModel(BaseModel):
    """Complex nested model for JSON Schema generation."""

    class Status(str, Enum):
        """Status enum for the nested model."""
        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"

    class Location(BaseModel):
        """Location submodel."""
        latitude: float
        longitude: float
        name: Optional[str] = None

    class Contact(BaseModel):
        """Contact submodel."""
        email: str
        phone: Optional[str] = None

    id: int
    name: str
    status: Status = Status.ACTIVE
    location: Optional[Location] = None
    contacts: List[Contact] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


def generate_json_schema(model_class):
    """
    Generate JSON schema for a model class.

    Args:
        model_class: Pydantic model class

    Returns:
        JSON schema as a dictionary
    """
    return model_class.model_json_schema()


if __name__ == "__main__":
    # Demonstrate blog post schema
    author = Author(
        id=1,
        username="johndoe",
        email="john@example.com",
        profile=UserProfile(
            display_name="John Doe",
            bio="Tech blogger and developer",
            avatar_url="https://example.com/avatar.jpg"
        )
    )

    tag1 = Tag(id=1, name="Python", slug="python")
    tag2 = Tag(id=2, name="Pydantic", slug="pydantic")

    comment = Comment(
        id=1,
        author=CommentAuthor(id=2, username="janedoe", avatar_url="https://example.com/jane.jpg"),
        content="Great post!"
    )

    post = BlogPost(
        id=1,
        title="Advanced Pydantic Features",
        slug="advanced-pydantic-features",
        content="This is a detailed post about Pydantic...",
        author=author,
        tags=[tag1, tag2],
        comments=[comment],
        status=PostStatus.PUBLISHED,
        published_at=datetime.now()
    )

    print(f"Blog Post: {post.title} by {post.author.profile.display_name}")
    print(f"Tags: {[tag.name for tag in post.tags]}")
    print(f"Comments: {len(post.comments)}")

    # Demonstrate schema evolution
    profile_v1_data = {
        "id": 1,
        "username": "johndoe",
        "email": "john@example.com",
        "name": "John Doe",
        "bio": "Tech enthusiast",
        "created_at": datetime.now()
    }

    profile_v2_data = SchemaEvolution.migrate_user_profile_v1_to_v2(profile_v1_data)
    profile_v3_data = SchemaEvolution.migrate_user_profile_v2_to_v3(profile_v2_data)

    print(f"\nMigrated from v1 to v2: {profile_v2_data}")
    print(f"Migrated from v2 to v3: {profile_v3_data}")

    # Demonstrate JSON Schema generation
    config_schema = generate_json_schema(ConfigModel)
    nested_schema = generate_json_schema(NestedModel)

    print(f"\nConfigModel JSON Schema: {config_schema}")
    print(f"NestedModel JSON Schema: {nested_schema}")
