"""
Tests for Lesson 2 Exercise Solutions
----------------------------------
This module contains tests for the lesson2_exercises module.
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from module3.exercises.lesson2_exercises import (
    # Exercise 1
    BlogPost, Author, UserProfile, CommentAuthor, Comment, Tag, PostStatus, PostMetadata,
    # Exercise 2
    UserProfileV1, UserProfileV2, UserProfileV3, Address, SocialMedia, UserSettings,
    # Exercise 3
    SchemaEvolution,
    # Exercise 4
    ConfigModel, NestedModel, generate_json_schema
)


class TestLesson2Exercises(unittest.TestCase):
    """Test cases for lesson2_exercises module."""

    def test_blog_post_schema(self):
        """Test the blog post schema."""
        # Create author
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

        # Create tags
        tag1 = Tag(id=1, name="Python", slug="python")
        tag2 = Tag(id=2, name="Pydantic", slug="pydantic")

        # Create comment
        comment = Comment(
            id=1,
            author=CommentAuthor(id=2, username="janedoe", avatar_url="https://example.com/jane.jpg"),
            content="Great post!"
        )

        # Create blog post
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

        # Test basic properties
        self.assertEqual(post.id, 1)
        self.assertEqual(post.title, "Advanced Pydantic Features")
        self.assertEqual(post.author.username, "johndoe")
        self.assertEqual(post.author.profile.display_name, "John Doe")
        self.assertEqual(len(post.tags), 2)
        self.assertEqual(post.tags[0].name, "Python")
        self.assertEqual(len(post.comments), 1)
        self.assertEqual(post.comments[0].content, "Great post!")
        self.assertEqual(post.status, PostStatus.PUBLISHED)

        # Test excerpt generation
        self.assertEqual(post.excerpt, post.content)

        # Test metadata defaults
        self.assertEqual(post.metadata.views_count, 0)
        self.assertEqual(post.metadata.likes_count, 0)
        self.assertFalse(post.metadata.is_featured)

        # Test validation
        with self.assertRaises(ValidationError):
            # Title too short
            BlogPost(
                id=2,
                title="A",  # Too short
                slug="a",
                content="Content",
                author=author
            )

        with self.assertRaises(ValidationError):
            # Empty comment content
            Comment(
                id=2,
                author=CommentAuthor(id=2, username="janedoe"),
                content=""  # Empty content
            )

    def test_user_profile_versioning(self):
        """Test user profile versioning."""
        # Create v1 profile
        profile_v1 = UserProfileV1(
            id=1,
            username="johndoe",
            email="john@example.com",
            name="John Doe",
            bio="Tech enthusiast",
            created_at=datetime.now()
        )

        # Test v1 properties
        self.assertEqual(profile_v1.id, 1)
        self.assertEqual(profile_v1.username, "johndoe")
        self.assertEqual(profile_v1.name, "John Doe")

        # Migrate to v2
        profile_v2 = UserProfileV2.from_v1(profile_v1)

        # Test v2 properties
        self.assertEqual(profile_v2.id, 1)
        self.assertEqual(profile_v2.username, "johndoe")
        self.assertEqual(profile_v2.first_name, "John")
        self.assertEqual(profile_v2.last_name, "Doe")
        self.assertEqual(profile_v2.bio, "Tech enthusiast")
        self.assertIsNone(profile_v2.avatar_url)
        self.assertIsNone(profile_v2.address)
        self.assertIsNone(profile_v2.phone_number)
        self.assertEqual(profile_v2.social_media.twitter, None)
        self.assertEqual(profile_v2.preferences, {})

        # Create v2 profile with additional fields
        profile_v2 = UserProfileV2(
            id=1,
            username="johndoe",
            email="john@example.com",
            first_name="John",
            last_name="Doe",
            bio="Tech enthusiast",
            avatar_url="https://example.com/avatar.jpg",
            address=Address(
                street="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="12345",
                country="USA"
            ),
            phone_number="555-1234",
            social_media=SocialMedia(
                twitter="@johndoe",
                github="johndoe"
            ),
            preferences={"theme": "dark", "email_notifications": False},
            created_at=datetime.now()
        )

        # Test v2 properties with additional fields
        self.assertEqual(profile_v2.avatar_url, "https://example.com/avatar.jpg")
        self.assertEqual(profile_v2.address.city, "Anytown")
        self.assertEqual(profile_v2.social_media.twitter, "@johndoe")
        self.assertEqual(profile_v2.preferences["theme"], "dark")

        # Migrate to v3
        profile_v3 = UserProfileV3.from_v2(profile_v2)

        # Test v3 properties
        self.assertEqual(profile_v3.id, 1)
        self.assertEqual(profile_v3.username, "johndoe")
        self.assertEqual(profile_v3.first_name, "John")
        self.assertEqual(profile_v3.last_name, "Doe")
        self.assertEqual(profile_v3.avatar_url, "https://example.com/avatar.jpg")
        self.assertEqual(profile_v3.address.city, "Anytown")
        self.assertEqual(profile_v3.social_media.twitter, "@johndoe")
        self.assertEqual(profile_v3.settings.theme, "dark")
        self.assertEqual(profile_v3.settings.email_notifications, False)
        self.assertEqual(profile_v3.settings.language, "en")  # Default value
        self.assertEqual(profile_v3.roles, ["reader"])  # Default role
        self.assertFalse(profile_v3.is_verified)  # Default value

    def test_schema_evolution(self):
        """Test schema evolution utility."""
        # Create v1 data
        profile_v1_data = {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "name": "John Doe",
            "bio": "Tech enthusiast",
            "created_at": datetime.now()
        }

        # Migrate v1 to v2
        profile_v2_data = SchemaEvolution.migrate_user_profile_v1_to_v2(profile_v1_data)

        # Test v2 data
        self.assertEqual(profile_v2_data["id"], 1)
        self.assertEqual(profile_v2_data["username"], "johndoe")
        self.assertEqual(profile_v2_data["first_name"], "John")
        self.assertEqual(profile_v2_data["last_name"], "Doe")
        self.assertEqual(profile_v2_data["bio"], "Tech enthusiast")
        self.assertIsNone(profile_v2_data["avatar_url"])
        self.assertIsNone(profile_v2_data["address"])
        self.assertIsNone(profile_v2_data["phone_number"])
        self.assertEqual(profile_v2_data["social_media"], {})
        self.assertEqual(profile_v2_data["preferences"], {})

        # Create v2 data with preferences
        profile_v2_data = {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "bio": "Tech enthusiast",
            "avatar_url": "https://example.com/avatar.jpg",
            "address": None,
            "phone_number": "555-1234",
            "social_media": {"twitter": "@johndoe", "github": "johndoe"},
            "preferences": {"theme": "dark", "email_notifications": False, "language": "fr"},
            "created_at": datetime.now()
        }

        # Migrate v2 to v3
        profile_v3_data = SchemaEvolution.migrate_user_profile_v2_to_v3(profile_v2_data)

        # Test v3 data
        self.assertEqual(profile_v3_data["id"], 1)
        self.assertEqual(profile_v3_data["username"], "johndoe")
        self.assertEqual(profile_v3_data["first_name"], "John")
        self.assertEqual(profile_v3_data["last_name"], "Doe")
        self.assertEqual(profile_v3_data["avatar_url"], "https://example.com/avatar.jpg")
        self.assertEqual(profile_v3_data["phone_number"], "555-1234")
        self.assertEqual(profile_v3_data["social_media"]["twitter"], "@johndoe")
        self.assertEqual(profile_v3_data["settings"]["theme"], "dark")
        self.assertEqual(profile_v3_data["settings"]["email_notifications"], False)
        self.assertEqual(profile_v3_data["settings"]["language"], "fr")
        self.assertEqual(profile_v3_data["roles"], ["reader"])
        self.assertFalse(profile_v3_data["is_verified"])

        # Test direct migration from v1 to v3
        profile_v3_data_direct = SchemaEvolution.migrate_user_profile(profile_v1_data, 1, 3)

        # Test v3 data from direct migration
        self.assertEqual(profile_v3_data_direct["id"], 1)
        self.assertEqual(profile_v3_data_direct["username"], "johndoe")
        self.assertEqual(profile_v3_data_direct["first_name"], "John")
        self.assertEqual(profile_v3_data_direct["last_name"], "Doe")

        # Test error cases
        with self.assertRaises(ValueError):
            # Cannot downgrade
            SchemaEvolution.migrate_user_profile(profile_v2_data, 2, 1)

        with self.assertRaises(ValueError):
            # Unsupported migration
            SchemaEvolution.migrate_user_profile(profile_v1_data, 1, 4)

    def test_json_schema_generation(self):
        """Test JSON schema generation."""
        # Generate schema for ConfigModel
        config_schema = generate_json_schema(ConfigModel)

        # Test schema properties
        self.assertEqual(config_schema["title"], "Configured Item")
        self.assertEqual(config_schema["description"], "Model with custom configuration for JSON Schema generation.")
        self.assertTrue("examples" in config_schema)
        self.assertTrue("properties" in config_schema)
        self.assertTrue("id" in config_schema["properties"])
        self.assertTrue("name" in config_schema["properties"])
        self.assertTrue("description" in config_schema["properties"])
        self.assertEqual(config_schema["properties"]["description"]["description"],
                         "A detailed description of the item")

        # Generate schema for NestedModel
        nested_schema = generate_json_schema(NestedModel)

        # Test schema properties
        self.assertEqual(nested_schema["title"], "NestedModel")
        self.assertTrue("properties" in nested_schema)
        self.assertTrue("id" in nested_schema["properties"])
        self.assertTrue("name" in nested_schema["properties"])
        self.assertTrue("status" in nested_schema["properties"])
        self.assertTrue("location" in nested_schema["properties"])
        self.assertTrue("contacts" in nested_schema["properties"])

        # Test nested properties
        self.assertTrue("$defs" in nested_schema)
        self.assertTrue("Location" in nested_schema["$defs"])
        self.assertTrue("Contact" in nested_schema["$defs"])
        self.assertTrue("latitude" in nested_schema["$defs"]["Location"]["properties"])
        self.assertTrue("longitude" in nested_schema["$defs"]["Location"]["properties"])
        self.assertTrue("email" in nested_schema["$defs"]["Contact"]["properties"])


if __name__ == "__main__":
    unittest.main()
