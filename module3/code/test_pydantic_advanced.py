"""
Tests for Advanced Pydantic Features
---------------------------------
This module contains tests for the pydantic_advanced module.
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from module3.code.pydantic_advanced import (
    User, Post, UserWithPosts,
    Address, ContactInfo, UserWithContact,
    UserBasic, UserDetailed, UserComplete,
    Image, Author, Article,
    Paginated,
    TextContent, ImageContent, VideoContent, Message,
    Dog, Cat, Parrot, process_pet,
    UserV1, UserV2,
    ProductV1, ProductV2,
    Command, CreateTaskV1, DeleteTaskV1, CreateTaskV2, DeleteTaskV2,
    CommandRegistry, SchemaRegistry
)


class TestAdvancedPydanticFeatures(unittest.TestCase):
    """Test cases for pydantic_advanced module."""
    
    def test_single_responsibility(self):
        """Test models with single responsibility."""
        # Create a user
        user = User(id=1, username="john_doe", email="john@example.com")
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "john_doe")
        
        # Create posts
        post1 = Post(id=1, user_id=1, title="First Post", content="Hello World")
        post2 = Post(id=2, user_id=1, title="Second Post", content="Another post")
        
        # Create user with posts
        user_with_posts = UserWithPosts(user=user, posts=[post1, post2])
        self.assertEqual(user_with_posts.user.id, 1)
        self.assertEqual(len(user_with_posts.posts), 2)
        self.assertEqual(user_with_posts.posts[0].title, "First Post")
    
    def test_composition(self):
        """Test model composition."""
        # Create address
        address = Address(
            street="123 Main St",
            city="Anytown",
            state="CA",
            zip_code="12345",
            country="USA"
        )
        
        # Create contact info
        contact_info = ContactInfo(
            email="john@example.com",
            phone="555-1234",
            address=address
        )
        
        # Create user with contact
        user = UserWithContact(
            id=1,
            username="john_doe",
            contact_info=contact_info
        )
        
        self.assertEqual(user.id, 1)
        self.assertEqual(user.contact_info.email, "john@example.com")
        self.assertEqual(user.contact_info.address.city, "Anytown")
    
    def test_progressive_disclosure(self):
        """Test progressive disclosure pattern."""
        # Basic user
        basic_user = UserBasic(id=1, username="john_doe")
        self.assertEqual(basic_user.id, 1)
        
        # Detailed user
        detailed_user = UserDetailed(
            id=1,
            username="john_doe",
            email="john@example.com",
            full_name="John Doe",
            created_at=datetime.now()
        )
        self.assertEqual(detailed_user.id, 1)
        self.assertEqual(detailed_user.email, "john@example.com")
        
        # Complete user
        address = Address(
            street="123 Main St",
            city="Anytown",
            state="CA",
            zip_code="12345",
            country="USA"
        )
        
        contact_info = ContactInfo(
            email="john@example.com",
            phone="555-1234",
            address=address
        )
        
        complete_user = UserComplete(
            id=1,
            username="john_doe",
            email="john@example.com",
            full_name="John Doe",
            created_at=datetime.now(),
            contact_info=contact_info,
            preferences={"theme": "dark", "notifications": True},
            security_settings={"two_factor": True}
        )
        
        self.assertEqual(complete_user.id, 1)
        self.assertEqual(complete_user.contact_info.address.city, "Anytown")
        self.assertEqual(complete_user.preferences["theme"], "dark")
    
    def test_nested_models(self):
        """Test nested models."""
        # Create image
        image = Image(url="https://example.com/image.jpg", width=800, height=600)
        
        # Create author
        author = Author(
            name="Jane Smith",
            bio="Tech writer and developer",
            avatar=image
        )
        
        # Create article
        article = Article(
            title="Advanced Pydantic Features",
            content="This is an article about Pydantic...",
            author=author,
            cover_image=image,
            tags=["python", "pydantic", "validation"]
        )
        
        self.assertEqual(article.title, "Advanced Pydantic Features")
        self.assertEqual(article.author.name, "Jane Smith")
        self.assertEqual(article.cover_image.width, 800)
        self.assertEqual(len(article.tags), 3)
    
    def test_generic_models(self):
        """Test generic models."""
        # Create users
        users = [
            User(id=1, username="alice", email="alice@example.com"),
            User(id=2, username="bob", email="bob@example.com")
        ]
        
        # Create paginated users
        user_page = Paginated[User](
            items=users,
            total=10,
            page=1,
            page_size=2
        )
        
        self.assertEqual(len(user_page.items), 2)
        self.assertEqual(user_page.items[0].username, "alice")
        self.assertEqual(user_page.total_pages, 5)
        self.assertTrue(user_page.has_next)
        self.assertFalse(user_page.has_previous)
        
        # Create posts
        posts = [
            Post(id=1, user_id=1, title="First Post", content="Hello World"),
            Post(id=2, user_id=1, title="Second Post", content="Another post")
        ]
        
        # Create paginated posts
        post_page = Paginated[Post](
            items=posts,
            total=5,
            page=1,
            page_size=2
        )
        
        self.assertEqual(len(post_page.items), 2)
        self.assertEqual(post_page.items[0].title, "First Post")
        self.assertEqual(post_page.total_pages, 3)
    
    def test_union_types(self):
        """Test union types."""
        # Create text message
        text_message = Message(
            id=1,
            sender="alice",
            content=TextContent(text="Hello, how are you?")
        )
        
        self.assertEqual(text_message.id, 1)
        self.assertEqual(text_message.content.type, "text")
        self.assertEqual(text_message.content.text, "Hello, how are you?")
        
        # Create image message
        image_message = Message(
            id=2,
            sender="bob",
            content=ImageContent(url="https://example.com/image.jpg", caption="Check this out")
        )
        
        self.assertEqual(image_message.id, 2)
        self.assertEqual(image_message.content.type, "image")
        self.assertEqual(image_message.content.caption, "Check this out")
        
        # Create video message
        video_message = Message(
            id=3,
            sender="charlie",
            content=VideoContent(
                url="https://example.com/video.mp4",
                duration=120,
                thumbnail="https://example.com/thumbnail.jpg"
            )
        )
        
        self.assertEqual(video_message.id, 3)
        self.assertEqual(video_message.content.type, "video")
        self.assertEqual(video_message.content.duration, 120)
    
    def test_discriminated_unions(self):
        """Test discriminated unions."""
        # Process dog
        dog_data = {"type": "dog", "name": "Rex", "breed": "German Shepherd"}
        dog_result = process_pet(dog_data)
        self.assertEqual(dog_result, "Dog: Rex, breed: German Shepherd")
        
        # Process cat
        cat_data = {"type": "cat", "name": "Whiskers", "lives_left": 9}
        cat_result = process_pet(cat_data)
        self.assertEqual(cat_result, "Cat: Whiskers, lives left: 9")
        
        # Process parrot
        parrot_data = {"type": "parrot", "name": "Polly", "can_speak": True}
        parrot_result = process_pet(parrot_data)
        self.assertEqual(parrot_result, "Parrot: Polly, can speak: True")
        
        # Test invalid type
        with self.assertRaises(ValidationError):
            process_pet({"type": "fish", "name": "Nemo"})
    
    def test_schema_evolution(self):
        """Test schema evolution strategies."""
        # Test versioning
        user_v1 = UserV1(id=1, name="John Doe", email="john@example.com")
        user_v2 = UserV2.from_v1(user_v1)
        
        self.assertEqual(user_v1.id, 1)
        self.assertEqual(user_v1.name, "John Doe")
        
        self.assertEqual(user_v2.id, 1)
        self.assertEqual(user_v2.first_name, "John")
        self.assertEqual(user_v2.last_name, "Doe")
        self.assertEqual(user_v2.email, "john@example.com")
        
        # Test backward compatibility
        product_v1 = ProductV1(id=1, name="Laptop", price=999.99)
        product_v2 = ProductV2(id=1, name="Laptop", price=999.99)
        product_v2_full = ProductV2(
            id=1,
            name="Laptop",
            price=999.99,
            description="A powerful laptop",
            category="Electronics"
        )
        
        self.assertEqual(product_v1.id, 1)
        self.assertEqual(product_v1.price, 999.99)
        
        self.assertEqual(product_v2.id, 1)
        self.assertEqual(product_v2.price, 999.99)
        self.assertIsNone(product_v2.description)
        
        self.assertEqual(product_v2_full.description, "A powerful laptop")
        self.assertEqual(product_v2_full.category, "Electronics")
    
    def test_schema_registry(self):
        """Test schema registry."""
        # Create registry
        registry = SchemaRegistry()
        
        # Register schemas
        registry.register_schema("user", 1, UserV1)
        registry.register_schema("user", 2, UserV2)
        
        # Register migrations
        registry.register_migration("user", 1, 2, lambda data: {
            "id": data["id"],
            "first_name": data["name"].split(" ")[0],
            "last_name": " ".join(data["name"].split(" ")[1:]),
            "email": data["email"]
        })
        
        # Get schema
        user_v1_class = registry.get_schema("user", 1)
        user_v2_class = registry.get_schema("user", 2)
        
        self.assertEqual(user_v1_class, UserV1)
        self.assertEqual(user_v2_class, UserV2)
        
        # Migrate data
        user_v1_data = {"id": 1, "name": "John Doe", "email": "john@example.com"}
        user_v2_data = registry.migrate(user_v1_data, "user", 1, 2)
        
        self.assertEqual(user_v2_data["id"], 1)
        self.assertEqual(user_v2_data["first_name"], "John")
        self.assertEqual(user_v2_data["last_name"], "Doe")
        self.assertEqual(user_v2_data["email"], "john@example.com")
    
    def test_command_registry(self):
        """Test command registry."""
        # Create registry
        registry = CommandRegistry()
        
        # Register command types
        registry.register(1, "create_task", CreateTaskV1)
        registry.register(1, "delete_task", DeleteTaskV1)
        registry.register(2, "create_task", CreateTaskV2)
        registry.register(2, "delete_task", DeleteTaskV2)
        
        # Parse v1 create task
        v1_create_data = {
            "command_type": "create_task",
            "title": "Complete project",
            "description": "Finish the project by Friday"
        }
        
        v1_create = registry.parse_command(1, v1_create_data)
        self.assertIsInstance(v1_create, CreateTaskV1)
        self.assertEqual(v1_create.title, "Complete project")
        
        # Parse v1 delete task
        v1_delete_data = {
            "command_type": "delete_task",
            "task_id": 123
        }
        
        v1_delete = registry.parse_command(1, v1_delete_data)
        self.assertIsInstance(v1_delete, DeleteTaskV1)
        self.assertEqual(v1_delete.task_id, 123)
        
        # Parse v2 create task
        v2_create_data = {
            "command_type": "create_task",
            "title": "Complete project",
            "description": "Finish the project by Friday",
            "priority": "high",
            "tags": ["work", "urgent"]
        }
        
        v2_create = registry.parse_command(2, v2_create_data)
        self.assertIsInstance(v2_create, CreateTaskV2)
        self.assertEqual(v2_create.title, "Complete project")
        self.assertEqual(v2_create.priority, "high")
        self.assertEqual(v2_create.tags, ["work", "urgent"])
        
        # Parse v2 delete task
        v2_delete_data = {
            "command_type": "delete_task",
            "task_id": 123,
            "soft_delete": True
        }
        
        v2_delete = registry.parse_command(2, v2_delete_data)
        self.assertIsInstance(v2_delete, DeleteTaskV2)
        self.assertEqual(v2_delete.task_id, 123)
        self.assertEqual(v2_delete.soft_delete, True)
        
        # Test unknown command type
        with self.assertRaises(ValueError):
            registry.parse_command(1, {"command_type": "unknown_command"})
        
        # Test missing command type
        with self.assertRaises(ValueError):
            registry.parse_command(1, {"title": "Test"})


if __name__ == "__main__":
    unittest.main()
