"""
Tests for the model_composition module.

This module contains tests for the various model composition patterns
implemented in the model_composition module.
"""

import unittest
from datetime import datetime, date, timedelta
from model_composition import (
    # Inheritance
    BaseItem, Product, User,
    # Multi-level inheritance
    BaseContent, Article, BlogPost, NewsArticle,
    # Mixins
    TimestampMixin, VersionMixin, AuditMixin, Document,
    # Abstract base models
    Searchable, SearchableProduct,
    # Composition
    Address, ContactInfo, Education, Person,
    # Shared fields
    UserWithSharedFields, EmployeeWithSharedFields,
    # Factories
    create_address_model, create_user_model,
    # Dynamic models
    create_dynamic_model, create_model_from_schema,
    # Transformation
    UserInput, UserDB, UserOutput, create_user_db, create_user_output,
    # Adapters
    ModelAdapter, ProductDTO,
    # Form system
    FormField, StringField, NumberField, BooleanField, DateField, SelectField,
    FormDefinition, create_contact_form_definition
)


class TestInheritancePatterns(unittest.TestCase):
    """Test inheritance patterns."""
    
    def test_basic_inheritance(self):
        """Test basic inheritance with BaseItem."""
        # Create a product
        product = Product(id=1, name="Laptop", price=999.99)
        self.assertEqual(product.id, 1)
        self.assertEqual(product.name, "Laptop")
        self.assertEqual(product.price, 999.99)
        self.assertIsInstance(product.created_at, datetime)
        self.assertIsNone(product.updated_at)
        self.assertIsNone(product.description)
        
        # Create a user
        user = User(id=2, username="johndoe", email="john@example.com")
        self.assertEqual(user.id, 2)
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertTrue(user.is_active)
        self.assertIsInstance(user.created_at, datetime)
    
    def test_multi_level_inheritance(self):
        """Test multi-level inheritance with content models."""
        # Create a blog post
        blog_post = BlogPost(
            title="My First Post",
            body="This is the content of my first post.",
            author="John Doe",
            tags=["python", "pydantic"]
        )
        self.assertEqual(blog_post.title, "My First Post")
        self.assertEqual(blog_post.body, "This is the content of my first post.")
        self.assertEqual(blog_post.author, "John Doe")
        self.assertEqual(blog_post.tags, ["python", "pydantic"])
        self.assertTrue(blog_post.comments_enabled)
        
        # Create a news article
        news_article = NewsArticle(
            title="Breaking News",
            body="Something important happened.",
            author="Jane Smith",
            source="CNN",
            breaking=True
        )
        self.assertEqual(news_article.title, "Breaking News")
        self.assertEqual(news_article.source, "CNN")
        self.assertTrue(news_article.breaking)


class TestMixinPatterns(unittest.TestCase):
    """Test mixin patterns."""
    
    def test_mixins(self):
        """Test using multiple mixins."""
        # Create a document
        doc = Document(title="Important Document", content="This is important.")
        self.assertEqual(doc.title, "Important Document")
        self.assertEqual(doc.content, "This is important.")
        self.assertIsInstance(doc.created_at, datetime)
        self.assertIsNone(doc.updated_at)
        self.assertEqual(doc.version, "1.0")
        self.assertIsNone(doc.created_by)
        self.assertIsNone(doc.last_modified_by)
        
        # Test mixin methods
        self.assertEqual(doc.get_latest_version(), "2.5")
        doc.record_modification("John Doe")
        self.assertEqual(doc.last_modified_by, "John Doe")


class TestAbstractBaseModels(unittest.TestCase):
    """Test abstract base models."""
    
    def test_searchable(self):
        """Test the Searchable abstract base model."""
        # Create a searchable product
        product = SearchableProduct(
            id=1,
            name="Laptop",
            description="Powerful laptop for developers",
            price=999.99
        )
        
        # Test search text generation
        search_text = product.get_search_text()
        self.assertEqual(search_text, "Laptop Powerful laptop for developers")
        
        # Test with missing description
        product2 = SearchableProduct(id=2, name="Mouse", price=29.99)
        search_text2 = product2.get_search_text()
        self.assertEqual(search_text2, "Mouse")


class TestCompositionPatterns(unittest.TestCase):
    """Test composition patterns."""
    
    def test_nested_models(self):
        """Test nested model composition."""
        # Create a person with nested models
        person = Person(
            name="John Doe",
            birth_date=date(1990, 1, 15),
            contact=ContactInfo(
                email="john@example.com",
                phone="555-1234",
                address=Address(
                    street="123 Main St",
                    city="Anytown",
                    state="CA",
                    zip_code="12345",
                    country="USA"
                )
            ),
            education=[
                Education(
                    institution="University of Example",
                    degree="Bachelor of Science",
                    graduation_date=date(2012, 5, 15)
                )
            ]
        )
        
        # Test nested access
        self.assertEqual(person.name, "John Doe")
        self.assertEqual(person.contact.email, "john@example.com")
        self.assertEqual(person.contact.address.city, "Anytown")
        self.assertEqual(person.education[0].degree, "Bachelor of Science")


class TestSharedFieldDefinitions(unittest.TestCase):
    """Test shared field definitions."""
    
    def test_shared_fields(self):
        """Test models with shared field definitions."""
        # Create a user with shared fields
        user = UserWithSharedFields(
            name="John Doe",
            email="john@example.com",
            username="johndoe",
            age=30
        )
        self.assertEqual(user.name, "John Doe")
        self.assertEqual(user.email, "john@example.com")
        
        # Test validation
        with self.assertRaises(ValueError):
            UserWithSharedFields(
                name="J",  # Too short
                email="john@example.com",
                username="johndoe",
                age=30
            )
        
        with self.assertRaises(ValueError):
            UserWithSharedFields(
                name="John Doe",
                email="invalid-email",  # Invalid format
                username="johndoe",
                age=30
            )
        
        with self.assertRaises(ValueError):
            UserWithSharedFields(
                name="John Doe",
                email="john@example.com",
                username="john@doe",  # Invalid characters
                age=30
            )


class TestFactoryPatterns(unittest.TestCase):
    """Test factory patterns."""
    
    def test_model_factories(self):
        """Test model factory functions."""
        # Create models with factories
        BasicAddress = create_address_model(country_specific=False)
        FullAddress = create_address_model(country_specific=True)
        
        # Test basic address
        basic_address = BasicAddress(
            street="123 Main St",
            city="Anytown",
            zip_code="12345"
        )
        self.assertEqual(basic_address.street, "123 Main St")
        self.assertEqual(basic_address.city, "Anytown")
        
        # Test full address
        full_address = FullAddress(
            street="123 Main St",
            city="Anytown",
            zip_code="12345",
            country="USA",
            state="CA"
        )
        self.assertEqual(full_address.country, "USA")
        self.assertEqual(full_address.state, "CA")
        
        # Test user models
        BasicUser = create_user_model(with_address=False)
        FullUser = create_user_model(with_address=True, with_payment=True)
        
        # Test basic user
        basic_user = BasicUser(
            name="John Doe",
            email="john@example.com",
            username="johndoe"
        )
        self.assertEqual(basic_user.name, "John Doe")
        
        # Test full user
        full_user = FullUser(
            name="Jane Smith",
            email="jane@example.com",
            username="janesmith",
            address={
                "street": "456 Oak St",
                "city": "Othertown",
                "zip_code": "67890",
                "country": "USA",
                "state": "NY"
            },
            payment_methods=[{"type": "credit_card", "last4": "1234"}]
        )
        self.assertEqual(full_user.name, "Jane Smith")
        self.assertEqual(full_user.address.street, "456 Oak St")
        self.assertEqual(full_user.payment_methods[0]["type"], "credit_card")


class TestDynamicModelGeneration(unittest.TestCase):
    """Test dynamic model generation."""
    
    def test_dynamic_model_creation(self):
        """Test creating models dynamically."""
        # Create a dynamic model
        user_fields = {
            "name": {"type": str, "min_length": 2},
            "email": {"type": str, "pattern": r"[^@]+@[^@]+\.[^@]+"},
            "age": {"type": int, "ge": 18, "optional": True},
            "tags": {"type": list, "optional": True, "default": []}
        }
        
        UserModel = create_dynamic_model("User", user_fields)
        
        # Test the model
        user = UserModel(name="John Doe", email="john@example.com")
        self.assertEqual(user.name, "John Doe")
        self.assertEqual(user.email, "john@example.com")
        self.assertIsNone(user.age)
        self.assertEqual(user.tags, [])
        
        # Test with all fields
        user2 = UserModel(
            name="Jane Smith",
            email="jane@example.com",
            age=25,
            tags=["developer", "python"]
        )
        self.assertEqual(user2.age, 25)
        self.assertEqual(user2.tags, ["developer", "python"])
        
        # Test validation
        with self.assertRaises(ValueError):
            UserModel(name="J", email="john@example.com")  # Name too short
        
        with self.assertRaises(ValueError):
            UserModel(name="John", email="invalid-email")  # Invalid email
    
    def test_schema_driven_models(self):
        """Test creating models from schemas."""
        # Create a model from schema
        user_schema = {
            "title": "User",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 100
                },
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["name", "email"]
        }
        
        UserModel = create_model_from_schema(user_schema)
        
        # Test the model
        user = UserModel(name="John Doe", email="john@example.com", age=30)
        self.assertEqual(user.name, "John Doe")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(user.age, 30)


class TestModelTransformation(unittest.TestCase):
    """Test model transformation patterns."""
    
    def test_model_conversion(self):
        """Test converting between related models."""
        # Create input model
        user_input = UserInput(
            name="John Doe",
            email="john@example.com",
            password="secret123",
            age=30
        )
        
        # Convert to DB model
        user_db = create_user_db(user_input, user_id=1)
        self.assertEqual(user_db.id, 1)
        self.assertEqual(user_db.name, "John Doe")
        self.assertEqual(user_db.email, "john@example.com")
        self.assertEqual(user_db.hashed_password, "hashed_secret123")
        self.assertEqual(user_db.age, 30)
        
        # Convert to output model
        user_output = create_user_output(user_db)
        self.assertEqual(user_output.id, 1)
        self.assertEqual(user_output.name, "John Doe")
        self.assertEqual(user_output.email, "john@example.com")
        self.assertEqual(user_output.age, 30)
        self.assertIsInstance(user_output.created_at, datetime)
        
        # Ensure password is not in output
        with self.assertRaises(AttributeError):
            user_output.hashed_password


class TestModelAdapters(unittest.TestCase):
    """Test model adapter patterns."""
    
    def test_model_adapter(self):
        """Test the ModelAdapter for converting between models."""
        # Define product model
        class Product(BaseItem):
            name: str
            price: float
            description: str
        
        # Create adapter
        product_adapter = ModelAdapter(
            Product, 
            ProductDTO, 
            field_mapping={
                "product_id": "id",
                "title": "name",
                "details": "description"
            }
        )
        
        # Create product
        product = Product(
            id=1,
            name="Laptop",
            price=999.99,
            description="Powerful laptop"
        )
        
        # Adapt to DTO
        product_dto = product_adapter.adapt(product)
        
        # Test adapter results
        self.assertEqual(product_dto.product_id, 1)
        self.assertEqual(product_dto.title, "Laptop")
        self.assertEqual(product_dto.price, 999.99)
        self.assertEqual(product_dto.details, "Powerful laptop")


class TestFormSystem(unittest.TestCase):
    """Test the form system with model composition."""
    
    def test_form_definition(self):
        """Test creating a form definition."""
        # Create form definition
        form_def = create_contact_form_definition()
        self.assertEqual(form_def.title, "Contact Form")
        self.assertEqual(len(form_def.fields), 5)
        
        # Check field types
        self.assertIsInstance(form_def.fields[0], StringField)
        self.assertIsInstance(form_def.fields[4], BooleanField)
        
        # Check field properties
        self.assertEqual(form_def.fields[0].name, "name")
        self.assertEqual(form_def.fields[0].min_length, 2)
        self.assertEqual(form_def.fields[1].pattern, r"[^@]+@[^@]+\.[^@]+")
    
    def test_form_model_generation(self):
        """Test generating a model from a form definition."""
        # Create form definition
        form_def = create_contact_form_definition()
        
        # Generate model
        ContactForm = form_def.create_model()
        
        # Test valid data
        form_data = ContactForm(
            name="John Doe",
            email="john@example.com",
            subject="Question about your service",
            message="I would like to know more about your services.",
            subscribe=True
        )
        self.assertEqual(form_data.name, "John Doe")
        self.assertEqual(form_data.email, "john@example.com")
        self.assertTrue(form_data.subscribe)
        
        # Test validation
        with self.assertRaises(ValueError):
            ContactForm(
                name="J",  # Too short
                email="john@example.com",
                subject="Question",
                message="Hello"  # Too short
            )
        
        with self.assertRaises(ValueError):
            ContactForm(
                name="John Doe",
                email="invalid-email",  # Invalid format
                subject="Question about your service",
                message="I would like to know more about your services."
            )


if __name__ == "__main__":
    unittest.main()
