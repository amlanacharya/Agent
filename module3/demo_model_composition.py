"""
Demo script for Advanced Model Composition with Pydantic.

This script demonstrates the key concepts of model composition patterns
in Pydantic, including inheritance, mixins, nested models, and dynamic
model generation.
"""

import sys
import os
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Type

# Add the module3 directory to the path so we can import the code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.code.model_composition import (
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
    # Factories
    create_address_model, create_user_model,
    # Dynamic models
    create_dynamic_model, create_model_from_schema,
    # Adapters
    ModelAdapter, ProductDTO,
    # Form system
    FormDefinition, create_contact_form_definition
)


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """Print a subsection title."""
    print("\n" + "-" * 40)
    print(f"  {title}")
    print("-" * 40)


def demo_inheritance():
    """Demonstrate inheritance patterns."""
    print_section("Inheritance Patterns")
    
    print_subsection("Basic Inheritance")
    # Create a product
    product = Product(id=1, name="Laptop", price=999.99, description="Powerful laptop")
    print(f"Product: {product.model_dump_json(indent=2)}")
    
    # Create a user
    user = User(id=2, username="johndoe", email="john@example.com")
    print(f"User: {user.model_dump_json(indent=2)}")
    
    print_subsection("Multi-Level Inheritance")
    # Create a blog post
    blog_post = BlogPost(
        title="My First Post",
        body="This is the content of my first post.",
        author="John Doe",
        tags=["python", "pydantic"]
    )
    print(f"BlogPost: {blog_post.model_dump_json(indent=2)}")
    
    # Create a news article
    news_article = NewsArticle(
        title="Breaking News",
        body="Something important happened.",
        author="Jane Smith",
        source="CNN",
        breaking=True
    )
    print(f"NewsArticle: {news_article.model_dump_json(indent=2)}")


def demo_mixins():
    """Demonstrate mixin patterns."""
    print_section("Mixin Patterns")
    
    # Create a document
    doc = Document(title="Important Document", content="This is important.")
    print(f"Document: {doc.model_dump_json(indent=2)}")
    
    # Demonstrate mixin methods
    print(f"\nLatest version: {doc.get_latest_version()}")
    
    # Record a modification
    doc.record_modification("John Doe")
    print(f"After modification: {doc.model_dump_json(indent=2)}")


def demo_abstract_base_models():
    """Demonstrate abstract base models."""
    print_section("Abstract Base Models")
    
    # Create a searchable product
    product = SearchableProduct(
        id=1,
        name="Laptop",
        description="Powerful laptop for developers",
        price=999.99
    )
    
    # Get search text
    search_text = product.get_search_text()
    print(f"Product: {product.model_dump_json(indent=2)}")
    print(f"Search text: {search_text}")


def demo_composition():
    """Demonstrate composition patterns."""
    print_section("Composition Patterns")
    
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
            ),
            Education(
                institution="Example Tech",
                degree="Master of Computer Science",
                graduation_date=date(2014, 5, 20)
            )
        ]
    )
    
    print(f"Person: {person.model_dump_json(indent=2)}")
    
    # Access nested data
    print("\nAccessing nested data:")
    print(f"Name: {person.name}")
    print(f"Email: {person.contact.email}")
    print(f"City: {person.contact.address.city}")
    print(f"First degree: {person.education[0].degree}")
    print(f"Second institution: {person.education[1].institution}")


def demo_factories():
    """Demonstrate factory patterns."""
    print_section("Factory Patterns")
    
    print_subsection("Address Factory")
    # Create address models
    BasicAddress = create_address_model(country_specific=False)
    FullAddress = create_address_model(country_specific=True)
    
    # Create addresses
    basic_address = BasicAddress(
        street="123 Main St",
        city="Anytown",
        zip_code="12345"
    )
    
    full_address = FullAddress(
        street="456 Oak St",
        city="Othertown",
        zip_code="67890",
        country="USA",
        state="NY"
    )
    
    print(f"Basic Address: {basic_address.model_dump_json(indent=2)}")
    print(f"Full Address: {full_address.model_dump_json(indent=2)}")
    
    print_subsection("User Factory")
    # Create user models
    BasicUser = create_user_model(with_address=False)
    FullUser = create_user_model(with_address=True, with_payment=True)
    
    # Create users
    basic_user = BasicUser(
        name="John Doe",
        email="john@example.com",
        username="johndoe"
    )
    
    full_user = FullUser(
        name="Jane Smith",
        email="jane@example.com",
        username="janesmith",
        address={
            "street": "789 Pine St",
            "city": "Somewhere",
            "zip_code": "54321",
            "country": "USA",
            "state": "CA"
        },
        payment_methods=[
            {"type": "credit_card", "last4": "1234"},
            {"type": "paypal", "email": "jane@example.com"}
        ]
    )
    
    print(f"Basic User: {basic_user.model_dump_json(indent=2)}")
    print(f"Full User: {full_user.model_dump_json(indent=2)}")


def demo_dynamic_models():
    """Demonstrate dynamic model generation."""
    print_section("Dynamic Model Generation")
    
    print_subsection("Field Configuration")
    # Create a dynamic model
    user_fields = {
        "name": {"type": str, "min_length": 2},
        "email": {"type": str, "pattern": r"[^@]+@[^@]+\.[^@]+"},
        "age": {"type": int, "ge": 18, "optional": True},
        "tags": {"type": list, "optional": True, "default": []}
    }
    
    UserModel = create_dynamic_model("User", user_fields)
    
    # Create users
    user1 = UserModel(name="John Doe", email="john@example.com")
    user2 = UserModel(
        name="Jane Smith",
        email="jane@example.com",
        age=25,
        tags=["developer", "python"]
    )
    
    print(f"User 1: {user1.model_dump_json(indent=2)}")
    print(f"User 2: {user2.model_dump_json(indent=2)}")
    
    print_subsection("JSON Schema")
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
    
    SchemaUser = create_model_from_schema(user_schema)
    
    # Create a user
    schema_user = SchemaUser(
        name="Alice Johnson",
        email="alice@example.com",
        age=30,
        tags=["manager", "sales"]
    )
    
    print(f"Schema User: {schema_user.model_dump_json(indent=2)}")


def demo_adapters():
    """Demonstrate model adapters."""
    print_section("Model Adapters")
    
    # Define a product model
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
    
    print(f"Original Product: {product.model_dump_json(indent=2)}")
    print(f"Product DTO: {product_dto.model_dump_json(indent=2)}")


def demo_form_system():
    """Demonstrate the form system."""
    print_section("Form System")
    
    # Create form definition
    form_def = create_contact_form_definition()
    print("Form Definition:")
    print(f"Title: {form_def.title}")
    print("Fields:")
    for i, field in enumerate(form_def.fields):
        print(f"  {i+1}. {field.name} ({field.__class__.__name__}): {field.label}")
        if hasattr(field, "min_length") and field.min_length:
            print(f"     Min Length: {field.min_length}")
        if hasattr(field, "pattern") and field.pattern:
            print(f"     Pattern: {field.pattern}")
    
    # Generate model
    ContactForm = form_def.create_model()
    
    # Create form data
    try:
        form_data = ContactForm(
            name="John Doe",
            email="john@example.com",
            subject="Question about your service",
            message="I would like to know more about your services.",
            subscribe=True
        )
        print("\nValid form data:")
        print(form_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nValidation error: {e}")
    
    # Try invalid data
    print("\nTrying invalid data:")
    try:
        invalid_form = ContactForm(
            name="J",  # Too short
            email="invalid-email",  # Invalid format
            subject="Q",  # Too short
            message="Hi",  # Too short
            subscribe=True
        )
    except Exception as e:
        print(f"Validation error: {e}")


def main():
    """Run all demos."""
    print("=" * 80)
    print("  ADVANCED MODEL COMPOSITION WITH PYDANTIC")
    print("=" * 80)
    print("\nThis demo showcases various model composition patterns in Pydantic.")
    
    demo_inheritance()
    demo_mixins()
    demo_abstract_base_models()
    demo_composition()
    demo_factories()
    demo_dynamic_models()
    demo_adapters()
    demo_form_system()
    
    print("\n" + "=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
