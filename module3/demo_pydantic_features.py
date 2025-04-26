"""
Demo for Module 3: Data Validation & Structured Outputs with Pydantic
---------------------------
This script demonstrates the functionality of Pydantic features from Module 3.
"""

import os
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Literal
from enum import Enum

# Import the modules to be demonstrated
# Adjust the import path based on how you're running the script
try:
    # When running from the module3 directory
    from code.pydantic_basics import (
        User, Product, AdvancedUser, SignupForm, TaskInput, process_task_creation
    )
    from code.schema_design import (
        Author, Image, Article, Dog, Cat, Parrot, UserV1, UserV2, UserV3, CommandRegistry,
        CreateTaskV1, DeleteTaskV1, CreateTaskV2, DeleteTaskV2
    )
    from code.output_parsing import (
        Person, ContactForm, PydanticOutputParser, StructuredOutputParser,
        parse_with_pydantic, parse_with_retry, two_stage_parsing,
        simulate_llm_call as simulate_llm_response
    )
    from code.model_composition import (
        BaseItem, Product as ProductModel, User as UserModel,
        BaseContent, Article as ArticleModel, BlogPost, NewsArticle,
        TimestampMixin, VersionMixin, AuditMixin, Document,
        Searchable, SearchableProduct,
        Address, ContactInfo, Education, Person as PersonModel,
        create_address_model, create_user_model,
        create_dynamic_model, create_model_from_schema,
        ModelAdapter, ProductDTO,
        FormDefinition, create_contact_form_definition
    )
    try:
        from code.validation_patterns import (
            DateRange, PasswordReset, PaymentMethod, CreditCardInfo, PayPalInfo, BankTransferInfo,
            Payment, ContentType, Content, BookingRequest, BookingSystem, DynamicModel,
            create_range_validator, create_list_length_validator,
            EducationLevel, EmploymentStatus, EmploymentInfo, Address as AddressModel,
            LoanApplication
        )
    except ImportError:
        validation_patterns_available = False
    else:
        validation_patterns_available = True
except ImportError:
    # When running from the project root
    from module3.code.pydantic_basics import (
        User, Product, AdvancedUser, SignupForm, TaskInput, process_task_creation
    )
    from module3.code.schema_design import (
        Author, Image, Article, Dog, Cat, Parrot, UserV1, UserV2, UserV3, CommandRegistry,
        CreateTaskV1, DeleteTaskV1, CreateTaskV2, DeleteTaskV2
    )
    from module3.code.output_parsing import (
        Person, ContactForm, PydanticOutputParser, StructuredOutputParser,
        parse_with_pydantic, parse_with_retry, two_stage_parsing,
        simulate_llm_call as simulate_llm_response
    )
    from module3.code.model_composition import (
        BaseItem, Product as ProductModel, User as UserModel,
        BaseContent, Article as ArticleModel, BlogPost, NewsArticle,
        TimestampMixin, VersionMixin, AuditMixin, Document,
        Searchable, SearchableProduct,
        Address, ContactInfo, Education, Person as PersonModel,
        create_address_model, create_user_model,
        create_dynamic_model, create_model_from_schema,
        ModelAdapter, ProductDTO,
        FormDefinition, create_contact_form_definition
    )
    try:
        from module3.code.validation_patterns import (
            DateRange, PasswordReset, PaymentMethod, CreditCardInfo, PayPalInfo, BankTransferInfo,
            Payment, ContentType, Content, BookingRequest, BookingSystem, DynamicModel,
            create_range_validator, create_list_length_validator,
            EducationLevel, EmploymentStatus, EmploymentInfo, Address as AddressModel,
            LoanApplication
        )
    except ImportError:
        validation_patterns_available = False
    else:
        validation_patterns_available = True


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def demo_pydantic_basics() -> None:
    """Demonstrate the basic Pydantic features."""
    print_subheader("Pydantic Basics Demo")
    
    # Basic model validation
    print("\n1. Basic Model Validation")
    
    # Valid user
    user = User(id=1, name="John Doe", email="john@example.com")
    print(f"Valid user: {user}")
    
    # Type coercion
    user_with_coercion = User(id="42", name="Jane Doe", email="jane@example.com")
    print(f"User with coerced ID: {user_with_coercion.id} (type: {type(user_with_coercion.id).__name__})")
    
    # Invalid user
    try:
        invalid_user = User(id="not_an_int", name=123, email="invalid_email")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Field constraints
    print("\n2. Field Constraints")
    
    # Valid product
    product = Product(id=1, name="Laptop", price=999.99)
    print(f"Valid product: {product}")
    
    # Invalid product - name too short
    try:
        product = Product(id=2, name="PC", price=1299.99)
    except Exception as e:
        print(f"Validation error (name too short): {e}")
    
    # Invalid product - negative price
    try:
        product = Product(id=3, name="Tablet", price=-199.99)
    except Exception as e:
        print(f"Validation error (negative price): {e}")
    
    # Custom validators
    print("\n3. Custom Validators")
    
    # Valid signup form
    form = SignupForm(
        username="johndoe", 
        password="password123", 
        password_confirm="password123"
    )
    print(f"Valid signup form: {form}")
    
    # Invalid signup form - passwords don't match
    try:
        form = SignupForm(
            username="janedoe", 
            password="password123", 
            password_confirm="password456"
        )
    except Exception as e:
        print(f"Validation error (passwords don't match): {e}")
    
    # Serialization
    print("\n4. Serialization")
    
    # Convert to dictionary
    user_dict = user.model_dump()
    print(f"User as dictionary: {user_dict}")
    
    # Convert to JSON
    user_json = user.model_dump_json()
    print(f"User as JSON: {user_json}")
    
    # Parse from JSON
    json_data = '{"id": 3, "name": "Bob Smith", "email": "bob@example.com", "age": 35}'
    parsed_user = User.model_validate_json(json_data)
    print(f"User parsed from JSON: {parsed_user}")
    
    print("\nPydantic basics provide strong typing and validation for your data models.")


def demo_schema_design() -> None:
    """Demonstrate schema design patterns."""
    print_subheader("Schema Design Demo")
    
    # Nested Models
    print("\n1. Nested Models")
    
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
    print(f"Article with nested models: {article.title} by {article.author.name}")
    print(f"Cover image: {article.cover_image.url} ({article.cover_image.width}x{article.cover_image.height})")
    
    # Discriminated Unions
    print("\n2. Discriminated Unions")
    
    # Create different pet types
    dog = Dog(name="Rex", breed="German Shepherd")
    cat = Cat(name="Whiskers", lives_left=9)
    parrot = Parrot(name="Polly", can_speak=True)
    
    # Process pets
    pets = [dog, cat, parrot]
    for pet in pets:
        if pet.type == "dog":
            print(f"{pet.name} is a {pet.breed} dog")
        elif pet.type == "cat":
            print(f"{pet.name} is a cat with {pet.lives_left} lives left")
        elif pet.type == "parrot":
            speaks = "can speak" if pet.can_speak else "cannot speak"
            print(f"{pet.name} is a parrot that {speaks}")
    
    # Schema Evolution
    print("\n3. Schema Evolution")
    
    # Create v1 user
    user_v1 = UserV1(id=1, name="John Doe", email="john@example.com")
    print(f"UserV1: {user_v1}")
    
    # Migrate to v2
    user_v2 = UserV2.from_v1(user_v1)
    print(f"UserV2 (migrated): {user_v2}")
    
    # Migrate to v3
    user_v3 = UserV3.from_v2(user_v2)
    print(f"UserV3 (migrated): {user_v3}")
    
    # Command Registry
    print("\n4. Command Registry")
    
    # Create registry
    registry = CommandRegistry()
    
    # Register command types
    registry.register(1, "create_task", CreateTaskV1)
    registry.register(1, "delete_task", DeleteTaskV1)
    registry.register(2, "create_task", CreateTaskV2)
    registry.register(2, "delete_task", DeleteTaskV2)
    
    # Parse commands
    v1_data = {
        "command_type": "create_task",
        "title": "Complete demo",
        "description": "Finish the schema design demo"
    }
    v1_command = registry.parse_command(1, v1_data)
    print(f"V1 Command: {v1_command}")
    
    v2_data = {
        "command_type": "create_task",
        "title": "Complete demo",
        "description": "Finish the schema design demo",
        "priority": "high",
        "tags": ["demo", "pydantic"]
    }
    v2_command = registry.parse_command(2, v2_data)
    print(f"V2 Command: {v2_command}")
    
    print("\nSchema design patterns help you create maintainable and evolvable data models.")


def demo_output_parsing() -> None:
    """Demonstrate structured output parsing."""
    print_subheader("Structured Output Parsing Demo")
    
    # Basic Parsing
    print("\n1. Basic Parsing")
    
    parser = PydanticOutputParser(pydantic_object=Person)
    
    print("Parsing clean JSON response:")
    try:
        clean_response = simulate_llm_response("Extract information about a person json")
        person = parser.parse(clean_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    print("\nParsing messy response with embedded JSON:")
    try:
        messy_response = simulate_llm_response("Extract information about a person")
        person = parser.parse(messy_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    # Retry Mechanism
    print("\n2. Retry Mechanism")
    
    print("Parsing with retry:")
    try:
        result = parse_with_retry(
            "Extract information about a person with missing fields",
            Person
        )
        print(f"Successfully parsed with retry: {result}")
    except Exception as e:
        print(f"Parsing with retry failed: {e}")
    
    # Two-Stage Parsing
    print("\n3. Two-Stage Parsing")
    
    print("Two-stage parsing:")
    try:
        result = two_stage_parsing(
            "Extract information about a person with incorrect format",
            Person
        )
        print(f"Successfully parsed with two stages: {result}")
    except Exception as e:
        print(f"Two-stage parsing failed: {e}")
    
    print("\nStructured output parsing helps you reliably extract structured data from LLM outputs.")


def demo_model_composition() -> None:
    """Demonstrate model composition patterns."""
    print_subheader("Model Composition Demo")
    
    # Inheritance
    print("\n1. Inheritance")
    
    product = ProductModel(id=1, name="Laptop", price=999.99, category="Electronics")
    user = UserModel(id=1, name="John Doe", email="john@example.com", is_active=True)
    
    print(f"Product: {product}")
    print(f"User: {user}")
    
    # Multi-level inheritance
    print("\n2. Multi-level Inheritance")
    
    blog_post = BlogPost(
        id=1,
        title="Advanced Pydantic",
        content="This is a blog post about Pydantic...",
        author="Jane Smith",
        tags=["python", "pydantic"],
        comments_count=5
    )
    
    news_article = NewsArticle(
        id=2,
        title="Python 4.0 Released",
        content="Python 4.0 has been released with exciting new features...",
        author="John Doe",
        source="Python Blog",
        category="Technology"
    )
    
    print(f"Blog Post: {blog_post.title} by {blog_post.author}")
    print(f"News Article: {news_article.title} from {news_article.source}")
    
    # Mixins
    print("\n3. Mixins")
    
    document = Document(
        id=1,
        title="Important Document",
        content="This is an important document...",
        created_by="admin"
    )
    
    print(f"Document: {document.title}")
    print(f"Created at: {document.created_at}")
    print(f"Version: {document.version}")
    print(f"Audit trail: {document.audit_trail}")
    
    # Composition
    print("\n4. Composition")
    
    address = Address(
        street="123 Main St",
        city="Anytown",
        state="CA",
        zip_code="12345",
        country="USA"
    )
    
    contact_info = ContactInfo(
        email="john@example.com",
        phone="555-123-4567",
        address=address
    )
    
    education = Education(
        institution="University of Example",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        graduation_year=2020
    )
    
    person = PersonModel(
        name="John Doe",
        age=30,
        contact_info=contact_info,
        education=[education]
    )
    
    print(f"Person: {person.name}, {person.age}")
    print(f"Contact: {person.contact_info.email}, {person.contact_info.phone}")
    print(f"Address: {person.contact_info.address.street}, {person.contact_info.address.city}")
    print(f"Education: {person.education[0].degree} in {person.education[0].field_of_study}")
    
    # Dynamic Models
    print("\n5. Dynamic Models")
    
    # Create a dynamic model
    fields = {
        "name": (str, ...),
        "age": (int, Field(gt=0)),
        "is_active": (bool, True)
    }
    DynamicUser = create_dynamic_model("DynamicUser", fields)
    
    dynamic_user = DynamicUser(name="Jane Doe", age=25)
    print(f"Dynamic User: {dynamic_user}")
    
    # Create a model from schema
    schema = {
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "quantity": {"type": "integer", "minimum": 1},
            "price": {"type": "number", "minimum": 0}
        },
        "required": ["product_id", "quantity", "price"]
    }
    OrderItem = create_model_from_schema("OrderItem", schema)
    
    order_item = OrderItem(product_id="PROD-123", quantity=2, price=29.99)
    print(f"Order Item: {order_item}")
    
    print("\nModel composition patterns help you create flexible and reusable data models.")


def demo_validation_patterns() -> None:
    """Demonstrate advanced validation patterns."""
    print_subheader("Advanced Validation Patterns Demo")
    
    if not validation_patterns_available:
        print("\nValidation patterns module is not available.")
        return
    
    # Cross-field validation
    print("\n1. Cross-Field Validation")
    
    # Valid date range
    try:
        date_range = DateRange(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        print(f"Valid date range: {date_range}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Invalid date range (end before start)
    try:
        date_range = DateRange(
            start_date=date(2023, 12, 31),
            end_date=date(2023, 1, 1)
        )
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Password reset with matching passwords
    try:
        reset = PasswordReset(
            email="user@example.com",
            new_password="securePassword123",
            confirm_password="securePassword123"
        )
        print(f"Valid password reset: {reset}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Polymorphic validation
    print("\n2. Polymorphic Validation")
    
    # Credit card payment
    try:
        payment = Payment(
            amount=99.99,
            currency="USD",
            payment_method=PaymentMethod.CREDIT_CARD,
            credit_card=CreditCardInfo(
                card_number="4111111111111111",
                expiry_month=12,
                expiry_year=2025,
                cvv="123"
            )
        )
        print(f"Valid credit card payment: {payment}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # PayPal payment
    try:
        payment = Payment(
            amount=49.99,
            currency="USD",
            payment_method=PaymentMethod.PAYPAL,
            paypal=PayPalInfo(
                email="user@example.com"
            )
        )
        print(f"Valid PayPal payment: {payment}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Content-based validation
    print("\n3. Content-Based Validation")
    
    # Valid text content
    try:
        content = Content(
            content_type=ContentType.TEXT,
            title="My Text",
            text_content="This is some text content."
        )
        print(f"Valid text content: {content}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Valid image content
    try:
        content = Content(
            content_type=ContentType.IMAGE,
            title="My Image",
            file_path="image.jpg"
        )
        print(f"Valid image content: {content}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Dynamic validation
    print("\n4. Dynamic Validation")
    
    # Create a range validator
    validate_age = create_range_validator("age", 18, 65)
    
    # Create a model with the dynamic validator
    class UserWithDynamicValidation(DynamicModel):
        name: str
        age: int
        
        # Add the dynamic validator
        _validators = [validate_age]
    
    # Valid user
    try:
        user = UserWithDynamicValidation(name="John", age=30)
        print(f"Valid user with dynamic validation: {user}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Invalid user (age out of range)
    try:
        user = UserWithDynamicValidation(name="Young User", age=16)
    except Exception as e:
        print(f"Validation error: {e}")
    
    print("\nAdvanced validation patterns help you implement complex business rules in your data models.")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    print_header("DEMONSTRATION OF MODULE 3: DATA VALIDATION & STRUCTURED OUTPUTS WITH PYDANTIC")
    print("This script demonstrates the functionality of Pydantic features.")
    print("Follow along to see how each component works and how they can be used together.")
    
    # Run each demo with a pause between
    demo_pydantic_basics()
    time.sleep(1)
    
    demo_schema_design()
    time.sleep(1)
    
    demo_output_parsing()
    time.sleep(1)
    
    demo_model_composition()
    time.sleep(1)
    
    if validation_patterns_available:
        demo_validation_patterns()
    
    print_header("DEMONSTRATION COMPLETE")
    print("You've now seen all the key features of Pydantic for data validation and structured outputs.")
    print("For more details, refer to the documentation and source code.")


def main():
    """Main function to run the demo with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Module 3 Pydantic features")
    parser.add_argument("component", nargs="?", choices=[
        "basics", "schema", "parsing", "composition", "validation", "all"
    ], default="all", help="Component to demonstrate")
    
    args = parser.parse_args()
    
    if args.component == "basics":
        print_header("PYDANTIC BASICS DEMO")
        demo_pydantic_basics()
    elif args.component == "schema":
        print_header("SCHEMA DESIGN DEMO")
        demo_schema_design()
    elif args.component == "parsing":
        print_header("OUTPUT PARSING DEMO")
        demo_output_parsing()
    elif args.component == "composition":
        print_header("MODEL COMPOSITION DEMO")
        demo_model_composition()
    elif args.component == "validation" and validation_patterns_available:
        print_header("ADVANCED VALIDATION PATTERNS DEMO")
        demo_validation_patterns()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
