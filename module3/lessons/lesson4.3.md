# 🧩 Module 3: Structured Data Validation - Lesson 4.3: Advanced Model Composition 🏗️

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔄 Master advanced model composition patterns in Pydantic
- 🧱 Build complex models from simpler components
- 🔌 Create reusable validation logic across multiple models
- 🌐 Implement dynamic model generation techniques
- 🔄 Apply model transformation patterns for different contexts

---

## 📚 Introduction to Advanced Model Composition

As your data models grow in complexity, you'll need strategies to organize, reuse, and extend them. Model composition techniques help you build maintainable, flexible validation systems that can evolve with your application's needs.

## 🧩 The Power of Composition

Model composition allows you to:

1. **Reuse validation logic** across multiple models
2. **Build complex models** from simpler components
3. **Create specialized models** for different contexts
4. **Evolve schemas** without breaking existing code
5. **Generate models dynamically** based on runtime conditions

> 💡 **Key Insight**: Composition is a powerful design principle that helps manage complexity by breaking down large models into smaller, reusable components.

---

## 🔄 Inheritance Patterns

### 🌱 Basic Inheritance

Pydantic models can inherit from other models:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class BaseItem(BaseModel):
    id: int
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class Product(BaseItem):
    name: str
    price: float
    description: Optional[str] = None

class User(BaseItem):
    username: str
    email: str
    is_active: bool = True
```

### 🌲 Multi-Level Inheritance

You can create deeper inheritance hierarchies:

```python
class BaseContent(BaseModel):
    title: str
    created_at: datetime = Field(default_factory=datetime.now)

class Article(BaseContent):
    body: str
    author: str

class BlogPost(Article):
    tags: list[str] = []
    comments_enabled: bool = True

class NewsArticle(Article):
    source: str
    breaking: bool = False
```

### 🧬 Mixin Classes

Mixins provide reusable functionality across different model hierarchies:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, ClassVar, Dict, Any

class TimestampMixin(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class VersionMixin(BaseModel):
    version: str = "1.0"

    @classmethod
    def get_latest_version(cls) -> str:
        return cls.model_config.get("latest_version", cls.version)

    model_config = {"latest_version": "1.0"}

class AuditMixin(BaseModel):
    created_by: Optional[str] = None
    last_modified_by: Optional[str] = None

    def record_modification(self, user: str):
        self.last_modified_by = user

# Using mixins
class Document(TimestampMixin, VersionMixin, AuditMixin):
    title: str
    content: str

    model_config = {"latest_version": "2.5"}
```

### 🏛️ Abstract Base Models

Create abstract base models that define interfaces:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, ClassVar, List
from abc import ABC
from datetime import datetime

class Searchable(BaseModel, ABC):
    """Abstract base model for searchable items."""
    search_fields: ClassVar[List[str]] = []

    def get_search_text(self) -> str:
        """Get text for search indexing."""
        values = []
        for field in self.__class__.search_fields:
            if hasattr(self, field):
                value = getattr(self, field)
                if value:
                    values.append(str(value))
        return " ".join(values)

class Product(Searchable):
    id: int
    name: str
    description: Optional[str] = None
    price: float

    search_fields = ["name", "description"]

# Usage
product = Product(id=1, name="Laptop", description="Powerful laptop", price=999.99)
search_text = product.get_search_text()  # "Laptop Powerful laptop"
```

---

## 🧩 Composition Over Inheritance

### 📦 Nested Models

Compose models by nesting them:

```python
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import date

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class ContactInfo(BaseModel):
    email: EmailStr
    phone: Optional[str] = None
    address: Address

class Education(BaseModel):
    institution: str
    degree: str
    graduation_date: date

class Person(BaseModel):
    name: str
    birth_date: date
    contact: ContactInfo
    education: List[Education] = []
```

### 🔄 Reusing Field Definitions

Create reusable field definitions:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
import re

# Reusable field definitions
name_field = Field(..., min_length=2, max_length=100)
email_field = Field(..., pattern=r"[^@]+@[^@]+\.[^@]+")
password_field = Field(..., min_length=8, max_length=100)
age_field = Field(..., ge=0, le=120)

# Reusable validators
def validate_username(cls, v: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_]+$', v):
        raise ValueError("Username can only contain letters, numbers, and underscores")
    return v

# Models using shared definitions
class User(BaseModel):
    name: str = name_field
    email: str = email_field
    username: str = Field(..., min_length=3, max_length=20)
    age: int = age_field

    _validate_username = field_validator('username')(validate_username)

class Employee(BaseModel):
    name: str = name_field
    email: str = email_field
    employee_id: str
    department: str
    manager_name: Optional[str] = name_field.get_default()
```

### 🏭 Composition with Factories

Create model factories for flexible composition:

```python
from pydantic import BaseModel, create_model, Field
from typing import Dict, Any, Type, Optional, List

def create_address_model(country_specific: bool = False) -> Type[BaseModel]:
    """Create an address model with optional country-specific fields."""
    fields = {
        "street": (str, ...),
        "city": (str, ...),
        "zip_code": (str, ...)
    }

    if country_specific:
        fields.update({
            "state": (Optional[str], None),
            "province": (Optional[str], None),
            "country": (str, ...)
        })

    return create_model("Address", **fields)

def create_user_model(with_address: bool = True, with_payment: bool = False) -> Type[BaseModel]:
    """Create a user model with optional components."""
    fields = {
        "name": (str, Field(..., min_length=2)),
        "email": (str, ...),
        "username": (str, Field(..., min_length=3))
    }

    if with_address:
        Address = create_address_model(country_specific=True)
        fields["address"] = (Address, ...)

    if with_payment:
        fields["payment_methods"] = (List[Dict[str, Any]], [])

    return create_model("User", **fields)

# Usage
BasicUser = create_user_model(with_address=False)
FullUser = create_user_model(with_address=True, with_payment=True)

basic_user = BasicUser(name="John Doe", email="john@example.com", username="johndoe")
full_user = FullUser(
    name="Jane Smith",
    email="jane@example.com",
    username="janesmith",
    address={
        "street": "123 Main St",
        "city": "Anytown",
        "zip_code": "12345",
        "country": "USA",
        "state": "CA"
    }
)
```

---

## 🔄 Dynamic Model Generation

### ⚡ Creating Models at Runtime

Generate models dynamically based on runtime conditions:

```python
from pydantic import BaseModel, create_model, Field
from typing import Dict, Any, Type, Optional, List, get_type_hints

def create_dynamic_model(name: str, fields_config: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
    """Create a Pydantic model dynamically from a field configuration."""
    fields = {}

    for field_name, config in fields_config.items():
        field_type = config.get("type", str)

        # Handle optional fields
        if config.get("optional", False):
            field_type = Optional[field_type]

        # Get default value
        default = ... if not config.get("optional", False) else config.get("default", None)

        # Add field constraints
        constraints = {k: v for k, v in config.items()
                      if k not in ["type", "optional", "default"]}

        if constraints:
            field_def = (field_type, Field(default, **constraints))
        else:
            field_def = (field_type, default)

        fields[field_name] = field_def

    return create_model(name, **fields)

# Usage
user_fields = {
    "name": {"type": str, "min_length": 2},
    "email": {"type": str, "pattern": r"[^@]+@[^@]+\.[^@]+"},
    "age": {"type": int, "ge": 18, "optional": True},
    "tags": {"type": List[str], "optional": True, "default": []}
}

UserModel = create_dynamic_model("User", user_fields)
user = UserModel(name="John Doe", email="john@example.com")
print(user)
```

### 📝 Schema-Driven Models

Generate models from external schemas:

```python
from pydantic import BaseModel, create_model, Field
from typing import Dict, Any, Type, Optional, List, Union, get_type_hints

def type_from_schema(schema_type: str) -> Type:
    """Convert schema type string to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict
    }
    return type_mapping.get(schema_type, Any)

def create_model_from_schema(schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON Schema-like definition."""
    model_name = schema.get("title", "DynamicModel")
    fields = {}

    for prop_name, prop_schema in schema.get("properties", {}).items():
        prop_type = type_from_schema(prop_schema.get("type", "string"))

        # Handle arrays
        if prop_type == List and "items" in prop_schema:
            item_type = type_from_schema(prop_schema["items"].get("type", "string"))
            prop_type = List[item_type]

        # Handle required fields
        is_required = prop_name in schema.get("required", [])
        default = ... if is_required else None

        # Add field constraints
        constraints = {k: v for k, v in prop_schema.items()
                      if k not in ["type", "items"]}

        if constraints:
            field_def = (prop_type, Field(default, **constraints))
        else:
            field_def = (prop_type, default)

        fields[prop_name] = field_def

    return create_model(model_name, **fields)

# Example schema
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
user = UserModel(name="John Doe", email="john@example.com", age=30)
print(user)
```

---

## 🔄 Model Transformation Patterns

### 🔄 Model Conversion

Convert between related models:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Input model (from API)
class UserInput(BaseModel):
    name: str
    email: str
    password: str
    age: Optional[int] = None

# Database model
class UserDB(BaseModel):
    id: int
    name: str
    email: str
    hashed_password: str
    age: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Output model (to API)
class UserOutput(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    created_at: datetime

# Conversion methods
def create_user_db(user_input: UserInput, user_id: int) -> UserDB:
    """Convert UserInput to UserDB."""
    # In a real app, you'd hash the password
    hashed_password = f"hashed_{user_input.password}"

    return UserDB(
        id=user_id,
        name=user_input.name,
        email=user_input.email,
        hashed_password=hashed_password,
        age=user_input.age
    )

def create_user_output(user_db: UserDB) -> UserOutput:
    """Convert UserDB to UserOutput."""
    # Using model_dump to convert to dict, then creating new model
    user_data = user_db.model_dump(exclude={"hashed_password"})
    return UserOutput(**user_data)
```

### 🔌 Model Adapters

Create adapter patterns for model conversion:

```python
from pydantic import BaseModel
from typing import TypeVar, Type, Dict, Any, Generic

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=BaseModel)

class ModelAdapter(Generic[T, U]):
    """Adapter for converting between related models."""

    def __init__(self, source_model: Type[T], target_model: Type[U], field_mapping: Dict[str, str] = None):
        self.source_model = source_model
        self.target_model = target_model
        self.field_mapping = field_mapping or {}

    def adapt(self, source: T, **extra_fields) -> U:
        """Convert source model to target model."""
        # Get data from source model
        source_data = source.model_dump()

        # Apply field mapping
        target_data = {}
        for target_field, source_field in self.field_mapping.items():
            if source_field in source_data:
                target_data[target_field] = source_data[source_field]

        # Add unmapped fields that exist in both models
        for field in source_data:
            if field not in self.field_mapping.values() and field in self.target_model.model_fields:
                target_data[field] = source_data[field]

        # Add extra fields
        target_data.update(extra_fields)

        # Create target model
        return self.target_model(**target_data)

# Example usage
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str

class ProductDTO(BaseModel):
    product_id: int
    title: str
    price: float
    details: str

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

# Use adapter
product = Product(id=1, name="Laptop", price=999.99, description="Powerful laptop")
product_dto = product_adapter.adapt(product)
print(product_dto)  # ProductDTO(product_id=1, title='Laptop', price=999.99, details='Powerful laptop')
```

---

## 🔍 Practical Example: Form System with Model Composition

Let's build a flexible form system using model composition:

```python
from pydantic import BaseModel, Field, field_validator, create_model
from typing import Dict, Any, Type, Optional, List, Union, ClassVar
from datetime import date, datetime
import re

# Base form field types
class FormField(BaseModel):
    name: str
    label: str
    required: bool = True
    help_text: Optional[str] = None

class StringField(FormField):
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

class NumberField(FormField):
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class BooleanField(FormField):
    default: bool = False

class DateField(FormField):
    min_date: Optional[date] = None
    max_date: Optional[date] = None

class SelectField(FormField):
    choices: List[Dict[str, str]]  # List of {value: str, label: str}
    multiple: bool = False

# Form definition
class FormDefinition(BaseModel):
    title: str
    fields: List[Union[StringField, NumberField, BooleanField, DateField, SelectField]]

    def create_model(self) -> Type[BaseModel]:
        """Generate a Pydantic model from the form definition."""
        fields = {}
        validators = {}

        for field in self.fields:
            # Determine field type
            if isinstance(field, StringField):
                field_type = str
                field_default = "" if not field.required else ...
                field_constraints = {}

                if field.min_length is not None:
                    field_constraints["min_length"] = field.min_length
                if field.max_length is not None:
                    field_constraints["max_length"] = field.max_length

                # Add pattern validator if needed
                if field.pattern:
                    validator_name = f"validate_{field.name}"

                    def create_validator(pattern):
                        def validator(cls, v):
                            if not re.match(pattern, v):
                                raise ValueError(f"Value does not match pattern: {pattern}")
                            return v
                        return validator

                    validators[validator_name] = field_validator(field.name)(create_validator(field.pattern))

            elif isinstance(field, NumberField):
                field_type = float
                field_default = 0.0 if not field.required else ...
                field_constraints = {}

                if field.min_value is not None:
                    field_constraints["ge"] = field.min_value
                if field.max_value is not None:
                    field_constraints["le"] = field.max_value

            elif isinstance(field, BooleanField):
                field_type = bool
                field_default = field.default
                field_constraints = {}

            elif isinstance(field, DateField):
                field_type = date
                field_default = None if not field.required else ...
                field_constraints = {}

                # Add date range validator if needed
                if field.min_date or field.max_date:
                    validator_name = f"validate_{field.name}"

                    def create_validator(min_date, max_date):
                        def validator(cls, v):
                            if min_date and v < min_date:
                                raise ValueError(f"Date must be on or after {min_date}")
                            if max_date and v > max_date:
                                raise ValueError(f"Date must be on or before {max_date}")
                            return v
                        return validator

                    validators[validator_name] = field_validator(field.name)(
                        create_validator(field.min_date, field.max_date)
                    )

            elif isinstance(field, SelectField):
                if field.multiple:
                    field_type = List[str]
                    field_default = [] if not field.required else ...
                else:
                    field_type = str
                    field_default = "" if not field.required else ...

                field_constraints = {}

                # Add choices validator
                validator_name = f"validate_{field.name}"

                def create_validator(choices, multiple):
                    valid_values = [choice["value"] for choice in choices]

                    def validator(cls, v):
                        if multiple:
                            invalid = [x for x in v if x not in valid_values]
                            if invalid:
                                raise ValueError(f"Invalid choices: {', '.join(invalid)}")
                        elif v not in valid_values:
                            raise ValueError(f"Invalid choice: {v}")
                        return v

                    return validator

                validators[validator_name] = field_validator(field.name)(
                    create_validator(field.choices, field.multiple)
                )

            # Make field optional if not required
            if not field.required:
                field_type = Optional[field_type]

            # Create field definition
            if field_constraints:
                field_def = (field_type, Field(field_default, description=field.help_text, **field_constraints))
            else:
                field_def = (field_type, Field(field_default, description=field.help_text))

            fields[field.name] = field_def

        # Create model with fields and validators
        model = create_model(
            self.title.replace(" ", "") + "Form",
            **fields
        )

        # Add validators
        for name, validator in validators.items():
            setattr(model, name, validator)

        return model

# Example usage
contact_form_def = FormDefinition(
    title="Contact Form",
    fields=[
        StringField(
            name="name",
            label="Full Name",
            required=True,
            min_length=2,
            max_length=100,
            help_text="Your full name"
        ),
        StringField(
            name="email",
            label="Email Address",
            required=True,
            pattern=r"[^@]+@[^@]+\.[^@]+",
            help_text="Your email address"
        ),
        StringField(
            name="subject",
            label="Subject",
            required=True,
            min_length=5,
            max_length=200
        ),
        StringField(
            name="message",
            label="Message",
            required=True,
            min_length=10,
            help_text="Your message"
        ),
        BooleanField(
            name="subscribe",
            label="Subscribe to newsletter",
            required=False,
            default=False
        )
    ]
)

# Generate model from form definition
ContactForm = contact_form_def.create_model()

# Use the generated model
try:
    form_data = ContactForm(
        name="John Doe",
        email="john@example.com",
        subject="Question about your service",
        message="I would like to know more about your services.",
        subscribe=True
    )
    print("Valid form data:", form_data)
except Exception as e:
    print("Validation error:", e)
```

---

## 💪 Practice Exercises

1. **Create a Model Hierarchy**: Design a model hierarchy for different types of users (Guest, RegisteredUser, AdminUser) with appropriate inheritance relationships.

2. **Implement a Change Tracking Mixin**: Create a mixin for tracking model changes that records the previous and new values of fields when they're updated.

3. **Build a Database Schema Generator**: Develop a dynamic model generator that creates Pydantic models from database table schemas.

4. **Create an API Adapter System**: Implement an adapter system that can convert between API request models, database models, and API response models.

5. **Build a Form Generator**: Create a form builder that generates both Pydantic models for validation and HTML form elements from a single definition.

---

## 🔍 Key Concepts to Remember

1. **Model Composition**: Helps manage complexity in large applications by breaking down models into smaller, reusable components
2. **Inheritance Patterns**: Provide a way to share common fields and validation logic across related models
3. **Mixins**: Allow for reusable functionality to be added to different model hierarchies
4. **Dynamic Model Generation**: Enables flexible, runtime-defined validation schemas
5. **Model Transformation**: Facilitates conversion between related models for different contexts

---

## 📚 Additional Resources

- [Pydantic Model Composition Documentation](https://docs.pydantic.dev/latest/usage/models/)
- [Python Type Hints Guide](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)
- [JSON Schema to Pydantic Converter](https://jsontopydantic.com/)

---

## 🚀 Next Steps

In the next lesson, we'll explore how to integrate these advanced validation patterns into agent systems, focusing on input/output validation, state validation, and agent-specific validation patterns.

---

> 💡 **Note on LLM Integration**: The model composition patterns we've explored in this lesson can be particularly useful when working with LLMs. You can use these patterns to create structured validation for LLM inputs and outputs, ensuring that the data flowing through your agent system maintains consistency and type safety.

---

Happy coding! 🧩
