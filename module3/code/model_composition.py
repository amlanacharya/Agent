"""
Advanced Model Composition Patterns for Pydantic

This module demonstrates various model composition patterns using Pydantic,
including inheritance, mixins, nested models, and dynamic model generation.
"""

from pydantic import BaseModel, Field, field_validator, create_model
from typing import (
    Optional, List, Dict, Any, Type, TypeVar, Generic, 
    ClassVar, Union, get_type_hints
)
from datetime import datetime, date
from abc import ABC
import re


# ===== Inheritance Patterns =====

class BaseItem(BaseModel):
    """Base model with common fields for items."""
    id: int
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Product(BaseItem):
    """Product model inheriting from BaseItem."""
    name: str
    price: float
    description: Optional[str] = None
    
    
class User(BaseItem):
    """User model inheriting from BaseItem."""
    username: str
    email: str
    is_active: bool = True


# ===== Multi-Level Inheritance =====

class BaseContent(BaseModel):
    """Base model for content items."""
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    

class Article(BaseContent):
    """Article model extending BaseContent."""
    body: str
    author: str
    

class BlogPost(Article):
    """BlogPost model extending Article."""
    tags: List[str] = []
    comments_enabled: bool = True
    

class NewsArticle(Article):
    """NewsArticle model extending Article."""
    source: str
    breaking: bool = False


# ===== Mixin Classes =====

class TimestampMixin(BaseModel):
    """Mixin providing timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class VersionMixin(BaseModel):
    """Mixin providing versioning capabilities."""
    version: str = "1.0"
    
    @classmethod
    def get_latest_version(cls) -> str:
        """Get the latest version from model config or default."""
        return cls.model_config.get("latest_version", cls.version)
    
    model_config = {"latest_version": "1.0"}


class AuditMixin(BaseModel):
    """Mixin providing audit fields."""
    created_by: Optional[str] = None
    last_modified_by: Optional[str] = None
    
    def record_modification(self, user: str):
        """Record a modification by a user."""
        self.last_modified_by = user


# Using mixins
class Document(TimestampMixin, VersionMixin, AuditMixin):
    """Document model using multiple mixins."""
    title: str
    content: str
    
    model_config = {"latest_version": "2.5"}


# ===== Abstract Base Models =====

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


class SearchableProduct(Searchable):
    """Product model implementing Searchable."""
    id: int
    name: str
    description: Optional[str] = None
    price: float
    
    search_fields = ["name", "description"]


# ===== Composition Over Inheritance =====

class Address(BaseModel):
    """Address model for nested composition."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str


class ContactInfo(BaseModel):
    """Contact information model with nested Address."""
    email: str  # Using str instead of EmailStr for simplicity
    phone: Optional[str] = None
    address: Address


class Education(BaseModel):
    """Education model for nested lists."""
    institution: str
    degree: str
    graduation_date: date


class Person(BaseModel):
    """Person model composing other models."""
    name: str
    birth_date: date
    contact: ContactInfo
    education: List[Education] = []


# ===== Reusing Field Definitions =====

# Reusable field definitions
name_field = Field(..., min_length=2, max_length=100)
email_field = Field(..., pattern=r"[^@]+@[^@]+\.[^@]+")
password_field = Field(..., min_length=8, max_length=100)
age_field = Field(..., ge=0, le=120)


# Reusable validators
def validate_username(cls, v: str) -> str:
    """Validate username format."""
    if not re.match(r'^[a-zA-Z0-9_]+$', v):
        raise ValueError("Username can only contain letters, numbers, and underscores")
    return v


# Models using shared definitions
class UserWithSharedFields(BaseModel):
    """User model using shared field definitions."""
    name: str = name_field
    email: str = email_field
    username: str = Field(..., min_length=3, max_length=20)
    age: int = age_field
    
    _validate_username = field_validator('username')(validate_username)


class EmployeeWithSharedFields(BaseModel):
    """Employee model using shared field definitions."""
    name: str = name_field
    email: str = email_field
    employee_id: str
    department: str
    manager_name: Optional[str] = name_field.get_default()


# ===== Composition with Factories =====

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


# ===== Dynamic Model Generation =====

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


# ===== Schema-Driven Models =====

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


# ===== Model Transformation Patterns =====

# Input model (from API)
class UserInput(BaseModel):
    """User input model from API."""
    name: str
    email: str
    password: str
    age: Optional[int] = None


# Database model
class UserDB(BaseModel):
    """User database model."""
    id: int
    name: str
    email: str
    hashed_password: str
    age: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


# Output model (to API)
class UserOutput(BaseModel):
    """User output model for API responses."""
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


# ===== Model Adapters =====

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


class ProductDTO(BaseModel):
    """Product Data Transfer Object."""
    product_id: int
    title: str
    price: float
    details: str


# ===== Form System with Model Composition =====

class FormField(BaseModel):
    """Base form field type."""
    name: str
    label: str
    required: bool = True
    help_text: Optional[str] = None


class StringField(FormField):
    """String form field type."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


class NumberField(FormField):
    """Number form field type."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class BooleanField(FormField):
    """Boolean form field type."""
    default: bool = False


class DateField(FormField):
    """Date form field type."""
    min_date: Optional[date] = None
    max_date: Optional[date] = None


class SelectField(FormField):
    """Select form field type."""
    choices: List[Dict[str, str]]  # List of {value: str, label: str}
    multiple: bool = False


class FormDefinition(BaseModel):
    """Form definition model."""
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
            if not field.required and field_type != bool:
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


# Example contact form definition
def create_contact_form_definition() -> FormDefinition:
    """Create an example contact form definition."""
    return FormDefinition(
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
