# Lesson 4.3: Model Composition Exercises

This document provides detailed information about the exercises for Lesson 4.3 on Model Composition.

## Overview

These exercises focus on advanced Pydantic model composition techniques, including:

- Inheritance and class hierarchies
- Mixins for cross-cutting concerns
- Dynamic model generation
- Model adaptation and conversion
- Form generation with validation

## Exercise 1: User Hierarchy with Inheritance

**File:** `lesson4_3_1_exercises.py`

This exercise demonstrates creating a model hierarchy for different types of users (Guest, RegisteredUser, AdminUser) with appropriate inheritance relationships.

**Key concepts:**
- Class inheritance with Pydantic models
- Field validation with field_validator
- Type-specific functionality

**Usage:**
```python
from module3.exercises.lesson4_3_1_exercises import GuestUser, RegisteredUser, AdminUser

# Create a guest user
guest = GuestUser(
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0",
    session_id="sess_12345"
)

# Create a registered user
registered = RegisteredUser(
    ip_address="192.168.1.2",
    user_agent="Mozilla/5.0",
    username="johndoe",
    email="john@example.com"
)

# Create an admin user
admin = AdminUser(
    ip_address="192.168.1.3",
    user_agent="Mozilla/5.0",
    username="admin_jane",
    email="jane@example.com",
    admin_level=2,
    permissions=["manage_users", "edit_content"]
)

# Use type-specific methods
guest.increment_visit()
registered.update_login()
admin.grant_permission("delete_users")
admin.promote()
```

## Exercise 2: Change Tracking Mixin

**File:** `lesson4_3_2_exercises.py`

This exercise implements a mixin for tracking model changes that records the previous and new values of fields when they're updated.

**Key concepts:**
- Mixins with Pydantic models
- Change tracking and history
- Field comparison and recording

**Usage:**
```python
from module3.exercises.lesson4_3_2_exercises import ChangeTrackingMixin, UserProfile

# Create a user profile with change tracking
user = UserProfile(
    username="johndoe",
    email="john@example.com"
)

# Make changes and track them
changes = user.update(
    display_name="John Doe",
    bio="Software developer",
    age=30
)

# View change history
history = user.get_change_history()
field_history = user.get_field_history("display_name")

# Revert changes
user.revert_last_change()  # Revert the most recent change
user.revert_all_changes()  # Revert all changes
```

## Exercise 3: Database Model Generator

**File:** `lesson4_3_3_exercises.py`

This exercise implements a dynamic model generator that creates Pydantic models from database table schemas.

**Key concepts:**
- Dynamic model creation
- Type mapping (DB to Python)
- Field constraints generation

**Usage:**
```python
from module3.exercises.lesson4_3_3_exercises import DBModelGenerator, DBTable, DBColumn

# Define a database table schema
users_table = DBTable(
    name="users",
    description="User accounts table",
    columns=[
        DBColumn(
            name="id",
            data_type="integer",
            nullable=False,
            primary_key=True
        ),
        DBColumn(
            name="username",
            data_type="varchar",
            nullable=False,
            unique=True,
            max_length=50
        ),
        DBColumn(
            name="email",
            data_type="varchar",
            nullable=False
        )
    ]
)

# Generate a Pydantic model
User = DBModelGenerator.create_model_from_table(users_table)

# Create an instance of the generated model
user = User(id=1, username="johndoe", email="john@example.com")
```

## Exercise 4: Model Adapter System

**File:** `lesson4_3_4_exercises.py`

This exercise implements an adapter system that can convert between API request models, database models, and API response models.

**Key concepts:**
- Model conversion and adaptation
- Field mapping and transformation
- Adapter registry pattern

**Usage:**
```python
from module3.exercises.lesson4_3_4_exercises import (
    ModelAdapter, AdapterRegistry,
    CreateUserRequest, UserDB, UserResponse
)

# Create an adapter
adapter = ModelAdapter(
    CreateUserRequest,
    UserDB,
    field_mapping={"password_hash": "password"},
    transformers={"password_hash": lambda p: f"hashed_{p}"}
)

# Create a registry
registry = AdapterRegistry()
registry.register(CreateUserRequest, UserDB, adapter)

# Create a request
request = CreateUserRequest(
    username="johndoe",
    email="john@example.com",
    password="securepassword"
)

# Convert to database model
user_db = registry.adapt(request, UserDB)
```

## Exercise 5: Form Builder

**File:** `lesson4_3_5_exercises.py`

This exercise implements a form builder that generates both Pydantic models for validation and HTML form elements from a single definition.

**Key concepts:**
- Field type mapping and constraints
- Validator generation
- HTML generation with proper escaping

**Usage:**
```python
from module3.exercises.lesson4_3_5_exercises import (
    FormDefinition, StringField, NumberField,
    BooleanField, SelectField
)

# Create a form definition
contact_form = FormDefinition(
    title="Contact Us",
    description="Please fill out this form to get in touch.",
    fields=[
        StringField(
            name="name",
            label="Full Name",
            required=True
        ),
        StringField(
            name="email",
            label="Email Address",
            type="email",
            required=True
        ),
        StringField(
            name="message",
            label="Message",
            type="textarea",
            required=True
        )
    ]
)

# Generate a Pydantic model
ContactForm = contact_form.create_model()

# Validate form data
form_data = ContactForm(
    name="John Doe",
    email="john@example.com",
    message="Hello, world!"
)

# Generate HTML form
html_form = contact_form.generate_html()
```

## Running the Exercises

You can run any of the exercise solutions directly from the command line:

```bash
python -m module3.exercises.lesson4_3_1_exercises
python -m module3.exercises.lesson4_3_2_exercises
python -m module3.exercises.lesson4_3_3_exercises
python -m module3.exercises.lesson4_3_4_exercises
python -m module3.exercises.lesson4_3_5_exercises
```
