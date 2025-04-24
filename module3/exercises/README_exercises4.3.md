# Exercises for Lesson 4.3: Advanced Model Composition

This directory contains solutions for the exercises from Lesson 4.3 on Advanced Model Composition with Pydantic.

## Exercise 1: User Hierarchy

**File:** `exercise4.3.1_user_hierarchy.py`

This exercise implements a model hierarchy for different types of users (Guest, RegisteredUser, AdminUser) with appropriate inheritance relationships.

**Key concepts:**
- Basic inheritance from a common base class
- Field validation with field_validator
- Method implementation in model classes
- Type-specific functionality

**Usage:**
```python
from exercise4.3.1_user_hierarchy import GuestUser, RegisteredUser, AdminUser

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

# Use methods
guest.increment_visit()
registered.update_login()
admin.grant_permission("delete_users")
admin.promote()
```

## Exercise 2: Change Tracking Mixin

**File:** `exercise4.3.2_change_tracking_mixin.py`

This exercise implements a mixin for tracking model changes that records the previous and new values of fields when they're updated.

**Key concepts:**
- Mixin implementation for reusable functionality
- Change tracking and history
- Field comparison and state management
- Reversion capabilities

**Usage:**
```python
from exercise4.3.2_change_tracking_mixin import ChangeTrackingMixin, UserProfile

# Create a user profile with change tracking
user = UserProfile(
    username="johndoe",
    email="john@example.com",
    display_name="John Doe"
)

# Make changes and track them
changes = user.update(
    display_name="John D.",
    bio="Software developer",
    age=30
)

# View change history
for change in user.get_change_history():
    print(f"{change.field_name}: {change.old_value} -> {change.new_value}")

# Revert changes
user.revert_last_change()
user.revert_all_changes()
```

## Exercise 3: Database Model Generator

**File:** `exercise4.3.3_db_model_generator.py`

This exercise implements a dynamic model generator that creates Pydantic models from database table schemas.

**Key concepts:**
- Dynamic model generation from schema definitions
- Type mapping between database and Python types
- Field constraint generation
- Multi-model relationships

**Usage:**
```python
from exercise4.3.3_db_model_generator import DBModelGenerator, DBTable, DBColumn

# Define a table schema
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
            max_length=50,
            nullable=False
        ),
        # ... more columns
    ]
)

# Generate a model
User = DBModelGenerator.create_model_from_table(users_table)

# Use the generated model
user = User(
    id=1,
    username="johndoe",
    email="john@example.com"
)
```

## Exercise 4: Model Adapter System

**File:** `exercise4.3.4_model_adapter_system.py`

This exercise implements an adapter system that can convert between API request models, database models, and API response models.

**Key concepts:**
- Generic adapter implementation
- Field mapping and transformation
- Adapter registry for managing multiple adapters
- Batch conversion capabilities

**Usage:**
```python
from exercise4.3.4_model_adapter_system import (
    ModelAdapter, AdapterRegistry,
    CreateUserRequest, UserDB, UserResponse
)

# Create adapter registry
registry = AdapterRegistry()

# Register adapters
create_user_adapter = ModelAdapter(
    CreateUserRequest,
    UserDB,
    field_mapping={"password_hash": "password"},
    transformers={"password_hash": hash_password}
)
registry.register(CreateUserRequest, UserDB, create_user_adapter)

# Use adapters
create_request = CreateUserRequest(
    username="johndoe",
    email="john@example.com",
    password="securepassword"
)

# Convert to database model
user_db = registry.adapt(create_request, UserDB)

# Convert to API response
user_response = registry.adapt(user_db, UserResponse)
```

## Exercise 5: Form Builder

**File:** `exercise4.3.5_form_builder.py`

This exercise implements a form builder that generates both Pydantic models for validation and HTML form elements from a single definition.

**Key concepts:**
- Dual-purpose model generation (validation and UI)
- HTML generation from model definitions
- Field type mapping and constraints
- Validator generation

**Usage:**
```python
from exercise4.3.5_form_builder import (
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
            required=True,
            min_length=2
        ),
        StringField(
            name="email",
            label="Email Address",
            required=True,
            input_type="email"
        ),
        SelectField(
            name="subject",
            label="Subject",
            required=True,
            choices=[
                {"value": "general", "label": "General Inquiry"},
                {"value": "support", "label": "Technical Support"}
            ]
        ),
        BooleanField(
            name="subscribe",
            label="Subscribe to newsletter",
            default=False
        )
    ]
)

# Generate Pydantic model
ContactForm = contact_form.create_model()

# Generate HTML
html_form = contact_form.generate_html()

# Use the model
form_data = ContactForm(
    name="John Doe",
    email="john@example.com",
    subject="general",
    subscribe=True
)
```

## Running the Exercises

Each exercise file can be run directly to see a demonstration of the implemented functionality:

```bash
python exercise4.3.1_user_hierarchy.py
python exercise4.3.2_change_tracking_mixin.py
python exercise4.3.3_db_model_generator.py
python exercise4.3.4_model_adapter_system.py
python exercise4.3.5_form_builder.py
```
