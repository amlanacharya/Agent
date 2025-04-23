# Lesson 4.2: Error Handling and Recovery 🛠️

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

## 📋 Overview

In this lesson, we'll explore advanced error handling and recovery strategies for validation failures. While validation is essential for ensuring data integrity, equally important is how we handle validation errors when they occur. Well-designed error handling improves user experience, aids debugging, and makes systems more robust.

## 🧩 The Challenge of Validation Errors

Validation errors in agent systems present several challenges:

1. **User Experience**: Technical error messages can confuse end users
2. **Error Recovery**: Systems need strategies to recover from invalid inputs
3. **Error Aggregation**: Multiple errors may occur simultaneously
4. **Contextual Errors**: Error messages need context to be useful
5. **Debugging**: Developers need detailed error information

![Error Handling](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT9IgIc0lH5qULvgLm/giphy.gif)

## 🛠️ Understanding Pydantic Validation Errors

### Anatomy of ValidationError

Pydantic's `ValidationError` contains detailed information about what went wrong:

```python
from pydantic import BaseModel, ValidationError, Field
from typing import List

class User(BaseModel):
    username: str = Field(..., min_length=3)
    email: str
    age: int = Field(..., ge=18)
    tags: List[str] = []

try:
    User(username="a", email="not-an-email", age=16, tags=["one", 2])
except ValidationError as e:
    print(f"Error: {e}")
    
    # Access structured error data
    print("\nError details:")
    for error in e.errors():
        print(f"- Location: {'.'.join(str(loc) for loc in error['loc'])}")
        print(f"  Type: {error['type']}")
        print(f"  Message: {error['msg']}")
```

Output:
```
Error: 3 validation errors for User
username
  String should have at least 3 characters [type=string_too_short, input_value='a', input_type=str]
email
  value is not a valid email address [type=value_error.email, input_value='not-an-email', input_type=str]
age
  Input should be greater than or equal to 18 [type=greater_than_equal, input_value=16, input_type=int]

Error details:
- Location: username
  Type: string_too_short
  Message: String should have at least 3 characters
- Location: email
  Type: value_error.email
  Message: value is not a valid email address
- Location: age
  Type: greater_than_equal
  Message: Input should be greater than or equal to 18
```

### Customizing Error Messages

You can customize error messages in Pydantic:

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class User(BaseModel):
    username: str = Field(
        ..., 
        min_length=3,
        description="Username must be at least 3 characters long"
    )
    email: str
    age: int = Field(
        ..., 
        ge=18,
        description="User must be at least 18 years old"
    )
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError("Email must contain an @ symbol")
        return v
```

## 🔄 Error Handling Strategies

### 1. Try-Except Pattern

The basic pattern for handling validation errors:

```python
from pydantic import BaseModel, ValidationError

class UserInput(BaseModel):
    username: str
    email: str

def process_user_input(data: dict):
    try:
        user_input = UserInput(**data)
        # Process valid input
        return {"status": "success", "data": user_input.model_dump()}
    except ValidationError as e:
        # Handle validation error
        return {
            "status": "error",
            "message": "Invalid input data",
            "details": e.errors()
        }
```

### 2. Error Aggregation

Collect and report multiple errors at once:

```python
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

def validate_multiple_items(items: List[Dict[str, Any]], model_class):
    """Validate multiple items and aggregate errors."""
    valid_items = []
    errors = {}
    
    for i, item in enumerate(items):
        try:
            valid_item = model_class(**item)
            valid_items.append(valid_item)
        except ValidationError as e:
            errors[i] = e.errors()
    
    return {
        "valid_items": valid_items,
        "errors": errors,
        "success_count": len(valid_items),
        "error_count": len(errors)
    }

# Usage
class Product(BaseModel):
    name: str
    price: float
    quantity: int

products = [
    {"name": "Laptop", "price": 999.99, "quantity": 10},
    {"name": "", "price": -50, "quantity": "five"},  # Invalid
    {"name": "Mouse", "price": 25.99, "quantity": 100}
]

result = validate_multiple_items(products, Product)
print(f"Successfully validated {result['success_count']} products")
print(f"Found errors in {result['error_count']} products")
```

### 3. Partial Validation

Sometimes it's useful to accept partial data and validate what's available:

```python
from pydantic import BaseModel, ValidationError
from typing import Optional

class UserProfile(BaseModel):
    username: str
    email: Optional[str] = None
    bio: Optional[str] = None
    age: Optional[int] = None

def update_user_profile(user_id: int, update_data: dict):
    """Update user profile with partial data."""
    try:
        # Validate the update data
        profile_update = UserProfile(**update_data)
        
        # In a real app, you'd fetch the existing profile first
        existing_profile = {"username": "current_user", "email": "user@example.com"}
        
        # Update only the fields that were provided
        updated_profile = {**existing_profile}
        for field, value in profile_update.model_dump(exclude_unset=True).items():
            if value is not None:  # Only update fields that were explicitly set
                updated_profile[field] = value
        
        return {"status": "success", "profile": updated_profile}
    except ValidationError as e:
        return {"status": "error", "message": "Invalid profile data", "details": e.errors()}
```

### 4. Hierarchical Validation

For complex objects, validate in stages to provide better error context:

```python
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    country: str

class User(BaseModel):
    username: str
    email: str
    addresses: List[Address]

def validate_user_data(data: Dict[str, Any]):
    """Validate user data with hierarchical error handling."""
    # First, validate addresses separately
    if "addresses" in data and isinstance(data["addresses"], list):
        address_errors = {}
        valid_addresses = []
        
        for i, addr in enumerate(data["addresses"]):
            try:
                valid_address = Address(**addr)
                valid_addresses.append(valid_address.model_dump())
            except ValidationError as e:
                address_errors[i] = e.errors()
        
        # If there are address errors, report them
        if address_errors:
            return {
                "status": "error",
                "message": "Invalid address data",
                "address_errors": address_errors
            }
        
        # Replace with validated addresses
        data["addresses"] = valid_addresses
    
    # Then validate the entire user
    try:
        user = User(**data)
        return {"status": "success", "user": user.model_dump()}
    except ValidationError as e:
        return {"status": "error", "message": "Invalid user data", "details": e.errors()}
```

## 🧠 Advanced Error Recovery Techniques

### 1. Default Value Substitution

Replace invalid values with defaults:

```python
from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Any, Optional

class Product(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)
    tags: List[str] = []

def safe_parse_with_defaults(data: Dict[str, Any], model_class, default_values: Dict[str, Any]):
    """Try to parse data, substituting defaults for invalid fields."""
    try:
        return model_class(**data)
    except ValidationError as e:
        # Get error locations
        error_fields = ['.'.join(str(loc) for loc in error['loc']) for error in e.errors()]
        
        # Create a new dict with defaults for error fields
        fixed_data = {**data}
        for field in error_fields:
            if field in default_values:
                fixed_data[field] = default_values[field]
        
        # Try again with fixed data
        try:
            return model_class(**fixed_data)
        except ValidationError:
            # If still failing, raise the original error
            raise e

# Usage
try:
    # This would fail validation
    data = {"name": "Product", "price": -10, "quantity": "invalid"}
    
    # Define defaults for recovery
    defaults = {
        "price": 0.99,
        "quantity": 1
    }
    
    product = safe_parse_with_defaults(data, Product, defaults)
    print(f"Recovered with defaults: {product}")
except ValidationError as e:
    print(f"Could not recover: {e}")
```

### 2. Error Correction Strategies

Attempt to fix common errors:

```python
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List
import re

class UserInput(BaseModel):
    email: str
    age: int
    tags: List[str]

def attempt_error_correction(data: Dict[str, Any]):
    """Try to correct common errors in input data."""
    corrected = {**data}
    corrections_applied = []
    
    # Fix email format
    if 'email' in data and isinstance(data['email'], str) and '@' not in data['email']:
        # Try to guess if it's missing the domain
        if not re.search(r'@[^@]+\.[^@]+$', data['email']):
            corrected['email'] = f"{data['email']}@example.com"
            corrections_applied.append(f"Added default domain to email: {corrected['email']}")
    
    # Convert string age to int
    if 'age' in data and isinstance(data['age'], str):
        try:
            corrected['age'] = int(data['age'])
            corrections_applied.append(f"Converted age from string to int: {corrected['age']}")
        except ValueError:
            pass
    
    # Ensure tags is a list
    if 'tags' in data:
        if isinstance(data['tags'], str):
            # Convert comma-separated string to list
            corrected['tags'] = [tag.strip() for tag in data['tags'].split(',')]
            corrections_applied.append(f"Converted tags from string to list: {corrected['tags']}")
        elif not isinstance(data['tags'], list):
            # Convert single item to list
            corrected['tags'] = [str(data['tags'])]
            corrections_applied.append(f"Converted tags to list: {corrected['tags']}")
    
    # Try to validate with corrections
    try:
        validated = UserInput(**corrected)
        return {
            "status": "success",
            "data": validated.model_dump(),
            "corrections": corrections_applied
        }
    except ValidationError as e:
        return {
            "status": "error",
            "message": "Validation failed even with corrections",
            "original_data": data,
            "attempted_corrections": corrections_applied,
            "errors": e.errors()
        }

# Usage
result = attempt_error_correction({
    "email": "user.example",
    "age": "25",
    "tags": "python, pydantic, validation"
})
print(result)
```

### 3. Progressive Validation

Validate in stages, accepting partial success:

```python
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional

class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    country: str

class User(BaseModel):
    username: str
    email: str
    addresses: List[Address]

def progressive_validation(data: Dict[str, Any]):
    """Validate as much as possible, returning partial results."""
    result = {
        "valid_fields": {},
        "invalid_fields": {},
        "valid_addresses": [],
        "invalid_addresses": []
    }
    
    # Validate simple fields
    for field in ['username', 'email']:
        if field in data:
            try:
                # Create a simple model just for this field
                class FieldModel(BaseModel):
                    value: str
                
                validated = FieldModel(value=data[field])
                result["valid_fields"][field] = validated.value
            except ValidationError:
                result["invalid_fields"][field] = data[field]
    
    # Validate addresses
    if 'addresses' in data and isinstance(data['addresses'], list):
        for i, addr in enumerate(data['addresses']):
            try:
                valid_address = Address(**addr)
                result["valid_addresses"].append(valid_address.model_dump())
            except ValidationError as e:
                result["invalid_addresses"].append({
                    "data": addr,
                    "errors": e.errors()
                })
    
    # Determine overall status
    if not result["invalid_fields"] and not result["invalid_addresses"]:
        result["status"] = "success"
    elif result["valid_fields"] or result["valid_addresses"]:
        result["status"] = "partial_success"
    else:
        result["status"] = "failure"
    
    return result
```

## 🔍 User-Friendly Error Messages

Transform technical validation errors into user-friendly messages:

```python
from pydantic import BaseModel, ValidationError, Field
from typing import Dict, List, Any

class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=18)

def user_friendly_errors(validation_error: ValidationError) -> Dict[str, List[str]]:
    """Convert validation errors to user-friendly messages."""
    error_messages = {}
    
    # Error message mapping
    friendly_messages = {
        "string_too_short": "This field is too short",
        "string_too_long": "This field is too long",
        "value_error.email": "Please enter a valid email address",
        "greater_than_equal": "This value must be greater than or equal to {limit_value}",
        "less_than_equal": "This value must be less than or equal to {limit_value}",
        "type_error.integer": "Please enter a whole number",
        "type_error.float": "Please enter a number",
        "value_error.missing": "This field is required"
    }
    
    # Field-specific messages
    field_messages = {
        "username": {
            "string_too_short": "Username must be at least {limit_value} characters",
            "string_too_long": "Username cannot exceed {limit_value} characters"
        },
        "password": {
            "string_too_short": "Password must be at least {limit_value} characters"
        },
        "age": {
            "greater_than_equal": "You must be at least {limit_value} years old to register"
        }
    }
    
    for error in validation_error.errors():
        field = '.'.join(str(loc) for loc in error['loc'])
        error_type = error['type']
        
        # Get field-specific message if available
        if field in field_messages and error_type in field_messages[field]:
            message_template = field_messages[field][error_type]
        # Fall back to general message
        elif error_type in friendly_messages:
            message_template = friendly_messages[error_type]
        # Use the original message if no mapping exists
        else:
            message_template = error['msg']
        
        # Format the message with any context values
        context = {k: v for k, v in error.items() if k not in ('loc', 'type', 'msg')}
        try:
            message = message_template.format(**context)
        except KeyError:
            message = message_template
        
        if field not in error_messages:
            error_messages[field] = []
        error_messages[field].append(message)
    
    return error_messages

# Usage
try:
    UserRegistration(username="a", email="not-an-email", password="123", age=16)
except ValidationError as e:
    friendly_errors = user_friendly_errors(e)
    print("Please fix the following errors:")
    for field, messages in friendly_errors.items():
        print(f"{field.capitalize()}:")
        for msg in messages:
            print(f"  - {msg}")
```

## 🧪 Practical Example: Form Validation System

Let's build a complete form validation system with error handling:

```python
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Dict, List, Any, Optional, Union
from datetime import date
import re

# Form models
class ContactForm(BaseModel):
    name: str = Field(..., min_length=2)
    email: str
    message: str = Field(..., min_length=10)
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v

class RegistrationForm(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str = Field(..., min_length=8)
    confirm_password: str
    birth_date: date
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('confirm_password')
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError("Passwords do not match")
        return v
    
    @field_validator('birth_date')
    def validate_age(cls, v):
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 18:
            raise ValueError("You must be at least 18 years old")
        return v

# Form processor
class FormProcessor:
    def __init__(self):
        self.form_types = {
            "contact": ContactForm,
            "registration": RegistrationForm
        }
        
        # User-friendly error messages
        self.error_messages = {
            "string_too_short": "This field is too short (minimum {limit_value} characters)",
            "string_too_long": "This field is too long (maximum {limit_value} characters)",
            "value_error.email": "Please enter a valid email address",
            "value_error.missing": "This field is required",
            "type_error.date": "Please enter a valid date in YYYY-MM-DD format"
        }
        
        # Field-specific messages
        self.field_messages = {
            "password": {
                "string_too_short": "Password must be at least {limit_value} characters long"
            },
            "confirm_password": {
                "value_error": "Passwords do not match"
            },
            "birth_date": {
                "value_error": "You must be at least 18 years old"
            }
        }
    
    def process_form(self, form_type: str, data: Dict[str, Any]):
        """Process a form submission with comprehensive error handling."""
        if form_type not in self.form_types:
            return {
                "status": "error",
                "message": f"Unknown form type: {form_type}"
            }
        
        form_class = self.form_types[form_type]
        
        try:
            # Validate form data
            form = form_class(**data)
            
            # Process the valid form (in a real app, this would save to DB, send email, etc.)
            return {
                "status": "success",
                "message": "Form submitted successfully",
                "data": form.model_dump()
            }
        except ValidationError as e:
            # Convert to user-friendly errors
            friendly_errors = self._format_errors(e)
            
            return {
                "status": "error",
                "message": "Please fix the errors in your submission",
                "errors": friendly_errors
            }
    
    def _format_errors(self, validation_error: ValidationError) -> Dict[str, List[str]]:
        """Format validation errors into user-friendly messages."""
        error_messages = {}
        
        for error in validation_error.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            error_type = error['type']
            
            # Get field-specific message if available
            if field in self.field_messages and error_type in self.field_messages[field]:
                message_template = self.field_messages[field][error_type]
            # Fall back to general message
            elif error_type in self.error_messages:
                message_template = self.error_messages[error_type]
            # Use the original message if no mapping exists
            else:
                message_template = error['msg']
            
            # Format the message with any context values
            context = {k: v for k, v in error.items() if k not in ('loc', 'type', 'msg')}
            try:
                message = message_template.format(**context)
            except KeyError:
                message = message_template
            
            if field not in error_messages:
                error_messages[field] = []
            error_messages[field].append(message)
        
        return error_messages

# Usage example
processor = FormProcessor()

# Process a contact form with errors
contact_result = processor.process_form("contact", {
    "name": "J",  # Too short
    "email": "not-an-email",  # Invalid email
    "message": "Hi"  # Too short
})

print("Contact Form Result:")
print(contact_result)

# Process a registration form with errors
registration_result = processor.process_form("registration", {
    "username": "user123",
    "email": "user@example.com",
    "password": "password123",
    "confirm_password": "different_password",  # Doesn't match
    "birth_date": "2010-01-01"  # Under 18
})

print("\nRegistration Form Result:")
print(registration_result)
```

## 🧪 Exercises

1. Create a validation system for a multi-step form that preserves valid data between steps and only shows errors for the current step.

2. Implement a "suggestion" system that proposes corrections for common validation errors (e.g., suggesting "gmail.com" when a user types "user@gmal.com").

3. Build an error logging system that tracks validation errors to identify common user mistakes for future UI improvements.

4. Create a validation middleware for a web API that standardizes error responses across different endpoints.

5. Implement a form that allows partial submissions, saving valid fields as draft data while highlighting fields that need correction.

## 🔍 Key Takeaways

- Well-designed error handling improves user experience and system robustness
- Pydantic's ValidationError provides detailed information about what went wrong
- Error recovery strategies can salvage partial data from invalid inputs
- User-friendly error messages are essential for good UX
- Progressive validation allows for partial success in complex forms

## 📚 Additional Resources

- [Pydantic Error Handling Documentation](https://docs.pydantic.dev/latest/usage/validation_errors/)
- [Form Validation Best Practices](https://www.smashingmagazine.com/2009/07/web-form-validation-best-practices-and-tutorials/)
- [Error Message Guidelines](https://www.nngroup.com/articles/error-message-guidelines/)

## 🚀 Next Steps

In the next lesson, we'll explore advanced model composition patterns, including inheritance, mixins, and dynamic model generation to create flexible and reusable validation systems.
