# Lesson 4.1: Complex Validation Scenarios 🛡️

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

## 📋 Overview

In this lesson, we'll explore advanced validation patterns for complex data scenarios. While basic Pydantic validation is sufficient for many use cases, real-world applications often require more sophisticated validation logic that spans multiple fields, depends on context, or involves complex business rules.

## 🧩 Beyond Basic Validation

Basic validation in Pydantic focuses on individual fields, but complex validation often requires:

1. **Cross-field validation**: Validating fields in relation to each other
2. **Conditional validation**: Applying different validation rules based on conditions
3. **Context-dependent validation**: Validating based on external context
4. **Dynamic validation**: Validation rules that change at runtime

![Complex Validation](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPrc2ngFZ6BTyww/giphy.gif)

## 🛠️ Cross-Field Validation

### Using model_validator

The `@model_validator` decorator allows you to validate multiple fields together:

```python
from pydantic import BaseModel, model_validator

class TimeRange(BaseModel):
    start_time: datetime
    end_time: datetime
    
    @model_validator(mode='after')
    def check_times_order(self, data):
        if self.start_time >= self.end_time:
            raise ValueError("end_time must be after start_time")
        return self
```

### Dependent Field Validation

Sometimes fields depend on each other's values:

```python
from pydantic import BaseModel, field_validator
from typing import Optional, Literal

class ShippingInfo(BaseModel):
    shipping_method: Literal["standard", "express", "international"]
    tracking_number: Optional[str] = None
    customs_id: Optional[str] = None
    
    @field_validator('tracking_number')
    def validate_tracking_number(cls, v, info):
        shipping_method = info.data.get('shipping_method')
        if shipping_method in ["express", "standard"] and not v:
            raise ValueError(f"{shipping_method} shipping requires a tracking number")
        return v
    
    @field_validator('customs_id')
    def validate_customs_id(cls, v, info):
        shipping_method = info.data.get('shipping_method')
        if shipping_method == "international" and not v:
            raise ValueError("International shipping requires a customs ID")
        return v
```

## 🔄 Conditional Validation

### Type-Based Validation

Apply different validation based on the type of data:

```python
from pydantic import BaseModel, field_validator
from typing import Union, Literal

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    content: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: str
    alt_text: Optional[str] = None

class VideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: str
    duration: int  # seconds

Content = Union[TextContent, ImageContent, VideoContent]

class Message(BaseModel):
    id: int
    content: Content
    
    @field_validator('content')
    def validate_content(cls, v):
        if isinstance(v, TextContent) and len(v.content) < 1:
            raise ValueError("Text content cannot be empty")
        elif isinstance(v, ImageContent) and not v.url.startswith(('http://', 'https://')):
            raise ValueError("Image URL must be a valid HTTP/HTTPS URL")
        elif isinstance(v, VideoContent) and v.duration <= 0:
            raise ValueError("Video duration must be positive")
        return v
```

### Value-Based Validation

Apply different validation based on field values:

```python
from pydantic import BaseModel, field_validator
from typing import Optional, Literal

class Payment(BaseModel):
    method: Literal["credit_card", "bank_transfer", "paypal"]
    credit_card_number: Optional[str] = None
    bank_account: Optional[str] = None
    paypal_email: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_payment_details(self):
        if self.method == "credit_card" and not self.credit_card_number:
            raise ValueError("Credit card number is required for credit card payments")
        elif self.method == "bank_transfer" and not self.bank_account:
            raise ValueError("Bank account is required for bank transfers")
        elif self.method == "paypal" and not self.paypal_email:
            raise ValueError("PayPal email is required for PayPal payments")
        return self
```

## 🌐 Context-Dependent Validation

Sometimes validation depends on external context that isn't part of the model itself:

### Using Root Validators with Context

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class BookingRequest(BaseModel):
    room_id: int
    check_in_date: datetime
    check_out_date: datetime
    guest_count: int
    
    @model_validator(mode='after')
    def validate_booking(self):
        # Basic validation
        if self.check_in_date >= self.check_out_date:
            raise ValueError("Check-out must be after check-in")
        
        if (self.check_out_date - self.check_in_date).days > 14:
            raise ValueError("Maximum stay is 14 days")
        
        return self
    
    # This would be called externally with context
    def validate_with_context(self, context: Dict[str, Any]) -> bool:
        """Validate booking with external context like room capacity and availability."""
        # Check room capacity
        room_capacity = context.get('room_capacity', {}).get(self.room_id, 0)
        if self.guest_count > room_capacity:
            raise ValueError(f"Room {self.room_id} can only accommodate {room_capacity} guests")
        
        # Check room availability
        bookings = context.get('existing_bookings', [])
        for booking in bookings:
            if booking['room_id'] == self.room_id:
                existing_check_in = booking['check_in_date']
                existing_check_out = booking['check_out_date']
                
                # Check for overlap
                if (self.check_in_date < existing_check_out and 
                    self.check_out_date > existing_check_in):
                    raise ValueError(f"Room {self.room_id} is already booked during this period")
        
        return True
```

### Using External Validation Functions

```python
from pydantic import BaseModel, field_validator
from typing import Callable, Optional

class User(BaseModel):
    username: str
    email: str
    
    @field_validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v

# External validation function
def validate_unique_username(username: str, database_connection) -> bool:
    """Check if username is unique in the database."""
    # This would be a real database query in practice
    existing_users = database_connection.query(f"SELECT * FROM users WHERE username = '{username}'")
    return len(existing_users) == 0

# Usage
def create_user(user_data: dict, db_connection):
    user = User(**user_data)
    if not validate_unique_username(user.username, db_connection):
        raise ValueError(f"Username '{user.username}' is already taken")
    # Proceed with user creation
    return user
```

## 🧪 Dynamic Validation Rules

Sometimes validation rules need to be determined at runtime:

### Configurable Validation

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Dict, Any, Optional, List

class ConfigurableModel(BaseModel):
    name: str
    value: float
    tags: List[str] = []
    
    # Configuration for validation rules
    validation_config: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def validate_with_config(self):
        if not self.validation_config:
            return self
            
        # Apply min/max constraints if configured
        if 'min_value' in self.validation_config and self.value < self.validation_config['min_value']:
            raise ValueError(f"Value must be at least {self.validation_config['min_value']}")
            
        if 'max_value' in self.validation_config and self.value > self.validation_config['max_value']:
            raise ValueError(f"Value must be at most {self.validation_config['max_value']}")
            
        # Apply required tags if configured
        if 'required_tags' in self.validation_config:
            missing_tags = set(self.validation_config['required_tags']) - set(self.tags)
            if missing_tags:
                raise ValueError(f"Missing required tags: {', '.join(missing_tags)}")
                
        return self

# Usage
config = {
    'min_value': 0,
    'max_value': 100,
    'required_tags': ['important', 'verified']
}

# This will raise an error due to missing tags
try:
    model = ConfigurableModel(
        name="Test",
        value=50,
        tags=["important"],
        validation_config=config
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Validation Factories

Create validation functions dynamically:

```python
from pydantic import BaseModel, field_validator
from typing import List, Callable, Any

def create_range_validator(min_val: float, max_val: float) -> Callable:
    """Create a validator function for a specific range."""
    def validate_range(cls, v: float) -> float:
        if v < min_val or v > max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
        return v
    return validate_range

def create_list_length_validator(min_len: int, max_len: int) -> Callable:
    """Create a validator function for list length."""
    def validate_list_length(cls, v: List[Any]) -> List[Any]:
        if len(v) < min_len or len(v) > max_len:
            raise ValueError(f"List must have between {min_len} and {max_len} items")
        return v
    return validate_list_length

# Usage
class DynamicModel(BaseModel):
    value: float
    items: List[str]

# Add validators dynamically
setattr(DynamicModel, 'validate_value', field_validator('value')(create_range_validator(0, 100)))
setattr(DynamicModel, 'validate_items', field_validator('items')(create_list_length_validator(1, 5)))

# Test
try:
    model = DynamicModel(value=150, items=[])  # Will raise validation errors
except ValueError as e:
    print(f"Validation error: {e}")
```

## 🔍 Practical Example: Advanced Form Validation

Let's implement a complex form validation system for a job application:

```python
from pydantic import BaseModel, field_validator, model_validator, EmailStr, Field
from typing import List, Optional, Literal
from datetime import datetime, date

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_date: date
    end_date: Optional[date] = None
    current: bool = False
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.end_date and self.start_date > self.end_date:
            raise ValueError("End date must be after start date")
        
        if self.current and self.end_date:
            raise ValueError("Current education should not have an end date")
        
        if not self.current and not self.end_date:
            raise ValueError("Non-current education must have an end date")
            
        return self

class WorkExperience(BaseModel):
    company: str
    position: str
    start_date: date
    end_date: Optional[date] = None
    current: bool = False
    responsibilities: List[str]
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.end_date and self.start_date > self.end_date:
            raise ValueError("End date must be after start date")
        
        if self.current and self.end_date:
            raise ValueError("Current position should not have an end date")
        
        if not self.current and not self.end_date:
            raise ValueError("Non-current position must have an end date")
            
        return self
    
    @field_validator('responsibilities')
    def validate_responsibilities(cls, v):
        if len(v) < 1:
            raise ValueError("At least one responsibility is required")
        return v

class Skill(BaseModel):
    name: str
    level: Literal["beginner", "intermediate", "advanced", "expert"]
    years_of_experience: float
    
    @field_validator('years_of_experience')
    def validate_experience(cls, v, info):
        level = info.data.get('level')
        if level == "expert" and v < 3:
            raise ValueError("Expert level requires at least 3 years of experience")
        elif level == "advanced" and v < 2:
            raise ValueError("Advanced level requires at least 2 years of experience")
        return v

class JobApplication(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: EmailStr
    phone: str
    date_of_birth: date
    education: List[Education]
    work_experience: List[WorkExperience]
    skills: List[Skill]
    cover_letter: str = Field(..., min_length=100)
    
    @field_validator('phone')
    def validate_phone(cls, v):
        # Simple phone validation - would be more complex in real app
        digits = ''.join(filter(str.isdigit, v))
        if len(digits) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return v
    
    @field_validator('date_of_birth')
    def validate_age(cls, v):
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 18:
            raise ValueError("Applicant must be at least 18 years old")
        return v
    
    @model_validator(mode='after')
    def validate_application(self):
        # Ensure there's at least one education entry
        if not self.education:
            raise ValueError("At least one education entry is required")
            
        # Ensure there's at least one work experience
        if not self.work_experience:
            raise ValueError("At least one work experience is required")
            
        # Ensure there are at least three skills
        if len(self.skills) < 3:
            raise ValueError("At least three skills are required")
            
        # Check for required skill types (domain-specific validation)
        skill_names = [skill.name.lower() for skill in self.skills]
        required_skills = ["communication", "teamwork"]
        missing_skills = [skill for skill in required_skills if not any(req in name for name in skill_names for req in [skill])]
        
        if missing_skills:
            raise ValueError(f"Missing required skills: {', '.join(missing_skills)}")
            
        return self
```

## 🧪 Exercises

1. Create a `PaymentSystem` model with conditional validation based on payment method (credit card, PayPal, bank transfer) that validates the appropriate fields for each method.

2. Implement a `TravelBooking` model that validates dates, passenger information, and applies different validation rules based on domestic vs. international travel.

3. Build a `ProductInventory` model with dynamic validation rules that can be configured at runtime for different product categories.

4. Create a validation system for a survey form that enforces different validation rules based on previous answers (e.g., if a user selects "Other" for a question, a text field becomes required).

5. Implement cross-field validation for a password change form that ensures the new password is different from the old password and meets complexity requirements.

## 🔍 Key Takeaways

- Complex validation often requires validating multiple fields together
- Conditional validation allows for different rules based on context
- External context can be incorporated into validation logic
- Dynamic validation rules provide flexibility for changing requirements
- Proper validation error messages improve user experience

## 📚 Additional Resources

- [Pydantic Validators Documentation](https://docs.pydantic.dev/latest/usage/validators/)
- [Advanced Pydantic Usage Patterns](https://docs.pydantic.dev/latest/usage/models/)
- [Validation Best Practices](https://docs.pydantic.dev/latest/usage/validation_decorator/)

## 🚀 Next Steps

In the next lesson, we'll explore advanced error handling and recovery strategies for validation failures, ensuring that our applications can gracefully handle invalid inputs.
