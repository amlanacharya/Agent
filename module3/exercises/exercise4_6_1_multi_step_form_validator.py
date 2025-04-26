"""
Exercise 4.6.1: Multi-Step Form Validator

This module implements a multi-step form validation system using Pydantic.
The system allows defining form steps with specific fields and validation rules,
and ensures each step is valid before allowing progression to the next step.
"""

from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Union, Callable, Type
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
from datetime import datetime


class FormStepStatus(Enum):
    """Status of a form step."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INVALID = "invalid"


class FieldType(Enum):
    """Types of form fields."""
    TEXT = "text"
    NUMBER = "number"
    EMAIL = "email"
    DATE = "date"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTAREA = "textarea"


class ValidationRule(BaseModel):
    """Validation rule for a form field."""
    rule_type: str
    params: Dict[str, Any] = {}
    error_message: str

    @classmethod
    def required(cls, error_message: str = "This field is required"):
        """Create a required field validation rule."""
        return cls(rule_type="required", error_message=error_message)
    
    @classmethod
    def min_length(cls, min_length: int, error_message: str = None):
        """Create a minimum length validation rule."""
        if error_message is None:
            error_message = f"Must be at least {min_length} characters"
        return cls(
            rule_type="min_length", 
            params={"min_length": min_length}, 
            error_message=error_message
        )
    
    @classmethod
    def max_length(cls, max_length: int, error_message: str = None):
        """Create a maximum length validation rule."""
        if error_message is None:
            error_message = f"Must be at most {max_length} characters"
        return cls(
            rule_type="max_length", 
            params={"max_length": max_length}, 
            error_message=error_message
        )
    
    @classmethod
    def pattern(cls, pattern: str, error_message: str = "Invalid format"):
        """Create a pattern validation rule."""
        return cls(
            rule_type="pattern", 
            params={"pattern": pattern}, 
            error_message=error_message
        )
    
    @classmethod
    def email(cls, error_message: str = "Invalid email format"):
        """Create an email validation rule."""
        return cls(
            rule_type="email", 
            error_message=error_message
        )
    
    @classmethod
    def min_value(cls, min_value: Union[int, float], error_message: str = None):
        """Create a minimum value validation rule."""
        if error_message is None:
            error_message = f"Must be at least {min_value}"
        return cls(
            rule_type="min_value", 
            params={"min_value": min_value}, 
            error_message=error_message
        )
    
    @classmethod
    def max_value(cls, max_value: Union[int, float], error_message: str = None):
        """Create a maximum value validation rule."""
        if error_message is None:
            error_message = f"Must be at most {max_value}"
        return cls(
            rule_type="max_value", 
            params={"max_value": max_value}, 
            error_message=error_message
        )
    
    @classmethod
    def options(cls, options: List[Any], error_message: str = "Invalid option selected"):
        """Create an options validation rule."""
        return cls(
            rule_type="options", 
            params={"options": options}, 
            error_message=error_message
        )
    
    @classmethod
    def custom(cls, rule_type: str, params: Dict[str, Any], error_message: str):
        """Create a custom validation rule."""
        return cls(
            rule_type=rule_type, 
            params=params, 
            error_message=error_message
        )


class FormField(BaseModel):
    """Definition of a form field."""
    name: str
    label: str
    field_type: FieldType
    required: bool = True
    default_value: Any = None
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    validation_rules: List[ValidationRule] = Field(default_factory=list)
    options: Optional[List[Dict[str, Any]]] = None
    
    @field_validator('options')
    @classmethod
    def validate_options(cls, v, info):
        """Validate that options are provided for select, checkbox, and radio fields."""
        field_type = info.data.get('field_type')
        if field_type in [FieldType.SELECT, FieldType.CHECKBOX, FieldType.RADIO] and not v:
            raise ValueError(f"Options must be provided for {field_type.value} fields")
        return v


class FormStepDefinition(BaseModel):
    """Definition of a form step."""
    step_id: str
    title: str
    description: Optional[str] = None
    fields: List[FormField]
    next_steps: List[str] = Field(default_factory=list)
    previous_step: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_fields_unique(self):
        """Validate that field names are unique within a step."""
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("Field names must be unique within a step")
        return self


class ValidationError(BaseModel):
    """Validation error for a form field."""
    field: str
    error: str


class FormStepState(BaseModel):
    """State of a form step."""
    step_id: str
    status: FormStepStatus = FormStepStatus.NOT_STARTED
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[ValidationError] = Field(default_factory=list)
    visited: bool = False
    completed_at: Optional[datetime] = None


class MultiStepForm(BaseModel):
    """Multi-step form with validation."""
    form_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    steps: Dict[str, FormStepDefinition]
    current_step_id: Optional[str] = None
    step_states: Dict[str, FormStepState] = Field(default_factory=dict)
    start_step_id: str
    submission_data: Dict[str, Any] = Field(default_factory=dict)
    is_submitted: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, **data):
        """Initialize the multi-step form."""
        super().__init__(**data)
        
        # Initialize step states if not provided
        if not self.step_states:
            for step_id, step_def in self.steps.items():
                self.step_states[step_id] = FormStepState(step_id=step_id)
        
        # Set current step to start step if not set
        if not self.current_step_id:
            self.current_step_id = self.start_step_id
            self.step_states[self.start_step_id].status = FormStepStatus.IN_PROGRESS
            self.step_states[self.start_step_id].visited = True
    
    def get_current_step(self) -> FormStepDefinition:
        """Get the current step definition."""
        if not self.current_step_id:
            raise ValueError("No current step set")
        return self.steps[self.current_step_id]
    
    def get_current_step_state(self) -> FormStepState:
        """Get the current step state."""
        if not self.current_step_id:
            raise ValueError("No current step set")
        return self.step_states[self.current_step_id]
    
    def validate_field(self, field: FormField, value: Any) -> List[str]:
        """Validate a field value against its validation rules."""
        errors = []
        
        # Check required field
        if field.required and (value is None or value == ""):
            errors.append("This field is required")
            return errors
        
        # Skip further validation if value is empty and field is not required
        if not field.required and (value is None or value == ""):
            return errors
        
        # Apply validation rules
        for rule in field.validation_rules:
            if rule.rule_type == "required":
                # Already checked above
                continue
                
            elif rule.rule_type == "min_length":
                min_length = rule.params.get("min_length", 0)
                if len(str(value)) < min_length:
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "max_length":
                max_length = rule.params.get("max_length", float('inf'))
                if len(str(value)) > max_length:
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "pattern":
                import re
                pattern = rule.params.get("pattern", "")
                if not re.match(pattern, str(value)):
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "email":
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "min_value":
                min_value = rule.params.get("min_value", float('-inf'))
                try:
                    if float(value) < min_value:
                        errors.append(rule.error_message)
                except (ValueError, TypeError):
                    errors.append("Invalid number format")
                    
            elif rule.rule_type == "max_value":
                max_value = rule.params.get("max_value", float('inf'))
                try:
                    if float(value) > max_value:
                        errors.append(rule.error_message)
                except (ValueError, TypeError):
                    errors.append("Invalid number format")
                    
            elif rule.rule_type == "options":
                options = rule.params.get("options", [])
                if value not in options:
                    errors.append(rule.error_message)
        
        return errors
    
    def validate_step(self, step_id: str, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate a step's data."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} does not exist")
        
        step = self.steps[step_id]
        errors = []
        
        # Validate each field
        for field in step.fields:
            field_value = data.get(field.name, field.default_value)
            field_errors = self.validate_field(field, field_value)
            
            for error in field_errors:
                errors.append(ValidationError(field=field.name, error=error))
        
        return errors
    
    def update_step_data(self, step_id: str, data: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """Update a step's data and optionally validate it."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} does not exist")
        
        step_state = self.step_states[step_id]
        
        # Update the data
        step_state.data.update(data)
        
        # Validate if requested
        if validate:
            errors = self.validate_step(step_id, step_state.data)
            step_state.errors = errors
            
            if errors:
                step_state.status = FormStepStatus.INVALID
            else:
                step_state.status = FormStepStatus.IN_PROGRESS
        
        self.updated_at = datetime.now()
        return {"success": True, "errors": step_state.errors}
    
    def complete_step(self, step_id: str) -> Dict[str, Any]:
        """Mark a step as completed after validation."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} does not exist")
        
        step_state = self.step_states[step_id]
        
        # Validate the step
        errors = self.validate_step(step_id, step_state.data)
        step_state.errors = errors
        
        if errors:
            step_state.status = FormStepStatus.INVALID
            return {"success": False, "errors": errors}
        
        # Mark as completed
        step_state.status = FormStepStatus.COMPLETED
        step_state.completed_at = datetime.now()
        self.updated_at = datetime.now()
        
        return {"success": True}
    
    def can_proceed_to_step(self, target_step_id: str) -> bool:
        """Check if the user can proceed to the target step."""
        if target_step_id not in self.steps:
            return False
        
        # Can always go to the start step
        if target_step_id == self.start_step_id:
            return True
        
        # Check if the target step is a valid next step from the current step
        current_step = self.get_current_step()
        if target_step_id in current_step.next_steps:
            # Current step must be completed
            current_state = self.get_current_step_state()
            return current_state.status == FormStepStatus.COMPLETED
        
        # Check if the target step is a previous step that has been visited
        target_state = self.step_states[target_step_id]
        return target_state.visited
    
    def go_to_step(self, step_id: str) -> Dict[str, Any]:
        """Go to a specific step if allowed."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} does not exist")
        
        if not self.can_proceed_to_step(step_id):
            return {
                "success": False,
                "message": "Cannot proceed to this step"
            }
        
        # Update current step
        self.current_step_id = step_id
        
        # Mark step as visited and in progress if not completed
        step_state = self.step_states[step_id]
        step_state.visited = True
        if step_state.status != FormStepStatus.COMPLETED:
            step_state.status = FormStepStatus.IN_PROGRESS
        
        self.updated_at = datetime.now()
        
        return {
            "success": True,
            "step_id": step_id,
            "step_data": step_state.data
        }
    
    def next_step(self) -> Dict[str, Any]:
        """Go to the next step if the current step is valid."""
        current_step = self.get_current_step()
        
        # Complete the current step first
        result = self.complete_step(self.current_step_id)
        if not result["success"]:
            return {
                "success": False,
                "message": "Current step has validation errors",
                "errors": result["errors"]
            }
        
        # If there are no next steps, we're at the end
        if not current_step.next_steps:
            return {
                "success": False,
                "message": "No next step available"
            }
        
        # Go to the first available next step
        return self.go_to_step(current_step.next_steps[0])
    
    def previous_step(self) -> Dict[str, Any]:
        """Go to the previous step."""
        current_step = self.get_current_step()
        
        if not current_step.previous_step:
            return {
                "success": False,
                "message": "No previous step available"
            }
        
        return self.go_to_step(current_step.previous_step)
    
    def is_form_complete(self) -> bool:
        """Check if all steps in the form are completed."""
        return all(
            state.status == FormStepStatus.COMPLETED
            for state in self.step_states.values()
        )
    
    def get_form_data(self) -> Dict[str, Any]:
        """Get all data from all steps."""
        data = {}
        for step_id, state in self.step_states.items():
            data.update(state.data)
        return data
    
    def submit_form(self) -> Dict[str, Any]:
        """Submit the form if all steps are completed."""
        # Check if all steps are completed
        incomplete_steps = []
        for step_id, state in self.step_states.items():
            if state.status != FormStepStatus.COMPLETED:
                incomplete_steps.append(step_id)
        
        if incomplete_steps:
            return {
                "success": False,
                "message": "Form has incomplete steps",
                "incomplete_steps": incomplete_steps
            }
        
        # Collect all data
        self.submission_data = self.get_form_data()
        self.is_submitted = True
        self.updated_at = datetime.now()
        
        return {
            "success": True,
            "message": "Form submitted successfully",
            "data": self.submission_data
        }
    
    def reset_form(self) -> Dict[str, Any]:
        """Reset the form to its initial state."""
        # Reset step states
        for step_id in self.steps:
            self.step_states[step_id] = FormStepState(step_id=step_id)
        
        # Reset to start step
        self.current_step_id = self.start_step_id
        self.step_states[self.start_step_id].status = FormStepStatus.IN_PROGRESS
        self.step_states[self.start_step_id].visited = True
        
        # Reset submission data
        self.submission_data = {}
        self.is_submitted = False
        self.updated_at = datetime.now()
        
        return {
            "success": True,
            "message": "Form reset successfully"
        }


# Example usage
if __name__ == "__main__":
    # Define a multi-step registration form
    personal_info_step = FormStepDefinition(
        step_id="personal_info",
        title="Personal Information",
        description="Please provide your personal details",
        fields=[
            FormField(
                name="first_name",
                label="First Name",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_length(2),
                    ValidationRule.max_length(50)
                ]
            ),
            FormField(
                name="last_name",
                label="Last Name",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_length(2),
                    ValidationRule.max_length(50)
                ]
            ),
            FormField(
                name="email",
                label="Email Address",
                field_type=FieldType.EMAIL,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.email()
                ]
            ),
            FormField(
                name="age",
                label="Age",
                field_type=FieldType.NUMBER,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_value(18),
                    ValidationRule.max_value(120)
                ]
            )
        ],
        next_steps=["address"]
    )
    
    address_step = FormStepDefinition(
        step_id="address",
        title="Address Information",
        description="Please provide your address details",
        fields=[
            FormField(
                name="street",
                label="Street Address",
                field_type=FieldType.TEXT,
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="city",
                label="City",
                field_type=FieldType.TEXT,
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="state",
                label="State/Province",
                field_type=FieldType.SELECT,
                options=[
                    {"value": "CA", "label": "California"},
                    {"value": "NY", "label": "New York"},
                    {"value": "TX", "label": "Texas"}
                ],
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="zip_code",
                label="ZIP/Postal Code",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.pattern(r'^\d{5}(-\d{4})?$', "Invalid ZIP code format")
                ]
            )
        ],
        next_steps=["preferences"],
        previous_step="personal_info"
    )
    
    preferences_step = FormStepDefinition(
        step_id="preferences",
        title="Preferences",
        description="Please tell us about your preferences",
        fields=[
            FormField(
                name="interests",
                label="Interests",
                field_type=FieldType.CHECKBOX,
                options=[
                    {"value": "tech", "label": "Technology"},
                    {"value": "sports", "label": "Sports"},
                    {"value": "music", "label": "Music"},
                    {"value": "travel", "label": "Travel"}
                ],
                required=False
            ),
            FormField(
                name="contact_method",
                label="Preferred Contact Method",
                field_type=FieldType.RADIO,
                options=[
                    {"value": "email", "label": "Email"},
                    {"value": "phone", "label": "Phone"},
                    {"value": "mail", "label": "Mail"}
                ],
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="comments",
                label="Additional Comments",
                field_type=FieldType.TEXTAREA,
                required=False
            )
        ],
        next_steps=["review"],
        previous_step="address"
    )
    
    review_step = FormStepDefinition(
        step_id="review",
        title="Review",
        description="Please review your information before submitting",
        fields=[],  # No input fields on review step
        previous_step="preferences"
    )
    
    # Create the multi-step form
    registration_form = MultiStepForm(
        title="User Registration",
        description="Complete this form to register for our service",
        steps={
            "personal_info": personal_info_step,
            "address": address_step,
            "preferences": preferences_step,
            "review": review_step
        },
        start_step_id="personal_info"
    )
    
    # Example usage
    print(f"Starting form: {registration_form.title}")
    print(f"Current step: {registration_form.get_current_step().title}")
    
    # Fill out personal info step
    personal_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30
    }
    
    result = registration_form.update_step_data("personal_info", personal_data)
    print(f"Updated personal info: {result}")
    
    # Complete and go to next step
    result = registration_form.next_step()
    print(f"Next step result: {result}")
    print(f"Current step: {registration_form.get_current_step().title}")
    
    # Fill out address step
    address_data = {
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip_code": "94105"
    }
    
    result = registration_form.update_step_data("address", address_data)
    print(f"Updated address info: {result}")
    
    # Complete and go to next step
    result = registration_form.next_step()
    print(f"Next step result: {result}")
    print(f"Current step: {registration_form.get_current_step().title}")
    
    # Fill out preferences step
    preferences_data = {
        "interests": ["tech", "travel"],
        "contact_method": "email",
        "comments": "Looking forward to using your service!"
    }
    
    result = registration_form.update_step_data("preferences", preferences_data)
    print(f"Updated preferences: {result}")
    
    # Complete and go to next step
    result = registration_form.next_step()
    print(f"Next step result: {result}")
    print(f"Current step: {registration_form.get_current_step().title}")
    
    # Submit the form
    result = registration_form.submit_form()
    print(f"Form submission result: {result}")
    
    # Check if form is complete
    print(f"Is form complete? {registration_form.is_form_complete()}")
    print(f"Is form submitted? {registration_form.is_submitted}")
