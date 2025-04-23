"""
Lesson 4.2 Exercises: Error Handling and Recovery
------------------------------------------------
This module contains solutions for the exercises in Lesson 4.2.
"""

from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Set
from enum import Enum
import re
from datetime import date, datetime


# Exercise 1: Multi-step Form with Error Preservation
# -------------------------------------------------
# Create a validation system for a multi-step form that preserves valid data 
# between steps and only shows errors for the current step.

class FormStep(Enum):
    """Enum for form steps."""
    PERSONAL = "personal"
    CONTACT = "contact"
    PREFERENCES = "preferences"
    REVIEW = "review"


class PersonalInfo(BaseModel):
    """Personal information form step."""
    first_name: str = Field(..., min_length=2)
    last_name: str = Field(..., min_length=2)
    birth_date: date
    gender: Optional[str] = None


class ContactInfo(BaseModel):
    """Contact information form step."""
    email: str
    phone: Optional[str] = None
    address: str = Field(..., min_length=5)
    city: str
    country: str
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        if v is not None and not re.match(r"^\+?[\d\s-]{10,15}$", v):
            raise ValueError("Invalid phone number format")
        return v


class Preferences(BaseModel):
    """User preferences form step."""
    interests: List[str] = []
    subscribe_newsletter: bool = False
    theme: str = "light"
    language: str = "en"


class MultiStepForm:
    """Multi-step form manager with error preservation."""
    
    def __init__(self):
        """Initialize the form with empty data."""
        self.current_step = FormStep.PERSONAL
        self.data = {}
        self.errors = {}
        self.completed_steps = set()
    
    def get_step_model(self, step: FormStep):
        """Get the Pydantic model for a step."""
        step_models = {
            FormStep.PERSONAL: PersonalInfo,
            FormStep.CONTACT: ContactInfo,
            FormStep.PREFERENCES: Preferences,
        }
        return step_models.get(step)
    
    def submit_step(self, step: FormStep, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit data for a specific step.
        
        Args:
            step: The form step
            step_data: The data for this step
            
        Returns:
            Dict with status and errors if any
        """
        # Verify this is the current step
        if step != self.current_step:
            return {
                "status": "error",
                "message": f"Current step is {self.current_step.value}, not {step.value}"
            }
        
        # Get the model for this step
        model = self.get_step_model(step)
        if not model:
            return {
                "status": "error",
                "message": f"Invalid step: {step.value}"
            }
        
        # Validate the step data
        try:
            validated_data = model(**step_data)
            
            # Store the validated data
            self.data[step.value] = validated_data.model_dump()
            
            # Clear any errors for this step
            if step.value in self.errors:
                del self.errors[step.value]
            
            # Mark this step as completed
            self.completed_steps.add(step)
            
            # Move to the next step
            self._advance_step()
            
            return {
                "status": "success",
                "message": f"Step {step.value} completed",
                "next_step": self.current_step.value
            }
        except ValidationError as e:
            # Store the errors for this step
            self.errors[step.value] = e.errors()
            
            return {
                "status": "error",
                "message": f"Validation failed for step {step.value}",
                "errors": e.errors()
            }
    
    def _advance_step(self):
        """Advance to the next step."""
        if self.current_step == FormStep.PERSONAL:
            self.current_step = FormStep.CONTACT
        elif self.current_step == FormStep.CONTACT:
            self.current_step = FormStep.PREFERENCES
        elif self.current_step == FormStep.PREFERENCES:
            self.current_step = FormStep.REVIEW
    
    def go_back(self) -> Dict[str, Any]:
        """Go back to the previous step."""
        if self.current_step == FormStep.CONTACT:
            self.current_step = FormStep.PERSONAL
        elif self.current_step == FormStep.PREFERENCES:
            self.current_step = FormStep.CONTACT
        elif self.current_step == FormStep.REVIEW:
            self.current_step = FormStep.PREFERENCES
        else:
            return {
                "status": "error",
                "message": "Already at the first step"
            }
        
        return {
            "status": "success",
            "message": f"Moved back to step {self.current_step.value}",
            "current_step": self.current_step.value,
            "step_data": self.data.get(self.current_step.value, {})
        }
    
    def get_current_step_data(self) -> Dict[str, Any]:
        """Get the data for the current step."""
        return self.data.get(self.current_step.value, {})
    
    def get_form_summary(self) -> Dict[str, Any]:
        """Get a summary of the form data and status."""
        return {
            "current_step": self.current_step.value,
            "completed_steps": [step.value for step in self.completed_steps],
            "data": self.data,
            "errors": self.errors,
            "is_complete": self.current_step == FormStep.REVIEW and len(self.errors) == 0
        }


# Exercise 2: Suggestion System for Common Errors
# ---------------------------------------------
# Implement a "suggestion" system that proposes corrections for common validation errors.

class EmailSuggestionEngine:
    """Suggestion engine for email validation errors."""
    
    def __init__(self):
        """Initialize with common domains and corrections."""
        self.common_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
            "icloud.com", "aol.com", "protonmail.com", "mail.com"
        ]
        
        self.common_typos = {
            "gmial": "gmail",
            "gmil": "gmail",
            "gmal": "gmail",
            "gamil": "gmail",
            "yaho": "yahoo",
            "yhaoo": "yahoo",
            "hotmial": "hotmail",
            "hotmil": "hotmail",
            "outlok": "outlook",
            "outloo": "outlook",
            "icould": "icloud",
            "iclod": "icloud"
        }
    
    def suggest_corrections(self, email: str) -> List[str]:
        """
        Suggest corrections for an invalid email.
        
        Args:
            email: The invalid email address
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        # Check if it's missing @ symbol
        if '@' not in email:
            for domain in self.common_domains:
                suggestions.append(f"{email}@{domain}")
            return suggestions
        
        # Split into username and domain
        username, domain = email.split('@', 1)
        
        # Check for domain typos
        domain_name = domain.split('.')[0] if '.' in domain else domain
        domain_extension = domain.split('.')[1] if '.' in domain and len(domain.split('.')) > 1 else "com"
        
        # Check for common domain typos
        if domain_name in self.common_typos:
            corrected_domain = f"{self.common_typos[domain_name]}.{domain_extension}"
            suggestions.append(f"{username}@{corrected_domain}")
        
        # Check for missing or incorrect TLD
        if '.' not in domain:
            for common_domain in self.common_domains:
                if domain in common_domain:
                    suggestions.append(f"{username}@{common_domain}")
        
        # Find close matches in common domains
        for common_domain in self.common_domains:
            # Simple string distance (not efficient but works for demo)
            if self._similar(domain, common_domain):
                suggestions.append(f"{username}@{common_domain}")
        
        return suggestions
    
    def _similar(self, a: str, b: str) -> bool:
        """Simple string similarity check."""
        # For a real implementation, use Levenshtein distance or similar
        if a == b:
            return False  # Already exact match
        
        # Check if one is a substring of the other
        if a in b or b in a:
            return True
        
        # Check if they share a prefix
        prefix_length = min(len(a), len(b)) // 2
        if a[:prefix_length] == b[:prefix_length]:
            return True
        
        return False


class ValidationWithSuggestions:
    """Validation system with suggestions for common errors."""
    
    def __init__(self):
        """Initialize with suggestion engines."""
        self.email_engine = EmailSuggestionEngine()
    
    def validate_with_suggestions(self, model_class, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data and provide suggestions for errors.
        
        Args:
            model_class: Pydantic model class
            data: Data to validate
            
        Returns:
            Dict with validation result and suggestions
        """
        try:
            # Try to validate
            validated = model_class(**data)
            return {
                "status": "success",
                "data": validated.model_dump()
            }
        except ValidationError as e:
            # Process errors and generate suggestions
            suggestions = {}
            
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'general'
                error_type = error['type']
                
                # Generate suggestions based on field and error type
                if field == 'email' and error_type == 'value_error':
                    if 'email' in data:
                        email_suggestions = self.email_engine.suggest_corrections(data['email'])
                        if email_suggestions:
                            suggestions['email'] = email_suggestions
                
                # Add more suggestion generators for other fields/error types
            
            return {
                "status": "error",
                "message": "Validation failed",
                "errors": e.errors(),
                "suggestions": suggestions
            }


# Exercise 3: Error Logging System
# ------------------------------
# Build an error logging system that tracks validation errors to identify common user mistakes.

class ValidationErrorLogger:
    """System to log and analyze validation errors."""
    
    def __init__(self):
        """Initialize with empty error logs."""
        self.error_logs = []
    
    def log_error(self, form_id: str, model_name: str, data: Dict[str, Any], 
                 errors: List[Dict[str, Any]]):
        """
        Log a validation error.
        
        Args:
            form_id: Identifier for the form
            model_name: Name of the Pydantic model
            data: The data that failed validation
            errors: The validation errors
        """
        self.error_logs.append({
            "timestamp": datetime.now().isoformat(),
            "form_id": form_id,
            "model_name": model_name,
            "data": data,
            "errors": errors
        })
    
    def get_error_frequency(self) -> Dict[str, int]:
        """
        Get frequency of errors by field.
        
        Returns:
            Dict mapping field names to error counts
        """
        field_errors = {}
        
        for log in self.error_logs:
            for error in log["errors"]:
                field = '.'.join(str(loc) for loc in error['loc'])
                
                if field not in field_errors:
                    field_errors[field] = 0
                
                field_errors[field] += 1
        
        return field_errors
    
    def get_common_error_types(self) -> Dict[str, int]:
        """
        Get frequency of error types.
        
        Returns:
            Dict mapping error types to counts
        """
        error_types = {}
        
        for log in self.error_logs:
            for error in log["errors"]:
                error_type = error['type']
                
                if error_type not in error_types:
                    error_types[error_type] = 0
                
                error_types[error_type] += 1
        
        return error_types
    
    def get_error_trends(self, time_window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get error trends over time.
        
        Args:
            time_window: Optional time window in hours
            
        Returns:
            List of error counts by timestamp
        """
        if not self.error_logs:
            return []
        
        # Filter by time window if specified
        filtered_logs = self.error_logs
        if time_window:
            cutoff = datetime.now() - datetime.timedelta(hours=time_window)
            cutoff_str = cutoff.isoformat()
            filtered_logs = [log for log in self.error_logs if log["timestamp"] > cutoff_str]
        
        # Group by hour
        hourly_counts = {}
        for log in filtered_logs:
            timestamp = datetime.fromisoformat(log["timestamp"])
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            
            if hour_key not in hourly_counts:
                hourly_counts[hour_key] = 0
            
            hourly_counts[hour_key] += 1
        
        # Convert to list of dicts
        return [{"timestamp": k, "count": v} for k, v in hourly_counts.items()]
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate suggestions for form improvements based on error patterns.
        
        Returns:
            List of improvement suggestions
        """
        field_errors = self.get_error_frequency()
        error_types = self.get_common_error_types()
        
        suggestions = []
        
        # Suggest improvements for fields with high error rates
        for field, count in sorted(field_errors.items(), key=lambda x: x[1], reverse=True):
            if count > 5:  # Arbitrary threshold
                suggestions.append({
                    "field": field,
                    "error_count": count,
                    "suggestion": f"Consider improving the {field} field - it has a high error rate"
                })
        
        # Suggest improvements based on error types
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            if count > 5:  # Arbitrary threshold
                if error_type == "value_error.email":
                    suggestions.append({
                        "error_type": error_type,
                        "error_count": count,
                        "suggestion": "Consider adding email format examples or auto-correction"
                    })
                elif error_type == "type_error.integer":
                    suggestions.append({
                        "error_type": error_type,
                        "error_count": count,
                        "suggestion": "Consider using a numeric input field with validation"
                    })
                elif "string_too_short" in error_type:
                    suggestions.append({
                        "error_type": error_type,
                        "error_count": count,
                        "suggestion": "Make minimum length requirements more visible"
                    })
        
        return suggestions


# Exercise 4: Validation Middleware for Web API
# ------------------------------------------
# Create a validation middleware for a web API that standardizes error responses.

class APIResponse(BaseModel):
    """Standardized API response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, Any]] = None


class ValidationMiddleware:
    """Middleware for standardizing validation in API endpoints."""
    
    @staticmethod
    def process_request(model_class, request_data: Dict[str, Any]) -> APIResponse:
        """
        Process an API request with validation.
        
        Args:
            model_class: Pydantic model for validation
            request_data: Request data to validate
            
        Returns:
            Standardized APIResponse
        """
        try:
            # Validate request data
            validated_data = model_class(**request_data)
            
            # Return success response
            return APIResponse(
                success=True,
                message="Request processed successfully",
                data=validated_data.model_dump()
            )
        except ValidationError as e:
            # Format errors in a standardized way
            formatted_errors = {}
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                
                if field not in formatted_errors:
                    formatted_errors[field] = []
                
                formatted_errors[field].append({
                    "code": error['type'],
                    "message": error['msg']
                })
            
            # Return error response
            return APIResponse(
                success=False,
                message="Validation failed",
                errors=formatted_errors
            )


# Exercise 5: Partial Submission Form
# ---------------------------------
# Implement a form that allows partial submissions, saving valid fields as draft data.

class PartialSubmissionForm(BaseModel):
    """Base class for forms that support partial submissions."""
    
    @classmethod
    def validate_partial(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fields individually, keeping valid ones.
        
        Args:
            data: Form data to validate
            
        Returns:
            Dict with validation results
        """
        valid_fields = {}
        errors = {}
        
        # Get field definitions from the model
        for field_name, field in cls.model_fields.items():
            # Skip if field not in data
            if field_name not in data:
                continue
            
            field_value = data[field_name]
            
            # Create a temporary dict with just this field
            temp_data = {field_name: field_value}
            
            try:
                # Try to validate just this field
                # We create a new model instance with just this field
                partial_instance = cls.model_validate({field_name: field_value})
                
                # If validation passes, add to valid fields
                valid_fields[field_name] = getattr(partial_instance, field_name)
            except ValidationError as e:
                # If validation fails, add to errors
                for error in e.errors():
                    # Only keep errors for this field
                    if error['loc'][0] == field_name:
                        if field_name not in errors:
                            errors[field_name] = []
                        errors[field_name].append(error)
        
        return {
            "valid_fields": valid_fields,
            "errors": errors,
            "is_complete": len(valid_fields) == len(cls.model_fields) and not errors
        }


class JobApplication(PartialSubmissionForm):
    """Job application form with partial submission support."""
    full_name: str = Field(..., min_length=3)
    email: str
    phone: str
    resume_url: str
    cover_letter: Optional[str] = None
    years_experience: int = Field(..., ge=0)
    skills: List[str] = []
    
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        if not re.match(r"^\+?[\d\s-]{10,15}$", v):
            raise ValueError("Invalid phone number format")
        return v
    
    @field_validator('resume_url')
    def validate_resume_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Resume URL must start with http:// or https://")
        return v


class DraftManager:
    """Manager for form drafts with partial validation."""
    
    def __init__(self):
        """Initialize with empty drafts."""
        self.drafts = {}
    
    def save_draft(self, user_id: str, form_type: str, 
                  form_class: type, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a form draft with partial validation.
        
        Args:
            user_id: User identifier
            form_type: Type of form
            form_class: Pydantic model class
            data: Form data
            
        Returns:
            Dict with draft status
        """
        # Validate partial submission
        validation_result = form_class.validate_partial(data)
        
        # Create draft key
        draft_key = f"{user_id}:{form_type}"
        
        # Get existing draft or create new one
        existing_draft = self.drafts.get(draft_key, {"valid_fields": {}, "errors": {}})
        
        # Update with new valid fields
        existing_draft["valid_fields"].update(validation_result["valid_fields"])
        
        # Update errors
        existing_draft["errors"] = validation_result["errors"]
        
        # Update completion status
        existing_draft["is_complete"] = validation_result["is_complete"]
        
        # Save updated draft
        self.drafts[draft_key] = existing_draft
        
        return {
            "status": "success",
            "message": "Draft saved",
            "draft": existing_draft
        }
    
    def get_draft(self, user_id: str, form_type: str) -> Dict[str, Any]:
        """
        Get a saved draft.
        
        Args:
            user_id: User identifier
            form_type: Type of form
            
        Returns:
            Dict with draft data or error
        """
        draft_key = f"{user_id}:{form_type}"
        
        if draft_key not in self.drafts:
            return {
                "status": "error",
                "message": "No draft found"
            }
        
        return {
            "status": "success",
            "draft": self.drafts[draft_key]
        }
    
    def submit_form(self, user_id: str, form_type: str, 
                   form_class: type) -> Dict[str, Any]:
        """
        Submit a form from a draft.
        
        Args:
            user_id: User identifier
            form_type: Type of form
            form_class: Pydantic model class
            
        Returns:
            Dict with submission status
        """
        draft_key = f"{user_id}:{form_type}"
        
        if draft_key not in self.drafts:
            return {
                "status": "error",
                "message": "No draft found"
            }
        
        draft = self.drafts[draft_key]
        
        if not draft["is_complete"]:
            return {
                "status": "error",
                "message": "Form is incomplete",
                "missing_fields": [field for field in form_class.model_fields 
                                  if field not in draft["valid_fields"]],
                "errors": draft["errors"]
            }
        
        # Form is complete, create a validated instance
        try:
            form_instance = form_class(**draft["valid_fields"])
            
            # In a real app, you would save to database here
            
            # Clear the draft
            del self.drafts[draft_key]
            
            return {
                "status": "success",
                "message": "Form submitted successfully",
                "data": form_instance.model_dump()
            }
        except ValidationError as e:
            # This shouldn't happen if is_complete was true
            return {
                "status": "error",
                "message": "Unexpected validation error",
                "errors": e.errors()
            }


# Example Usage
# -----------

if __name__ == "__main__":
    print("Exercise 1: Multi-step Form")
    print("-" * 40)
    form = MultiStepForm()
    
    # Submit personal info step
    result = form.submit_step(FormStep.PERSONAL, {
        "first_name": "John",
        "last_name": "Doe",
        "birth_date": "1990-01-01",
        "gender": "male"
    })
    print(f"Personal info step: {result}")
    
    # Submit contact info step
    result = form.submit_step(FormStep.CONTACT, {
        "email": "invalid-email",  # This will fail
        "address": "123 Main St",
        "city": "Anytown",
        "country": "USA"
    })
    print(f"Contact info step: {result}")
    print("\n")
    
    print("Exercise 2: Suggestion System")
    print("-" * 40)
    validator = ValidationWithSuggestions()
    
    class UserProfile(BaseModel):
        name: str
        email: str
        
        @field_validator('email')
        def validate_email(cls, v):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
                raise ValueError("Invalid email format")
            return v
    
    result = validator.validate_with_suggestions(UserProfile, {
        "name": "John Doe",
        "email": "john.doegmail.com"  # Missing @ symbol
    })
    print(f"Validation with suggestions: {result}")
    print("\n")
    
    print("Exercise 3: Error Logging System")
    print("-" * 40)
    logger = ValidationErrorLogger()
    
    # Log some errors
    logger.log_error("form1", "UserProfile", {"name": "John", "email": "invalid"}, 
                    [{"loc": ("email",), "type": "value_error.email", "msg": "Invalid email"}])
    
    logger.log_error("form2", "UserProfile", {"name": "Jane", "email": "invalid2"}, 
                    [{"loc": ("email",), "type": "value_error.email", "msg": "Invalid email"}])
    
    logger.log_error("form3", "ProductForm", {"name": "", "price": -10}, 
                    [
                        {"loc": ("name",), "type": "string_too_short", "msg": "String too short"},
                        {"loc": ("price",), "type": "greater_than", "msg": "Must be > 0"}
                    ])
    
    print(f"Error frequency: {logger.get_error_frequency()}")
    print(f"Common error types: {logger.get_common_error_types()}")
    print(f"Improvement suggestions: {logger.get_improvement_suggestions()}")
    print("\n")
    
    print("Exercise 4: Validation Middleware")
    print("-" * 40)
    
    class CreateUser(BaseModel):
        username: str = Field(..., min_length=3)
        email: str
        password: str = Field(..., min_length=8)
    
    # Valid request
    valid_request = {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "password123"
    }
    
    # Invalid request
    invalid_request = {
        "username": "jo",
        "email": "invalid-email",
        "password": "123"
    }
    
    middleware = ValidationMiddleware()
    
    valid_response = middleware.process_request(CreateUser, valid_request)
    print(f"Valid request response: {valid_response}")
    
    invalid_response = middleware.process_request(CreateUser, invalid_request)
    print(f"Invalid request response: {invalid_response}")
    print("\n")
    
    print("Exercise 5: Partial Submission Form")
    print("-" * 40)
    
    draft_manager = DraftManager()
    
    # Save partial draft with some valid and some invalid fields
    draft_result = draft_manager.save_draft(
        "user123", 
        "job_application",
        JobApplication,
        {
            "full_name": "John Doe",
            "email": "john@example.com",
            "phone": "invalid-phone",  # This will fail
            "resume_url": "not-a-url",  # This will fail
            "years_experience": 5
        }
    )
    
    print(f"Draft save result: {draft_result}")
    
    # Update the draft with corrections
    update_result = draft_manager.save_draft(
        "user123", 
        "job_application",
        JobApplication,
        {
            "phone": "+1 555-123-4567",  # Fixed
            "resume_url": "https://example.com/resume.pdf",  # Fixed
            "skills": ["Python", "Pydantic", "Error Handling"]
        }
    )
    
    print(f"Draft update result: {update_result}")
    
    # Try to submit (should succeed now)
    submit_result = draft_manager.submit_form(
        "user123", 
        "job_application",
        JobApplication
    )
    
    print(f"Submit result: {submit_result}")
