"""
Exercise 4.3.5: Form Builder

This exercise implements a form builder that generates both Pydantic models
for validation and HTML form elements from a single definition.
"""

from pydantic import BaseModel, Field, field_validator, create_model
from typing import Dict, Any, Type, Optional, List, Union, Callable, get_type_hints
from datetime import date, datetime
import re
import html


class FormField(BaseModel):
    """Base form field definition."""
    name: str
    label: str
    required: bool = True
    help_text: Optional[str] = None
    css_classes: List[str] = Field(default_factory=list)
    validators: List[Callable] = Field(default_factory=list, exclude=True)
    
    def get_html(self) -> str:
        """Generate HTML for this field (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_html()")
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_field_type()")
    
    def get_field_constraints(self) -> Dict[str, Any]:
        """Get Pydantic field constraints for this field."""
        constraints = {}
        
        if self.help_text:
            constraints["description"] = self.help_text
            
        return constraints
    
    def get_field_default(self):
        """Get the default value for this field."""
        if self.required:
            return ...
        return None
    
    def get_common_attributes(self) -> str:
        """Get common HTML attributes for this field."""
        attrs = [
            f'id="{html.escape(self.name)}"',
            f'name="{html.escape(self.name)}"'
        ]
        
        if self.required:
            attrs.append('required')
            
        if self.css_classes:
            class_str = " ".join(html.escape(cls) for cls in self.css_classes)
            attrs.append(f'class="{class_str}"')
            
        return " ".join(attrs)


class StringField(FormField):
    """String input field."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    placeholder: Optional[str] = None
    input_type: str = "text"  # text, email, password, tel, url, etc.
    
    def get_html(self) -> str:
        """Generate HTML for a string input field."""
        attrs = [self.get_common_attributes()]
        
        attrs.append(f'type="{html.escape(self.input_type)}"')
        
        if self.min_length is not None:
            attrs.append(f'minlength="{self.min_length}"')
            
        if self.max_length is not None:
            attrs.append(f'maxlength="{self.max_length}"')
            
        if self.pattern:
            attrs.append(f'pattern="{html.escape(self.pattern)}"')
            
        if self.placeholder:
            attrs.append(f'placeholder="{html.escape(self.placeholder)}"')
        
        html_parts = [
            f'<div class="form-group">',
            f'  <label for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
            f'  <input {" ".join(attrs)}>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return str
    
    def get_field_constraints(self) -> Dict[str, Any]:
        """Get Pydantic field constraints for this field."""
        constraints = super().get_field_constraints()
        
        if self.min_length is not None:
            constraints["min_length"] = self.min_length
            
        if self.max_length is not None:
            constraints["max_length"] = self.max_length
            
        if self.pattern:
            constraints["pattern"] = self.pattern
            
        return constraints


class NumberField(FormField):
    """Numeric input field."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    placeholder: Optional[str] = None
    is_integer: bool = False
    
    def get_html(self) -> str:
        """Generate HTML for a number input field."""
        attrs = [self.get_common_attributes()]
        
        attrs.append('type="number"')
        
        if self.min_value is not None:
            attrs.append(f'min="{self.min_value}"')
            
        if self.max_value is not None:
            attrs.append(f'max="{self.max_value}"')
            
        if self.step is not None:
            attrs.append(f'step="{self.step}"')
        elif self.is_integer:
            attrs.append('step="1"')
            
        if self.placeholder:
            attrs.append(f'placeholder="{html.escape(self.placeholder)}"')
        
        html_parts = [
            f'<div class="form-group">',
            f'  <label for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
            f'  <input {" ".join(attrs)}>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return int if self.is_integer else float
    
    def get_field_constraints(self) -> Dict[str, Any]:
        """Get Pydantic field constraints for this field."""
        constraints = super().get_field_constraints()
        
        if self.min_value is not None:
            constraints["ge"] = self.min_value
            
        if self.max_value is not None:
            constraints["le"] = self.max_value
            
        return constraints


class BooleanField(FormField):
    """Boolean checkbox field."""
    default: bool = False
    
    def get_html(self) -> str:
        """Generate HTML for a checkbox field."""
        attrs = [self.get_common_attributes()]
        
        attrs.append('type="checkbox"')
        
        if self.default:
            attrs.append('checked')
        
        html_parts = [
            f'<div class="form-check">',
            f'  <input {" ".join(attrs)}>',
            f'  <label class="form-check-label" for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return bool
    
    def get_field_default(self):
        """Get the default value for this field."""
        return self.default


class DateField(FormField):
    """Date input field."""
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    
    def get_html(self) -> str:
        """Generate HTML for a date input field."""
        attrs = [self.get_common_attributes()]
        
        attrs.append('type="date"')
        
        if self.min_date:
            attrs.append(f'min="{self.min_date.isoformat()}"')
            
        if self.max_date:
            attrs.append(f'max="{self.max_date.isoformat()}"')
        
        html_parts = [
            f'<div class="form-group">',
            f'  <label for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
            f'  <input {" ".join(attrs)}>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return date
    
    def get_field_constraints(self) -> Dict[str, Any]:
        """Get Pydantic field constraints for this field."""
        constraints = super().get_field_constraints()
        
        # Date constraints are handled via validators
        return constraints
    
    def get_validators(self) -> List[Callable]:
        """Get validators for this field."""
        validators = []
        
        if self.min_date or self.max_date:
            min_date = self.min_date
            max_date = self.max_date
            
            def validate_date_range(cls, v):
                if min_date and v < min_date:
                    raise ValueError(f"Date must be on or after {min_date.isoformat()}")
                if max_date and v > max_date:
                    raise ValueError(f"Date must be on or before {max_date.isoformat()}")
                return v
            
            validators.append(validate_date_range)
        
        return validators


class SelectField(FormField):
    """Select dropdown field."""
    choices: List[Dict[str, str]]  # List of {value: str, label: str}
    multiple: bool = False
    size: Optional[int] = None
    
    def get_html(self) -> str:
        """Generate HTML for a select field."""
        attrs = [self.get_common_attributes()]
        
        if self.multiple:
            attrs.append('multiple')
            
        if self.size:
            attrs.append(f'size="{self.size}"')
        
        options = []
        for choice in self.choices:
            value = html.escape(choice["value"])
            label = html.escape(choice["label"])
            options.append(f'    <option value="{value}">{label}</option>')
        
        html_parts = [
            f'<div class="form-group">',
            f'  <label for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
            f'  <select {" ".join(attrs)}>',
            "\n".join(options),
            '  </select>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return List[str] if self.multiple else str
    
    def get_field_default(self):
        """Get the default value for this field."""
        if self.multiple:
            return [] if not self.required else ...
        return "" if not self.required else ...
    
    def get_validators(self) -> List[Callable]:
        """Get validators for this field."""
        validators = []
        
        # Validate choices
        choices = self.choices
        multiple = self.multiple
        
        def validate_choices(cls, v):
            valid_values = [choice["value"] for choice in choices]
            
            if multiple:
                invalid = [x for x in v if x not in valid_values]
                if invalid:
                    raise ValueError(f"Invalid choices: {', '.join(invalid)}")
            elif v not in valid_values:
                raise ValueError(f"Invalid choice: {v}")
            
            return v
        
        validators.append(validate_choices)
        
        return validators


class TextAreaField(FormField):
    """Textarea for multiline text input."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    rows: int = 3
    cols: Optional[int] = None
    placeholder: Optional[str] = None
    
    def get_html(self) -> str:
        """Generate HTML for a textarea field."""
        attrs = [self.get_common_attributes()]
        
        attrs.append(f'rows="{self.rows}"')
        
        if self.cols:
            attrs.append(f'cols="{self.cols}"')
            
        if self.min_length is not None:
            attrs.append(f'minlength="{self.min_length}"')
            
        if self.max_length is not None:
            attrs.append(f'maxlength="{self.max_length}"')
            
        if self.placeholder:
            attrs.append(f'placeholder="{html.escape(self.placeholder)}"')
        
        html_parts = [
            f'<div class="form-group">',
            f'  <label for="{html.escape(self.name)}">{html.escape(self.label)}</label>',
            f'  <textarea {" ".join(attrs)}></textarea>',
        ]
        
        if self.help_text:
            html_parts.append(f'  <small class="form-text text-muted">{html.escape(self.help_text)}</small>')
            
        html_parts.append('</div>')
        
        return "\n".join(html_parts)
    
    def get_field_type(self) -> Type:
        """Get the Python type for this field."""
        return str
    
    def get_field_constraints(self) -> Dict[str, Any]:
        """Get Pydantic field constraints for this field."""
        constraints = super().get_field_constraints()
        
        if self.min_length is not None:
            constraints["min_length"] = self.min_length
            
        if self.max_length is not None:
            constraints["max_length"] = self.max_length
            
        return constraints


class FormDefinition(BaseModel):
    """Form definition containing multiple fields."""
    title: str
    description: Optional[str] = None
    fields: List[Union[
        StringField, NumberField, BooleanField, 
        DateField, SelectField, TextAreaField
    ]]
    submit_label: str = "Submit"
    cancel_url: Optional[str] = None
    method: str = "post"
    action: str = ""
    css_classes: List[str] = Field(default_factory=lambda: ["form"])
    
    def create_model(self) -> Type[BaseModel]:
        """Generate a Pydantic model from the form definition."""
        fields = {}
        validators = {}
        
        for field in self.fields:
            # Get field type
            field_type = field.get_field_type()
            
            # Make field optional if not required
            if not field.required and field_type != bool:
                field_type = Optional[field_type]
            
            # Get field constraints
            constraints = field.get_field_constraints()
            
            # Get default value
            default = field.get_field_default()
            
            # Create field definition
            if constraints:
                field_def = (field_type, Field(default, **constraints))
            else:
                field_def = (field_type, default)
            
            fields[field.name] = field_def
            
            # Add validators
            field_validators = field.get_validators() if hasattr(field, "get_validators") else []
            
            for i, validator_func in enumerate(field_validators):
                validator_name = f"validate_{field.name}_{i}"
                validators[validator_name] = field_validator(field.name)(validator_func)
        
        # Create model with fields
        model = create_model(
            self.title.replace(" ", "") + "Form",
            __doc__=self.description,
            **fields
        )
        
        # Add validators
        for name, validator in validators.items():
            setattr(model, name, validator)
        
        return model
    
    def generate_html(self) -> str:
        """Generate HTML for the complete form."""
        form_classes = " ".join(html.escape(cls) for cls in self.css_classes)
        
        html_parts = [
            f'<form method="{html.escape(self.method)}" action="{html.escape(self.action)}" class="{form_classes}">',
        ]
        
        if self.title:
            html_parts.append(f'  <h2>{html.escape(self.title)}</h2>')
            
        if self.description:
            html_parts.append(f'  <p class="form-description">{html.escape(self.description)}</p>')
        
        # Add fields
        for field in self.fields:
            # Indent the field HTML
            field_html = field.get_html()
            indented_field_html = "  " + field_html.replace("\n", "\n  ")
            html_parts.append(indented_field_html)
        
        # Add buttons
        button_group = ['  <div class="form-group button-group">']
        
        # Submit button
        button_group.append(f'    <button type="submit" class="btn btn-primary">{html.escape(self.submit_label)}</button>')
        
        # Cancel button if URL provided
        if self.cancel_url:
            button_group.append(f'    <a href="{html.escape(self.cancel_url)}" class="btn btn-secondary">Cancel</a>')
            
        button_group.append('  </div>')
        html_parts.append("\n".join(button_group))
        
        html_parts.append('</form>')
        
        return "\n".join(html_parts)


def create_contact_form() -> FormDefinition:
    """Create an example contact form definition."""
    return FormDefinition(
        title="Contact Us",
        description="Please fill out this form to get in touch with our team.",
        fields=[
            StringField(
                name="name",
                label="Full Name",
                required=True,
                min_length=2,
                max_length=100,
                help_text="Your full name",
                css_classes=["form-control"]
            ),
            StringField(
                name="email",
                label="Email Address",
                required=True,
                input_type="email",
                pattern=r"[^@]+@[^@]+\.[^@]+",
                help_text="Your email address",
                css_classes=["form-control"]
            ),
            SelectField(
                name="subject",
                label="Subject",
                required=True,
                choices=[
                    {"value": "general", "label": "General Inquiry"},
                    {"value": "support", "label": "Technical Support"},
                    {"value": "billing", "label": "Billing Question"},
                    {"value": "feedback", "label": "Feedback"}
                ],
                help_text="What is your message about?",
                css_classes=["form-control"]
            ),
            TextAreaField(
                name="message",
                label="Message",
                required=True,
                min_length=10,
                rows=5,
                help_text="Your message",
                css_classes=["form-control"]
            ),
            BooleanField(
                name="subscribe",
                label="Subscribe to newsletter",
                required=False,
                default=False,
                help_text="Receive updates and news from our team"
            )
        ],
        submit_label="Send Message",
        cancel_url="/",
        css_classes=["form", "contact-form"]
    )


def create_registration_form() -> FormDefinition:
    """Create an example user registration form definition."""
    return FormDefinition(
        title="Create an Account",
        description="Sign up for a new account to access our services.",
        fields=[
            StringField(
                name="username",
                label="Username",
                required=True,
                min_length=3,
                max_length=20,
                pattern=r"^[a-zA-Z0-9_]+$",
                help_text="Letters, numbers, and underscores only",
                css_classes=["form-control"]
            ),
            StringField(
                name="email",
                label="Email Address",
                required=True,
                input_type="email",
                help_text="We'll never share your email with anyone else",
                css_classes=["form-control"]
            ),
            StringField(
                name="password",
                label="Password",
                required=True,
                input_type="password",
                min_length=8,
                help_text="At least 8 characters",
                css_classes=["form-control"]
            ),
            StringField(
                name="confirm_password",
                label="Confirm Password",
                required=True,
                input_type="password",
                help_text="Enter the same password again",
                css_classes=["form-control"]
            ),
            DateField(
                name="birth_date",
                label="Date of Birth",
                required=True,
                min_date=date(1900, 1, 1),
                max_date=date.today(),
                help_text="You must be at least 13 years old",
                css_classes=["form-control"]
            ),
            SelectField(
                name="country",
                label="Country",
                required=True,
                choices=[
                    {"value": "us", "label": "United States"},
                    {"value": "ca", "label": "Canada"},
                    {"value": "uk", "label": "United Kingdom"},
                    {"value": "au", "label": "Australia"},
                    {"value": "other", "label": "Other"}
                ],
                css_classes=["form-control"]
            ),
            BooleanField(
                name="terms",
                label="I agree to the Terms of Service and Privacy Policy",
                required=True,
                default=False
            )
        ],
        submit_label="Register",
        css_classes=["form", "registration-form"]
    )


def main():
    """Demonstrate the form builder system."""
    print("=" * 80)
    print("FORM BUILDER DEMONSTRATION")
    print("=" * 80)
    
    # Create a contact form
    contact_form = create_contact_form()
    
    # Generate Pydantic model
    ContactForm = contact_form.create_model()
    
    print("\n1. Contact Form Definition")
    print(f"Title: {contact_form.title}")
    print(f"Description: {contact_form.description}")
    print("Fields:")
    for i, field in enumerate(contact_form.fields):
        print(f"  {i+1}. {field.name} ({field.__class__.__name__}): {field.label}")
        if hasattr(field, "min_length") and field.min_length:
            print(f"     Min Length: {field.min_length}")
        if hasattr(field, "pattern") and field.pattern:
            print(f"     Pattern: {field.pattern}")
    
    # Generate HTML
    contact_html = contact_form.generate_html()
    
    print("\n2. Generated HTML Form")
    print("-" * 40)
    print(contact_html)
    print("-" * 40)
    
    # Test the generated model
    print("\n3. Using the Generated Pydantic Model")
    
    # Valid data
    try:
        form_data = ContactForm(
            name="John Doe",
            email="john@example.com",
            subject="general",
            message="This is a test message that is long enough to pass validation.",
            subscribe=True
        )
        print("Valid form data:")
        print(form_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Invalid data
    print("\nTrying invalid data:")
    try:
        invalid_form = ContactForm(
            name="J",  # Too short
            email="invalid-email",  # Invalid format
            subject="invalid",  # Not in choices
            message="Short",  # Too short
            subscribe=True
        )
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Create a registration form
    registration_form = create_registration_form()
    
    # Generate Pydantic model
    RegistrationForm = registration_form.create_model()
    
    print("\n4. Registration Form")
    print(f"Title: {registration_form.title}")
    print(f"Fields: {len(registration_form.fields)}")
    
    # Generate HTML
    registration_html = registration_form.generate_html()
    
    print("\n5. Generated Registration HTML Form (excerpt)")
    print("-" * 40)
    # Print just the first few lines to avoid too much output
    print("\n".join(registration_html.split("\n")[:20]) + "\n...")
    print("-" * 40)
    
    # Add custom validator to the registration form model
    @field_validator('confirm_password')
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v
    
    # Add the validator to the model
    setattr(RegistrationForm, 'validate_passwords_match', passwords_match)
    
    # Test the registration model
    print("\n6. Testing Registration Form Validation")
    
    # Valid data
    try:
        reg_data = RegistrationForm(
            username="johndoe",
            email="john@example.com",
            password="securepass",
            confirm_password="securepass",
            birth_date=date(1990, 1, 15),
            country="us",
            terms=True
        )
        print("Valid registration data:")
        print(reg_data.model_dump_json(indent=2))
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Invalid data - passwords don't match
    print("\nTrying invalid registration data (passwords don't match):")
    try:
        invalid_reg = RegistrationForm(
            username="johndoe",
            email="john@example.com",
            password="securepass",
            confirm_password="different",  # Doesn't match
            birth_date=date(1990, 1, 15),
            country="us",
            terms=True
        )
    except Exception as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    main()
