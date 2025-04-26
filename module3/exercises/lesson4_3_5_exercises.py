"""
Lesson 4.3.5: Form Builder

This exercise implements a form builder that generates both Pydantic models
for validation and HTML form elements from a single definition.
"""

from pydantic import BaseModel, Field, create_model, field_validator
from typing import Dict, Any, Type, List, Optional, Union, Literal
from enum import Enum
import re
import html


class FieldType(str, Enum):
    """Types of form fields."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    EMAIL = "email"
    PASSWORD = "password"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXTAREA = "textarea"
    FILE = "file"
    HIDDEN = "hidden"


class BaseField(BaseModel):
    """Base class for form fields."""
    name: str
    label: str
    required: bool = False
    default: Any = None
    help_text: Optional[str] = None
    placeholder: Optional[str] = None
    disabled: bool = False
    readonly: bool = False
    css_classes: List[str] = []
    attributes: Dict[str, str] = {}


class StringField(BaseField):
    """String input field."""
    type: Literal[FieldType.STRING, FieldType.EMAIL, FieldType.PASSWORD, FieldType.TEXTAREA] = FieldType.STRING
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    input_type: str = "text"  # HTML input type (text, email, password, etc.)


class NumberField(BaseField):
    """Numeric input field."""
    type: Literal[FieldType.NUMBER, FieldType.INTEGER] = FieldType.NUMBER
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Union[int, float] = 1


class BooleanField(BaseField):
    """Boolean input field (checkbox)."""
    type: Literal[FieldType.BOOLEAN] = FieldType.BOOLEAN


class DateTimeField(BaseField):
    """Date/time input field."""
    type: Literal[FieldType.DATE, FieldType.TIME, FieldType.DATETIME] = FieldType.DATE
    min_value: Optional[str] = None
    max_value: Optional[str] = None


class SelectOption(BaseModel):
    """Option for select fields."""
    value: str
    label: str
    disabled: bool = False


class SelectField(BaseField):
    """Select dropdown field."""
    type: Literal[FieldType.SELECT, FieldType.MULTISELECT] = FieldType.SELECT
    choices: List[SelectOption] = []
    multiple: bool = False


class FileField(BaseField):
    """File upload field."""
    type: Literal[FieldType.FILE] = FieldType.FILE
    accept: Optional[str] = None  # File types to accept
    multiple: bool = False


class HiddenField(BaseField):
    """Hidden input field."""
    type: Literal[FieldType.HIDDEN] = FieldType.HIDDEN


# Union type for all field types
FormFieldType = Union[
    StringField,
    NumberField,
    BooleanField,
    DateTimeField,
    SelectField,
    FileField,
    HiddenField
]


class FormDefinition(BaseModel):
    """Definition of a form with fields and metadata."""
    title: str
    description: Optional[str] = None
    fields: List[FormFieldType] = []
    submit_label: str = "Submit"
    cancel_label: Optional[str] = "Cancel"
    method: str = "post"
    action: str = ""
    enctype: Optional[str] = None
    css_classes: List[str] = []
    
    def create_model(self) -> Type[BaseModel]:
        """
        Create a Pydantic model from the form definition.
        
        Returns:
            A Pydantic model class with fields based on the form definition.
        """
        fields = {}
        validators = {}
        
        for field in self.fields:
            # Skip file fields as they can't be validated by Pydantic
            if field.type == FieldType.FILE:
                continue
            
            # Determine the Python type based on field type
            if field.type in [FieldType.STRING, FieldType.EMAIL, FieldType.PASSWORD, FieldType.TEXTAREA]:
                python_type = str
            elif field.type == FieldType.INTEGER:
                python_type = int
            elif field.type == FieldType.NUMBER:
                python_type = float
            elif field.type == FieldType.BOOLEAN:
                python_type = bool
            elif field.type in [FieldType.DATE, FieldType.TIME, FieldType.DATETIME]:
                python_type = str  # Use string for date/time fields
            elif field.type in [FieldType.SELECT, FieldType.MULTISELECT]:
                if field.multiple:
                    python_type = List[str]
                else:
                    python_type = str
            elif field.type == FieldType.HIDDEN:
                python_type = str
            else:
                python_type = Any
            
            # Make the type optional if not required
            if not field.required:
                python_type = Optional[python_type]
            
            # Create field constraints
            field_constraints = {}
            
            # Add description from help text
            if field.help_text:
                field_constraints["description"] = field.help_text
            
            # Add default value if provided
            if field.default is not None:
                field_constraints["default"] = field.default
            elif not field.required:
                field_constraints["default"] = None
            
            # Add type-specific constraints
            if isinstance(field, StringField):
                if field.min_length is not None:
                    field_constraints["min_length"] = field.min_length
                if field.max_length is not None:
                    field_constraints["max_length"] = field.max_length
                if field.pattern is not None:
                    # Add a validator for the pattern
                    validator_name = f"validate_{field.name}_pattern"
                    pattern = field.pattern
                    
                    def create_pattern_validator(pattern):
                        def validate_pattern(cls, v):
                            if v is not None and not re.match(pattern, v):
                                raise ValueError(f"Value does not match pattern: {pattern}")
                            return v
                        return validate_pattern
                    
                    validators[validator_name] = field_validator(field.name)(create_pattern_validator(pattern))
                
                # Add email validator
                if field.type == FieldType.EMAIL:
                    validator_name = f"validate_{field.name}_email"
                    
                    def validate_email(cls, v):
                        if v is not None and not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", v):
                            raise ValueError("Invalid email format")
                        return v
                    
                    validators[validator_name] = field_validator(field.name)(validate_email)
            
            elif isinstance(field, NumberField):
                if field.min_value is not None:
                    field_constraints["ge"] = field.min_value
                if field.max_value is not None:
                    field_constraints["le"] = field.max_value
            
            elif isinstance(field, SelectField):
                # Add validator to check if value is in choices
                validator_name = f"validate_{field.name}_choices"
                choices = [option.value for option in field.choices]
                
                if field.multiple:
                    def validate_choices(cls, v):
                        if v is not None:
                            for item in v:
                                if item not in choices:
                                    raise ValueError(f"Invalid choice: {item}. Must be one of: {choices}")
                        return v
                else:
                    def validate_choices(cls, v):
                        if v is not None and v not in choices:
                            raise ValueError(f"Invalid choice: {v}. Must be one of: {choices}")
                        return v
                
                validators[validator_name] = field_validator(field.name)(validate_choices)
            
            # Add the field to the model
            if field_constraints:
                fields[field.name] = (python_type, Field(**field_constraints))
            else:
                fields[field.name] = (python_type, ...)
        
        # Create the model class
        model_name = "".join(word.capitalize() for word in self.title.split())
        model = create_model(
            model_name,
            __validators__=validators,
            **fields
        )
        
        return model
    
    def generate_html(self) -> str:
        """
        Generate HTML form from the form definition.
        
        Returns:
            HTML string representing the form.
        """
        # Start the form tag
        form_classes = " ".join(self.css_classes)
        form_html = f'<form method="{self.method}" action="{self.action}"'
        
        if self.enctype:
            form_html += f' enctype="{self.enctype}"'
        
        if form_classes:
            form_html += f' class="{form_classes}"'
        
        form_html += '>\n'
        
        # Add form title and description
        if self.title:
            form_html += f'  <h2>{html.escape(self.title)}</h2>\n'
        
        if self.description:
            form_html += f'  <p class="form-description">{html.escape(self.description)}</p>\n'
        
        # Add form fields
        for field in self.fields:
            form_html += self._generate_field_html(field)
        
        # Add submit and cancel buttons
        form_html += '  <div class="form-actions">\n'
        form_html += f'    <button type="submit" class="btn btn-primary">{html.escape(self.submit_label)}</button>\n'
        
        if self.cancel_label:
            form_html += f'    <button type="button" class="btn btn-secondary">{html.escape(self.cancel_label)}</button>\n'
        
        form_html += '  </div>\n'
        form_html += '</form>'
        
        return form_html
    
    def _generate_field_html(self, field: FormFieldType) -> str:
        """Generate HTML for a single form field."""
        field_id = f"id_{field.name}"
        field_classes = " ".join(field.css_classes)
        required_attr = ' required' if field.required else ''
        disabled_attr = ' disabled' if field.disabled else ''
        readonly_attr = ' readonly' if field.readonly else ''
        placeholder_attr = f' placeholder="{html.escape(field.placeholder)}"' if field.placeholder else ''
        
        # Add custom attributes
        custom_attrs = ""
        for attr_name, attr_value in field.attributes.items():
            custom_attrs += f' {attr_name}="{html.escape(attr_value)}"'
        
        # Start field container
        field_html = '  <div class="form-group">\n'
        
        # Add label (except for hidden fields)
        if field.type != FieldType.HIDDEN:
            required_marker = ' <span class="required">*</span>' if field.required else ''
            field_html += f'    <label for="{field_id}">{html.escape(field.label)}{required_marker}</label>\n'
        
        # Generate field-specific HTML
        if isinstance(field, StringField):
            if field.type == FieldType.TEXTAREA:
                field_html += f'    <textarea id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{placeholder_attr}{custom_attrs}>'
                if field.default:
                    field_html += html.escape(str(field.default))
                field_html += '</textarea>\n'
            else:
                input_type = field.input_type
                if field.type == FieldType.EMAIL:
                    input_type = "email"
                elif field.type == FieldType.PASSWORD:
                    input_type = "password"
                
                min_length_attr = f' minlength="{field.min_length}"' if field.min_length is not None else ''
                max_length_attr = f' maxlength="{field.max_length}"' if field.max_length is not None else ''
                pattern_attr = f' pattern="{field.pattern}"' if field.pattern else ''
                
                field_html += f'    <input type="{input_type}" id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{placeholder_attr}{min_length_attr}{max_length_attr}{pattern_attr}{custom_attrs}'
                
                if field.default:
                    field_html += f' value="{html.escape(str(field.default))}"'
                
                field_html += '>\n'
        
        elif isinstance(field, NumberField):
            input_type = "number"
            if field.type == FieldType.INTEGER:
                step_attr = ' step="1"'
            else:
                step_attr = f' step="{field.step}"' if field.step is not None else ''
            
            min_attr = f' min="{field.min_value}"' if field.min_value is not None else ''
            max_attr = f' max="{field.max_value}"' if field.max_value is not None else ''
            
            field_html += f'    <input type="{input_type}" id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{placeholder_attr}{min_attr}{max_attr}{step_attr}{custom_attrs}'
            
            if field.default is not None:
                field_html += f' value="{field.default}"'
            
            field_html += '>\n'
        
        elif isinstance(field, BooleanField):
            checked_attr = ' checked' if field.default else ''
            
            field_html += f'    <div class="form-check">\n'
            field_html += f'      <input type="checkbox" id="{field_id}" name="{field.name}" class="form-check-input{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{checked_attr}{custom_attrs}>\n'
            field_html += f'      <label class="form-check-label" for="{field_id}">{html.escape(field.label)}</label>\n'
            field_html += f'    </div>\n'
            
            # Remove the label we added earlier
            field_html = field_html.replace(f'    <label for="{field_id}">{html.escape(field.label)}{required_marker}</label>\n', '')
        
        elif isinstance(field, DateTimeField):
            input_type = field.type.value  # date, time, or datetime-local
            if input_type == "datetime":
                input_type = "datetime-local"
            
            min_attr = f' min="{field.min_value}"' if field.min_value is not None else ''
            max_attr = f' max="{field.max_value}"' if field.max_value is not None else ''
            
            field_html += f'    <input type="{input_type}" id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{min_attr}{max_attr}{custom_attrs}'
            
            if field.default:
                field_html += f' value="{html.escape(str(field.default))}"'
            
            field_html += '>\n'
        
        elif isinstance(field, SelectField):
            multiple_attr = ' multiple' if field.multiple else ''
            
            field_html += f'    <select id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{readonly_attr}{multiple_attr}{custom_attrs}>\n'
            
            # Add options
            for option in field.choices:
                option_disabled_attr = ' disabled' if option.disabled else ''
                selected_attr = ' selected' if field.default == option.value else ''
                
                field_html += f'      <option value="{html.escape(option.value)}"{option_disabled_attr}{selected_attr}>{html.escape(option.label)}</option>\n'
            
            field_html += '    </select>\n'
        
        elif isinstance(field, FileField):
            multiple_attr = ' multiple' if field.multiple else ''
            accept_attr = f' accept="{field.accept}"' if field.accept else ''
            
            field_html += f'    <input type="file" id="{field_id}" name="{field.name}" class="form-control{" " + field_classes if field_classes else ""}"{required_attr}{disabled_attr}{multiple_attr}{accept_attr}{custom_attrs}>\n'
        
        elif isinstance(field, HiddenField):
            field_html += f'    <input type="hidden" id="{field_id}" name="{field.name}"{custom_attrs}'
            
            if field.default:
                field_html += f' value="{html.escape(str(field.default))}"'
            
            field_html += '>\n'
        
        # Add help text
        if field.help_text and field.type != FieldType.HIDDEN:
            field_html += f'    <small class="form-text text-muted">{html.escape(field.help_text)}</small>\n'
        
        # Close field container
        field_html += '  </div>\n'
        
        return field_html


def create_contact_form_definition() -> FormDefinition:
    """Create a sample contact form definition."""
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
                help_text="Enter your full name",
                placeholder="John Doe"
            ),
            StringField(
                name="email",
                label="Email Address",
                type=FieldType.EMAIL,
                required=True,
                help_text="We'll never share your email with anyone else",
                placeholder="john@example.com"
            ),
            SelectField(
                name="subject",
                label="Subject",
                required=True,
                choices=[
                    SelectOption(value="general", label="General Inquiry"),
                    SelectOption(value="support", label="Technical Support"),
                    SelectOption(value="billing", label="Billing Question"),
                    SelectOption(value="feedback", label="Feedback")
                ],
                help_text="Select the subject of your message"
            ),
            StringField(
                name="message",
                label="Message",
                type=FieldType.TEXTAREA,
                required=True,
                min_length=10,
                help_text="Please provide as much detail as possible",
                placeholder="Your message here..."
            ),
            BooleanField(
                name="subscribe",
                label="Subscribe to newsletter",
                help_text="Receive updates and news from our team"
            ),
            HiddenField(
                name="source",
                label="Source",
                default="website"
            )
        ],
        submit_label="Send Message",
        cancel_label="Cancel",
        css_classes=["contact-form", "needs-validation"]
    )


def main():
    """Demonstrate the form builder."""
    print("Form Builder Demonstration")
    print("=" * 40)
    
    # Create a contact form definition
    form_def = create_contact_form_definition()
    
    # Generate Pydantic model
    ContactForm = form_def.create_model()
    
    print("\nGenerated Pydantic Model:")
    print(f"Model name: {ContactForm.__name__}")
    print("Fields:")
    for field_name, field_info in ContactForm.model_fields.items():
        print(f"  - {field_name}: {field_info.annotation}")
        if field_info.description:
            print(f"    Description: {field_info.description}")
        if field_info.default is not None and field_info.default is not ...:
            print(f"    Default: {field_info.default}")
    
    # Create a valid form submission
    form_data = ContactForm(
        name="John Doe",
        email="john@example.com",
        subject="general",
        message="This is a test message for the contact form.",
        subscribe=True
    )
    
    print("\nValid Form Data:")
    print(form_data.model_dump_json(indent=2))
    
    # Try an invalid submission
    print("\nInvalid Form Data (will raise ValidationError):")
    try:
        invalid_form = ContactForm(
            name="J",  # Too short
            email="invalid-email",  # Invalid email format
            subject="unknown",  # Not in choices
            message="Short"  # Too short
        )
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Generate HTML form
    html_form = form_def.generate_html()
    
    print("\nGenerated HTML Form:")
    print(html_form)


if __name__ == "__main__":
    main()
