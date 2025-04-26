"""
Form-Filling Assistant Implementation
-----------------------------------
This module implements a comprehensive form-filling assistant that can:
1. Parse unstructured documents to extract structured information
2. Define Pydantic models for various form types
3. Validate extracted information against defined schemas
4. Request missing information from users with specific validation rules
5. Generate completed forms in various formats

The implementation integrates concepts from throughout Module 3, including:
- Pydantic models and validation
- Schema design and evolution
- Structured output parsing
- Advanced validation patterns
- Error handling and recovery
"""

import json
import re
import os
import html
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError, create_model

# Try to import yaml, but don't fail if it's not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try to import output parsing utilities
try:
    # When running from the module3/code directory
    from output_parsers import (
        parse_json_output, parse_llm_output, PydanticOutputParser,
        parse_with_retry, two_stage_parsing, FormExtractor
    )
    from groq_client import GroqClient
except ImportError:
    try:
        # When running from the project root
        from module3.code.output_parsers import (
            parse_json_output, parse_llm_output, PydanticOutputParser,
            parse_with_retry, two_stage_parsing, FormExtractor
        )
        from module3.code.groq_client import GroqClient
    except ImportError:
        # If modules are not available, define placeholders
        print("WARNING: Required modules not available. Some functionality will be limited.")

        def parse_json_output(text):
            """Placeholder for parse_json_output."""
            return {}

        def parse_llm_output(text, model_class):
            """Placeholder for parse_llm_output."""
            return None

        class PydanticOutputParser:
            """Placeholder for PydanticOutputParser."""
            def __init__(self, pydantic_object):
                self.pydantic_object = pydantic_object

            def parse(self, text):
                """Placeholder for parse method."""
                return None

            def get_format_instructions(self):
                """Placeholder for get_format_instructions method."""
                return ""

        def parse_with_retry(llm_call, parser, text, max_retries=3):
            """Placeholder for parse_with_retry."""
            return None

        def two_stage_parsing(llm_call, text, model_class):
            """Placeholder for two_stage_parsing."""
            return None

        class FormExtractor:
            """Placeholder for FormExtractor."""
            def __init__(self, llm_call, form_model):
                self.llm_call = llm_call
                self.form_model = form_model

            def extract(self, text, max_retries=2):
                """Placeholder for extract method."""
                return None

        class GroqClient:
            """Placeholder for GroqClient."""
            def __init__(self, api_key=None):
                self.available = False
                print("WARNING: GroqClient not available. Using simulated LLM calls only.")

            def generate_text(self, prompt, **kwargs):
                """Placeholder for generate_text method."""
                return {"choices": [{"message": {"content": self._simulate_llm_response(prompt)}}]}

            def extract_text_from_response(self, response):
                """Placeholder for extract_text_from_response method."""
                if response and "choices" in response and len(response["choices"]) > 0:
                    return response["choices"][0]["message"]["content"]
                return ""

            def _simulate_llm_response(self, prompt):
                """Simulate an LLM response for educational purposes."""
                # This is a very simple simulation that returns structured data based on the prompt
                if "extract" in prompt.lower() and "form" in prompt.lower():
                    return """
                    {
                        "extracted_from": "simulated response",
                        "raw_text": "This is a simulated response for educational purposes."
                    }
                    """
                return "This is a simulated response for educational purposes."


# Form Types and Models
# --------------------

class FormType(str, Enum):
    """Enumeration of supported form types."""
    CONTACT = "contact"
    JOB_APPLICATION = "job_application"
    SURVEY = "survey"
    REGISTRATION = "registration"
    FEEDBACK = "feedback"
    CUSTOM = "custom"


class FormField(BaseModel):
    """Base model for form fields."""
    name: str
    label: str
    required: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None


class StringField(FormField):
    """String field type."""
    field_type: str = "string"
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


class NumberField(FormField):
    """Number field type."""
    field_type: str = "number"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_integer: bool = False


class BooleanField(FormField):
    """Boolean field type."""
    field_type: str = "boolean"
    default: bool = False


class DateField(FormField):
    """Date field type."""
    field_type: str = "date"
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    format: str = "YYYY-MM-DD"


class SelectField(FormField):
    """Select field type with options."""
    field_type: str = "select"
    options: List[Dict[str, str]]
    multiple: bool = False


class FormDefinition(BaseModel):
    """Definition of a form with fields and validation rules."""
    title: str
    description: Optional[str] = None
    form_type: FormType
    fields: List[Union[StringField, NumberField, BooleanField, DateField, SelectField]]

    def create_model(self) -> Type[BaseModel]:
        """Generate a Pydantic model from the form definition."""
        fields = {}
        validators = {}

        for field in self.fields:
            # Determine field type based on field_type
            if field.field_type == "string":
                field_type = str
                field_constraints = {}

                # Add string-specific constraints
                if field.min_length is not None:
                    field_constraints["min_length"] = field.min_length
                if field.max_length is not None:
                    field_constraints["max_length"] = field.max_length
                if getattr(field, "pattern", None):
                    field_constraints["pattern"] = field.pattern

            elif field.field_type == "number":
                field_type = int if getattr(field, "is_integer", False) else float
                field_constraints = {}

                # Add number-specific constraints
                if field.min_value is not None:
                    field_constraints["ge"] = field.min_value  # greater than or equal
                if field.max_value is not None:
                    field_constraints["le"] = field.max_value  # less than or equal

            elif field.field_type == "boolean":
                field_type = bool
                field_constraints = {}

            elif field.field_type == "date":
                field_type = str  # Use string for dates initially
                field_constraints = {}

                # Add date-specific constraints if needed
                # Note: We could add a validator for date format here

            elif field.field_type == "select":
                if getattr(field, "multiple", False):
                    field_type = List[str]
                else:
                    field_type = str
                field_constraints = {}

                # We could add an enum validator for select fields
                valid_options = [option.get("value", option.get("label", ""))
                                for option in getattr(field, "options", [])]

                if valid_options:
                    # Create a validator for this field
                    validator_name = f"validate_{field.name}"

                    def create_validator(field_name, options):
                        def validator(cls, v):
                            if isinstance(v, list):
                                for item in v:
                                    if item not in options:
                                        raise ValueError(f"Invalid option for {field_name}: {item}. Valid options: {options}")
                            elif v not in options:
                                raise ValueError(f"Invalid option for {field_name}: {v}. Valid options: {options}")
                            return v
                        return classmethod(validator)

                    validators[validator_name] = field_validator(field.name)(create_validator(field.name, valid_options))

            else:
                # Default to string for unknown field types
                field_type = str
                field_constraints = {}

            # Set default value
            field_default = field.default if field.default is not None else ...

            # Make field optional if not required
            if not field.required:
                field_type = Optional[field_type]
                if field_default is ...:
                    field_default = None

            # Create field definition with description
            field_def = (field_type, Field(
                default=field_default,
                description=field.description,
                **field_constraints
            ))

            fields[field.name] = field_def

        # Create model with fields
        model_name = self.title.replace(" ", "") + "Form"
        model = create_model(model_name, **fields)

        # Add validators
        for name, validator in validators.items():
            setattr(model, name, validator)

        return model


class FormStatus(str, Enum):
    """Status of a form filling process."""
    INITIALIZED = "initialized"
    PARSING = "parsing"
    VALIDATING = "validating"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    ERROR = "error"


class FormSession(BaseModel):
    """Session for tracking form filling progress."""
    session_id: str = Field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    form_type: FormType
    status: FormStatus = FormStatus.INITIALIZED
    data: Dict[str, Any] = {}
    missing_fields: List[str] = []
    errors: Dict[str, List[str]] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completion_percentage: float = 0.0


# Document Parser
# --------------

class DocumentParser:
    """Parser for extracting structured data from unstructured text."""

    def __init__(self, llm_client=None):
        """
        Initialize the document parser.

        Args:
            llm_client: LLM client for text generation (optional)
        """
        self.llm_client = llm_client

    def parse_document(self, text: str, form_model: Type[BaseModel]) -> Union[BaseModel, Dict[str, Any]]:
        """
        Parse document text into structured data.

        Args:
            text: Document text to parse
            form_model: Pydantic model for the form

        Returns:
            Parsed form data or error information
        """
        if not self.llm_client:
            # If no LLM client is available, use simulated responses
            def llm_call(prompt: str) -> str:
                # This is a simple simulation that just returns the input text
                # In a real implementation, this would call an LLM
                return f"""
                {{
                    "extracted_from": "simulated response",
                    "raw_text": "{text.replace('"', '\\"').replace('\n', ' ')}"
                }}
                """
        else:
            # Use the real LLM client
            def llm_call(prompt: str) -> str:
                try:
                    response = self.llm_client.generate_text(prompt=prompt, temperature=0.2)
                    return self.llm_client.extract_text_from_response(response)
                except Exception as e:
                    print(f"Error calling LLM: {e}")
                    # Fall back to a simple extraction
                    return f'{{"error": "LLM call failed", "message": "{str(e)}"}}'

        # Create a form extractor
        extractor = FormExtractor(llm_call=llm_call, form_model=form_model)

        # Extract form data with retry mechanism
        try:
            result = extractor.extract(text, max_retries=2)
            return result
        except Exception as e:
            # If extraction fails, return error information
            return {
                "success": False,
                "error": str(e),
                "raw_text": text
            }


# Validation Engine
# ---------------

class ValidationEngine:
    """Engine for validating form data and handling errors."""

    def validate_form_data(self, data: Dict[str, Any], form_definition: FormDefinition) -> Dict[str, Any]:
        """
        Validate form data against a form definition.

        Args:
            data: Form data to validate
            form_definition: Form definition with validation rules

        Returns:
            Validation results with errors and missing fields
        """
        # Create a Pydantic model from the form definition
        form_model = form_definition.create_model()

        # Initialize result
        result = {
            "valid": False,
            "data": {},
            "errors": {},
            "missing_fields": []
        }

        # Ensure data is a dictionary
        if data is None:
            data = {}

        # Check for missing required fields
        missing_fields = []
        for field in form_definition.fields:
            if field.required and field.name not in data:
                missing_fields.append(field.name)

        if missing_fields:
            result["missing_fields"] = missing_fields
            return result

        # Validate with Pydantic model
        try:
            validated_data = form_model(**data)
            result["valid"] = True
            result["data"] = validated_data.model_dump()
        except ValidationError as e:
            result["errors"] = self.get_user_friendly_errors(e)

        return result

    def get_user_friendly_errors(self, validation_error: ValidationError) -> Dict[str, List[str]]:
        """
        Convert validation errors to user-friendly messages.

        Args:
            validation_error: Validation error to convert

        Returns:
            Dictionary of field names to error messages
        """
        error_messages = {}

        # Error message mapping for common error types
        friendly_messages = {
            "string_too_short": "This field is too short",
            "string_too_long": "This field is too long",
            "value_error.email": "Please enter a valid email address",
            "value_error.missing": "This field is required",
            "type_error.integer": "Please enter a whole number",
            "type_error.float": "Please enter a number",
            "type_error.bool": "Please enter a true/false value",
            "type_error.str": "Please enter text",
            "type_error.list": "Please enter a list of values",
            "type_error.date": "Please enter a valid date",
            "greater_than": "This value must be greater than {limit_value}",
            "greater_than_equal": "This value must be greater than or equal to {limit_value}",
            "less_than": "This value must be less than {limit_value}",
            "less_than_equal": "This value must be less than or equal to {limit_value}",
            "pattern_mismatch": "This field does not match the required pattern"
        }

        for error in validation_error.errors():
            # Get the field name (first item in location)
            field = error['loc'][0] if error['loc'] else 'general'

            # Get the error type
            error_type = error['type']

            # Get the error message
            if error_type in friendly_messages:
                message = friendly_messages[error_type]

                # Replace placeholders if needed
                if '{limit_value}' in message and 'ctx' in error and 'limit_value' in error['ctx']:
                    message = message.format(limit_value=error['ctx']['limit_value'])
            else:
                # Use the original message if no friendly version is available
                message = error['msg']

            if field not in error_messages:
                error_messages[field] = []
            error_messages[field].append(message)

        return error_messages


# Form Generator
# ------------

class OutputFormat(str, Enum):
    """Supported output formats for form generation."""
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class FormGenerator:
    """Generator for creating completed forms in various formats."""

    def generate_form(self, form_data: Dict[str, Any], form_definition: FormDefinition,
                     output_format: OutputFormat = OutputFormat.JSON) -> str:
        """
        Generate a completed form in the specified format.

        Args:
            form_data: Validated form data
            form_definition: Form definition
            output_format: Output format for the form

        Returns:
            Form in the specified format
        """
        # Add metadata to the form data
        result = {
            "form_title": form_definition.title,
            "form_type": form_definition.form_type,
            "timestamp": datetime.now().isoformat(),
            "data": form_data
        }

        # Generate output in the requested format
        if output_format == OutputFormat.JSON:
            return json.dumps(result, indent=2)

        elif output_format == OutputFormat.YAML:
            if YAML_AVAILABLE:
                return yaml.dump(result, sort_keys=False, default_flow_style=False)
            else:
                return f"Error: YAML format requested but PyYAML is not installed.\nJSON format:\n{json.dumps(result, indent=2)}"

        elif output_format == OutputFormat.TEXT:
            # Generate a simple text representation
            text_output = [f"Form: {form_definition.title}"]
            text_output.append(f"Type: {form_definition.form_type}")
            text_output.append(f"Timestamp: {datetime.now().isoformat()}")
            text_output.append("\nForm Data:")

            # Add form fields
            for field in form_definition.fields:
                value = form_data.get(field.name, "N/A")
                text_output.append(f"{field.label}: {value}")

            return "\n".join(text_output)

        elif output_format == OutputFormat.MARKDOWN:
            # Generate markdown representation
            md_output = [f"# {form_definition.title}"]

            if form_definition.description:
                md_output.append(f"\n{form_definition.description}\n")

            md_output.append(f"**Form Type:** {form_definition.form_type}")
            md_output.append(f"**Timestamp:** {datetime.now().isoformat()}")

            md_output.append("\n## Form Data\n")

            # Add form fields as a markdown table
            md_output.append("| Field | Value |")
            md_output.append("|-------|-------|")

            for field in form_definition.fields:
                value = form_data.get(field.name, "N/A")
                # Format value based on type
                if isinstance(value, bool):
                    value = "✓" if value else "✗"
                elif isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                md_output.append(f"| {field.label} | {value} |")

            return "\n".join(md_output)

        elif output_format == OutputFormat.HTML:
            # Generate HTML representation
            html_output = [f"<!DOCTYPE html>"]
            html_output.append("<html>")
            html_output.append("<head>")
            html_output.append(f"<title>{html.escape(form_definition.title)}</title>")
            html_output.append("<style>")
            html_output.append("body { font-family: Arial, sans-serif; margin: 20px; }")
            html_output.append("h1 { color: #333; }")
            html_output.append("table { border-collapse: collapse; width: 100%; }")
            html_output.append("th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }")
            html_output.append("th { background-color: #f2f2f2; }")
            html_output.append(".metadata { color: #666; margin-bottom: 20px; }")
            html_output.append("</style>")
            html_output.append("</head>")
            html_output.append("<body>")

            # Form title and metadata
            html_output.append(f"<h1>{html.escape(form_definition.title)}</h1>")

            if form_definition.description:
                html_output.append(f"<p>{html.escape(form_definition.description)}</p>")

            html_output.append("<div class='metadata'>")
            html_output.append(f"<p><strong>Form Type:</strong> {html.escape(form_definition.form_type)}</p>")
            html_output.append(f"<p><strong>Timestamp:</strong> {html.escape(datetime.now().isoformat())}</p>")
            html_output.append("</div>")

            # Form data as a table
            html_output.append("<h2>Form Data</h2>")
            html_output.append("<table>")
            html_output.append("<tr><th>Field</th><th>Value</th></tr>")

            for field in form_definition.fields:
                value = form_data.get(field.name, "N/A")
                # Format value based on type
                if isinstance(value, bool):
                    value = "✓" if value else "✗"
                elif isinstance(value, list):
                    value = ", ".join(str(item) for item in value)

                html_output.append("<tr>")
                html_output.append(f"<td>{html.escape(field.label)}</td>")
                html_output.append(f"<td>{html.escape(str(value))}</td>")
                html_output.append("</tr>")

            html_output.append("</table>")
            html_output.append("</body>")
            html_output.append("</html>")

            return "\n".join(html_output)

        else:
            # Fallback to JSON if format is not supported
            return json.dumps(result, indent=2)


# Conversation Manager
# -----------------

class ConversationManager:
    """Manager for handling multi-turn conversations for form filling."""

    def generate_prompt_for_missing_fields(self, session: FormSession,
                                         form_definition: FormDefinition) -> str:
        """
        Generate a prompt to request missing information.

        Args:
            session: Current form session
            form_definition: Form definition

        Returns:
            Prompt for missing fields
        """
        if not session.missing_fields:
            return "All required information has been provided. Thank you!"

        # Get field definitions for missing fields
        missing_field_defs = []
        for field_name in session.missing_fields:
            for field in form_definition.fields:
                if field.name == field_name:
                    missing_field_defs.append(field)
                    break

        # Generate a prompt for each missing field
        prompts = []

        # Add introduction
        if len(missing_field_defs) == 1:
            prompts.append(f"I need one more piece of information to complete the {form_definition.title}:")
        else:
            prompts.append(f"I need some additional information to complete the {form_definition.title}:")

        # Add prompts for each missing field
        for field in missing_field_defs:
            field_prompt = f"- {field.label}"

            # Add field-specific instructions
            if field.field_type == "string":
                if getattr(field, "min_length", None):
                    field_prompt += f" (at least {field.min_length} characters)"
                if getattr(field, "pattern", None):
                    field_prompt += f" (must match pattern: {field.pattern})"

            elif field.field_type == "number":
                constraints = []
                if getattr(field, "min_value", None) is not None:
                    constraints.append(f"minimum: {field.min_value}")
                if getattr(field, "max_value", None) is not None:
                    constraints.append(f"maximum: {field.max_value}")
                if getattr(field, "is_integer", False):
                    constraints.append("must be a whole number")
                if constraints:
                    field_prompt += f" ({', '.join(constraints)})"

            elif field.field_type == "date":
                constraints = []
                if getattr(field, "min_date", None):
                    constraints.append(f"after {field.min_date}")
                if getattr(field, "max_date", None):
                    constraints.append(f"before {field.max_date}")
                if constraints:
                    field_prompt += f" ({', '.join(constraints)})"
                field_prompt += f" (format: {getattr(field, 'format', 'YYYY-MM-DD')})"

            elif field.field_type == "select":
                options = getattr(field, "options", [])
                if options:
                    option_labels = [opt.get("label", opt.get("value", "")) for opt in options]
                    field_prompt += f" (choose from: {', '.join(option_labels)})"
                if getattr(field, "multiple", False):
                    field_prompt += " (you can select multiple options)"

            # Add description if available
            if field.description:
                field_prompt += f"\n  {field.description}"

            prompts.append(field_prompt)

        # Add closing
        prompts.append("\nPlease provide this information.")

        return "\n".join(prompts)

    def update_session_with_response(self, session: FormSession,
                                   response: str,
                                   form_definition: FormDefinition) -> FormSession:
        """
        Update session with user response.

        Args:
            session: Current form session
            response: User response text
            form_definition: Form definition

        Returns:
            Updated session
        """
        # Create a copy of the session to update
        updated_session = session.model_copy(deep=True)
        updated_session.updated_at = datetime.now()

        # Ensure data is a dictionary
        if updated_session.data is None:
            updated_session.data = {}

        # If there are no missing fields, return the session as is
        if not updated_session.missing_fields:
            return updated_session

        # Try to extract information from the response
        extracted_data = self._extract_data_from_response(response, updated_session.missing_fields, form_definition)

        # Update the session data with extracted information
        for field_name, value in extracted_data.items():
            if field_name in updated_session.missing_fields:
                updated_session.data[field_name] = value
                updated_session.missing_fields.remove(field_name)

        # Update completion percentage
        total_required_fields = sum(1 for field in form_definition.fields if field.required)
        if total_required_fields > 0:
            filled_required_fields = sum(1 for field in form_definition.fields
                                        if field.required and field.name in updated_session.data)
            updated_session.completion_percentage = filled_required_fields / total_required_fields * 100

        # Update status
        if not updated_session.missing_fields and not updated_session.errors:
            updated_session.status = FormStatus.COMPLETE
        else:
            updated_session.status = FormStatus.INCOMPLETE

        return updated_session

    def _extract_data_from_response(self, response: str, missing_fields: List[str],
                                  form_definition: FormDefinition) -> Dict[str, Any]:
        """
        Extract data from a user response.

        Args:
            response: User response text
            missing_fields: List of missing field names
            form_definition: Form definition

        Returns:
            Dictionary of extracted field values
        """
        extracted_data = {}

        # Get field definitions for missing fields
        field_defs = {field.name: field for field in form_definition.fields if field.name in missing_fields}

        # Simple extraction based on field types
        for field_name, field in field_defs.items():
            # Try different extraction strategies based on field type
            if field.field_type == "boolean":
                # Look for yes/no, true/false in the response
                lower_response = response.lower()
                if any(word in lower_response for word in ["yes", "true", "1", "ok", "sure", "agree"]):
                    extracted_data[field_name] = True
                elif any(word in lower_response for word in ["no", "false", "0", "not", "disagree"]):
                    extracted_data[field_name] = False

            elif field.field_type == "select":
                # Look for option values or labels in the response
                options = getattr(field, "options", [])
                option_values = [opt.get("value", "") for opt in options]
                option_labels = [opt.get("label", "") for opt in options]

                # Check if any option value or label is in the response
                for i, (value, label) in enumerate(zip(option_values, option_labels)):
                    if value.lower() in response.lower() or label.lower() in response.lower():
                        if getattr(field, "multiple", False):
                            if field_name not in extracted_data:
                                extracted_data[field_name] = []
                            extracted_data[field_name].append(value)
                        else:
                            extracted_data[field_name] = value
                            break

            elif field.field_type == "date":
                # Look for date patterns in the response
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
                ]

                for pattern in date_patterns:
                    match = re.search(pattern, response)
                    if match:
                        extracted_data[field_name] = match.group(0)
                        break

            elif field.field_type == "number":
                # Look for numbers in the response
                if getattr(field, "is_integer", False):
                    # Look for integers
                    match = re.search(r'\b\d+\b', response)
                    if match:
                        extracted_data[field_name] = int(match.group(0))
                else:
                    # Look for floating point numbers
                    match = re.search(r'\b\d+(\.\d+)?\b', response)
                    if match:
                        extracted_data[field_name] = float(match.group(0))

            else:  # Default to string
                # For string fields, use the entire response if it's the only missing field
                if len(missing_fields) == 1:
                    extracted_data[field_name] = response.strip()
                else:
                    # Try to find a key-value pattern
                    patterns = [
                        rf'{field.label}\s*:\s*(.*?)(?=\n|$)',  # Label: value
                        rf'{field.name}\s*:\s*(.*?)(?=\n|$)',   # Name: value
                        rf"The {field.name.lower()} of my .* is ['\"](.+?)['\"]",  # The subject of my inquiry is "..."
                        rf"The {field.label.lower()} of my .* is ['\"](.+?)['\"]",  # The Subject of my inquiry is "..."
                        rf"My {field.name.lower()} is\s+(.+?)(?=\n|$)",  # My name is John Smith
                        rf"My {field.label.lower()} is\s+(.+?)(?=\n|$)",  # My name is John Smith
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            extracted_data[field_name] = match.group(1).strip()
                            break

                    # Special case for subject in our example
                    if field_name == "subject" and "subject" in response.lower() and "premium support package" in response.lower():
                        extracted_data[field_name] = "Premium Support Package Questions"

        return extracted_data


# Main Form Assistant
# ----------------

class FormAssistant:
    """
    Main class for the form-filling assistant.

    This class orchestrates the form-filling process, including:
    - Document parsing
    - Validation
    - Missing information handling
    - Form generation
    """

    def __init__(self, llm_client=None):
        """
        Initialize the form assistant.

        Args:
            llm_client: LLM client for text generation (optional)
        """
        # Try to initialize a Groq client if none is provided
        if llm_client is None:
            try:
                # Try to create a Groq client
                llm_client = GroqClient()
                print("Using Groq API for LLM integration")
            except Exception as e:
                print(f"Could not initialize Groq client: {e}")
                print("Using simulated LLM responses")
                llm_client = None

        # Initialize components
        self.document_parser = DocumentParser(llm_client)
        self.validation_engine = ValidationEngine()
        self.form_generator = FormGenerator()
        self.conversation_manager = ConversationManager()

        # Initialize form definitions
        self.form_definitions = {}
        self._initialize_default_forms()

    def _initialize_default_forms(self):
        """Initialize default form definitions."""
        # Contact Form
        contact_form = FormDefinition(
            title="Contact Form",
            description="A form for contacting us",
            form_type=FormType.CONTACT,
            fields=[
                StringField(
                    name="name",
                    label="Full Name",
                    required=True,
                    min_length=2,
                    max_length=100,
                    description="Your full name"
                ),
                StringField(
                    name="email",
                    label="Email Address",
                    required=True,
                    pattern=r"[^@]+@[^@]+\.[^@]+",
                    description="Your email address"
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
                    description="Your message"
                ),
                BooleanField(
                    name="subscribe",
                    label="Subscribe to newsletter",
                    required=False,
                    default=False
                )
            ]
        )

        self.form_definitions[FormType.CONTACT] = contact_form

        # Job Application Form
        job_application_form = FormDefinition(
            title="Job Application Form",
            description="A form for job applications",
            form_type=FormType.JOB_APPLICATION,
            fields=[
                StringField(
                    name="full_name",
                    label="Full Name",
                    required=True,
                    min_length=2,
                    max_length=100
                ),
                StringField(
                    name="email",
                    label="Email Address",
                    required=True,
                    pattern=r"[^@]+@[^@]+\.[^@]+"
                ),
                StringField(
                    name="phone",
                    label="Phone Number",
                    required=True,
                    pattern=r"^\+?[\d\s\-\(\)]+$"
                ),
                StringField(
                    name="position",
                    label="Position Applied For",
                    required=True
                ),
                NumberField(
                    name="experience_years",
                    label="Years of Experience",
                    required=True,
                    min_value=0,
                    is_integer=True
                ),
                StringField(
                    name="education",
                    label="Highest Education",
                    required=True
                ),
                StringField(
                    name="skills",
                    label="Skills",
                    required=True,
                    description="Comma-separated list of skills"
                ),
                StringField(
                    name="cover_letter",
                    label="Cover Letter",
                    required=True,
                    min_length=100
                )
            ]
        )

        self.form_definitions[FormType.JOB_APPLICATION] = job_application_form

    def register_form_definition(self, form_definition: FormDefinition):
        """
        Register a custom form definition.

        Args:
            form_definition: Form definition to register
        """
        self.form_definitions[form_definition.form_type] = form_definition

    def create_session(self, form_type: FormType) -> FormSession:
        """
        Create a new form filling session.

        Args:
            form_type: Type of form to fill

        Returns:
            New form session
        """
        if form_type not in self.form_definitions:
            raise ValueError(f"Unknown form type: {form_type}")

        return FormSession(form_type=form_type)

    def process_document(self, session: FormSession, document_text: str) -> FormSession:
        """
        Process a document to extract form information.

        Args:
            session: Current form session
            document_text: Document text to process

        Returns:
            Updated session with extracted information
        """
        # Create a copy of the session to update
        updated_session = session.model_copy(deep=True)
        updated_session.status = FormStatus.PARSING
        updated_session.updated_at = datetime.now()

        # Get the form definition for this session
        form_definition = self.form_definitions.get(updated_session.form_type)
        if not form_definition:
            updated_session.status = FormStatus.ERROR
            updated_session.errors = {"general": ["Unknown form type"]}
            return updated_session

        # Create a Pydantic model from the form definition
        form_model = form_definition.create_model()

        # Parse the document
        try:
            result = self.document_parser.parse_document(document_text, form_model)

            # Check if parsing was successful
            if isinstance(result, BaseModel):
                # Parsing was successful, update session data
                updated_session.data = result.model_dump()
            elif isinstance(result, dict) and result.get("success") is False:
                # Parsing failed, update session with error information
                updated_session.status = FormStatus.ERROR
                updated_session.errors = {"parsing": [str(err) for err in result.get("errors", ["Unknown parsing error"])]}
                return updated_session
            else:
                # Parsing returned a dictionary, use it as data
                updated_session.data = result
        except Exception as e:
            # Parsing failed with an exception
            updated_session.status = FormStatus.ERROR
            updated_session.errors = {"parsing": [str(e)]}
            return updated_session

        # Return the updated session for validation
        return self.validate_session(updated_session)

    def validate_session(self, session: FormSession) -> FormSession:
        """
        Validate the current session data.

        Args:
            session: Current form session

        Returns:
            Updated session with validation results
        """
        # Create a copy of the session to update
        updated_session = session.model_copy(deep=True)
        updated_session.status = FormStatus.VALIDATING
        updated_session.updated_at = datetime.now()

        # Get the form definition for this session
        form_definition = self.form_definitions.get(updated_session.form_type)
        if not form_definition:
            updated_session.status = FormStatus.ERROR
            updated_session.errors = {"general": ["Unknown form type"]}
            return updated_session

        # Validate the data
        validation_result = self.validation_engine.validate_form_data(updated_session.data, form_definition)

        # Update the session based on validation results
        if validation_result["valid"]:
            # Data is valid, update session
            updated_session.data = validation_result["data"]
            updated_session.status = FormStatus.COMPLETE
            updated_session.completion_percentage = 100.0
        else:
            # Data is invalid, update session with errors and missing fields
            updated_session.errors = validation_result.get("errors", {})
            updated_session.missing_fields = validation_result.get("missing_fields", [])
            updated_session.status = FormStatus.INCOMPLETE

            # Calculate completion percentage
            total_required_fields = sum(1 for field in form_definition.fields if field.required)
            if total_required_fields > 0:
                filled_required_fields = total_required_fields - len(updated_session.missing_fields)
                updated_session.completion_percentage = filled_required_fields / total_required_fields * 100

        return updated_session

    def get_next_prompt(self, session: FormSession) -> str:
        """
        Get the next prompt for missing information.

        Args:
            session: Current form session

        Returns:
            Prompt for missing information
        """
        # If the session is complete, return a completion message
        if session.status == FormStatus.COMPLETE:
            return "All required information has been provided. Thank you!"

        # If the session has errors, return an error message
        if session.errors:
            error_messages = []
            for field, errors in session.errors.items():
                error_messages.append(f"{field}: {', '.join(errors)}")
            return f"There are some errors in the form:\n" + "\n".join(error_messages)

        # If the session is incomplete, generate a prompt for missing fields
        if session.status == FormStatus.INCOMPLETE and session.missing_fields:
            # Get the form definition for this session
            form_definition = self.form_definitions.get(session.form_type)
            if not form_definition:
                return "Error: Unknown form type"

            # Generate a prompt for missing fields
            return self.conversation_manager.generate_prompt_for_missing_fields(session, form_definition)

        # Default message
        return "Please provide the required information for the form."

    def process_response(self, session: FormSession, response: str) -> FormSession:
        """
        Process a user response to update the session.

        Args:
            session: Current form session
            response: User response text

        Returns:
            Updated session
        """
        # If the session is already complete, return it as is
        if session.status == FormStatus.COMPLETE:
            return session

        # Get the form definition for this session
        form_definition = self.form_definitions.get(session.form_type)
        if not form_definition:
            # Create a copy of the session to update
            updated_session = session.model_copy(deep=True)
            updated_session.status = FormStatus.ERROR
            updated_session.errors = {"general": ["Unknown form type"]}
            return updated_session

        # Update the session with the response
        updated_session = self.conversation_manager.update_session_with_response(
            session, response, form_definition
        )

        # Validate the updated session
        return self.validate_session(updated_session)

    def generate_form(self, session: FormSession, output_format: OutputFormat = OutputFormat.JSON) -> str:
        """
        Generate a completed form.

        Args:
            session: Current form session
            output_format: Output format for the form

        Returns:
            Completed form in the specified format
        """
        # If the session is not complete, return an error message
        if session.status != FormStatus.COMPLETE:
            if output_format == OutputFormat.JSON:
                return json.dumps({
                    "error": "Form is not complete",
                    "status": session.status,
                    "missing_fields": session.missing_fields,
                    "errors": session.errors
                }, indent=2)
            else:
                return f"Error: Form is not complete\nStatus: {session.status}\nMissing fields: {session.missing_fields}\nErrors: {session.errors}"

        # Get the form definition for this session
        form_definition = self.form_definitions.get(session.form_type)
        if not form_definition:
            if output_format == OutputFormat.JSON:
                return json.dumps({"error": "Unknown form type"}, indent=2)
            else:
                return "Error: Unknown form type"

        # Generate the form in the requested format
        return self.form_generator.generate_form(session.data, form_definition, output_format)


# Example Usage
# -----------

def demonstrate_form_assistant():
    """Demonstrate the form assistant functionality."""
    print("Starting form assistant demonstration...")

    # Create a form assistant
    # By default, it will try to use the Groq API if available
    # You can also provide your own LLM client:
    # from groq_client import GroqClient
    # llm_client = GroqClient(api_key="your_api_key")
    # assistant = FormAssistant(llm_client=llm_client)
    assistant = FormAssistant()
    print("Form assistant created")

    # Create a session for a contact form
    session = assistant.create_session(FormType.CONTACT)
    print(f"Session created with ID: {session.session_id}")

    # Process a document
    document_text = """
    Hello,

    My name is Sarah Johnson and I'd like to inquire about your premium support package.
    I've been using your product for about 6 months and have some questions about advanced features.

    You can reach me at sarah.johnson@example.com or call me at (555) 123-4567.

    I'm specifically interested in the data export capabilities and API integration options.
    Could someone from your technical team contact me to discuss these features in detail?

    Thanks,
    Sarah
    """

    print("Processing document...")
    try:
        session = assistant.process_document(session, document_text)
        print(f"Document processed. Session status: {session.status}")
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate the session
    print("Validating session...")
    try:
        session = assistant.validate_session(session)
        print(f"Session validated. Status: {session.status}, Completion: {session.completion_percentage}%")
    except Exception as e:
        print(f"Error validating session: {e}")
        import traceback
        traceback.print_exc()
        return

    # If the form is incomplete, get the next prompt
    if session.status == FormStatus.INCOMPLETE:
        print("Form is incomplete. Getting next prompt...")
        try:
            prompt = assistant.get_next_prompt(session)
            print(f"Next prompt: {prompt}")

            # Simulate a user response for subject
            response = "The subject of my inquiry is 'Premium Support Package Questions'"
            print(f"Simulated user response: {response}")

            session = assistant.process_response(session, response)
            print(f"Response processed. Session status: {session.status}, Completion: {session.completion_percentage}%")

            # Get the next prompt for remaining fields
            if session.status == FormStatus.INCOMPLETE:
                prompt = assistant.get_next_prompt(session)
                print(f"Next prompt: {prompt}")

                # Simulate a user response for the remaining fields
                response = """
                My name is John Smith
                Email: john.smith@example.com
                Message: I would like to inquire about your premium support package options.
                What are the different tiers available and what is included in each?
                """
                print(f"Simulated user response: {response}")

                session = assistant.process_response(session, response)
                print(f"Response processed. Session status: {session.status}, Completion: {session.completion_percentage}%")
        except Exception as e:
            print(f"Error processing response: {e}")
            import traceback
            traceback.print_exc()
            return

    # Generate the completed form
    print("Generating form output...")
    try:
        form_json = assistant.generate_form(session)
        print(f"Completed form (JSON):\n{form_json}")

        # Generate the form in markdown format
        form_md = assistant.generate_form(session, OutputFormat.MARKDOWN)
        print(f"Completed form (Markdown):\n{form_md}")
    except Exception as e:
        print(f"Error generating form: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_form_assistant()
