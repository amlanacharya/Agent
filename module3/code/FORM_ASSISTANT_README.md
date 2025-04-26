# Form Assistant

A powerful form assistant that uses LLM capabilities to extract information from unstructured text, validate form data, and generate completed forms in various formats.

## Features

- **Document Parsing**: Extract structured data from unstructured text using LLM capabilities
- **Form Validation**: Validate form data against defined rules and constraints
- **Conversation Management**: Generate prompts for missing information and process user responses
- **Form Generation**: Generate completed forms in various formats (JSON, YAML, Markdown, HTML, Text)
- **LLM Integration**: Use the Groq API for advanced text processing capabilities
- **Fallback Mechanisms**: Gracefully degrade to simulated responses when LLM is not available

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: `pydantic`, `requests`, `python-dotenv`

### Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install pydantic requests python-dotenv
   ```
3. Set up your Groq API key:
   - Copy `.env.template` to `.env`
   - Add your Groq API key to the `.env` file

### Usage

```python
from form_assistant import FormAssistant, FormType, OutputFormat

# Create a form assistant
assistant = FormAssistant()

# Create a session for a contact form
session = assistant.create_session(FormType.CONTACT)

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

# Extract information from the document
session = assistant.process_document(session, document_text)

# Get the next prompt for missing information
prompt = assistant.get_next_prompt(session)
print(prompt)

# Process a user response
response = "The subject of my inquiry is 'Premium Support Package Questions'"
session = assistant.process_response(session, response)

# Generate the completed form in JSON format
form_json = assistant.generate_form(session)
print(form_json)

# Generate the form in markdown format
form_md = assistant.generate_form(session, OutputFormat.MARKDOWN)
print(form_md)
```

## LLM Integration

The form assistant can use the Groq API for advanced text processing capabilities. To use the Groq API:

1. Get an API key from [Groq Console](https://console.groq.com/)
2. Set the `GROQ_API_KEY` environment variable or provide it directly:

```python
from groq_client import GroqClient
from form_assistant import FormAssistant

# Create a Groq client with your API key
llm_client = GroqClient(api_key="your_api_key")

# Create a form assistant with the Groq client
assistant = FormAssistant(llm_client=llm_client)
```

If the Groq API is not available, the form assistant will fall back to simulated responses.

## Customizing Forms

You can create custom form definitions:

```python
from form_assistant import (
    FormAssistant, FormDefinition, FormType,
    StringField, NumberField, BooleanField, DateField, SelectField
)

# Create a custom form definition
survey_form = FormDefinition(
    title="Customer Satisfaction Survey",
    description="A survey to gather customer feedback",
    form_type=FormType.SURVEY,
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
            pattern=r"[^@]+@[^@]+\.[^@]+"
        ),
        SelectField(
            name="satisfaction",
            label="Overall Satisfaction",
            required=True,
            options=[
                {"label": "Very Satisfied", "value": "very_satisfied"},
                {"label": "Satisfied", "value": "satisfied"},
                {"label": "Neutral", "value": "neutral"},
                {"label": "Dissatisfied", "value": "dissatisfied"},
                {"label": "Very Dissatisfied", "value": "very_dissatisfied"}
            ]
        ),
        NumberField(
            name="rating",
            label="Rating (1-10)",
            required=True,
            min_value=1,
            max_value=10,
            is_integer=True
        ),
        StringField(
            name="feedback",
            label="Additional Feedback",
            required=False
        )
    ]
)

# Register the custom form definition
assistant = FormAssistant()
assistant.register_form_definition(survey_form)

# Create a session for the custom form
session = assistant.create_session(FormType.SURVEY)
```

## Architecture

The form assistant is built with a modular architecture:

1. **FormDefinition**: Defines the structure and validation rules for a form
2. **DocumentParser**: Extracts structured data from unstructured text
3. **ValidationEngine**: Validates form data against defined rules
4. **ConversationManager**: Manages the conversation flow for collecting missing information
5. **FormGenerator**: Generates completed forms in various formats
6. **FormAssistant**: Orchestrates the form-filling process

## License

This project is licensed under the MIT License - see the LICENSE file for details.
