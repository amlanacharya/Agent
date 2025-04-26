# 🧩 Module 3: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 3: Data Validation & Structured Outputs with Pydantic. These examples demonstrate Pydantic fundamentals, schema design patterns, structured output parsing, and advanced validation techniques.

## 🔍 File Descriptions

### Core Implementations
- **pydantic_basics.py**: Basic Pydantic models and validation techniques
- **schema_design.py**: Schema design and evolution patterns for maintainable data models
- **output_parsing.py**: Structured output parsing with LLMs and validation
- **model_composition.py**: Advanced model composition and inheritance patterns
- **form_assistant.py**: Complete form-filling assistant implementation

### Test Scripts
- **test_pydantic_basics.py**: Tests for basic Pydantic usage
- **test_schema_design.py**: Tests for schema design patterns
- **test_output_parsing.py**: Tests for output parsing techniques
- **test_model_composition.py**: Tests for model composition patterns
- **test_form_assistant.py**: Tests for the form-filling assistant

## 🚀 Running the Examples

You can run any of the examples directly from the command line:

```bash
# Run from the project root
python -m module3.code.pydantic_basics
python -m module3.code.schema_design
python -m module3.code.output_parsing
python -m module3.code.model_composition
python -m module3.code.form_assistant
```

To run the tests:

```bash
# Run from the project root
python -m module3.code.test_pydantic_basics
python -m module3.code.test_schema_design
python -m module3.code.test_output_parsing
python -m module3.code.test_model_composition
python -m module3.code.test_form_assistant
```

To run the interactive demos:

```bash
# Run the unified demo with a menu of all Pydantic features
python -m module3.demo_pydantic_features

# Or run a specific component demo directly
python -m module3.demo_pydantic_features basics      # Pydantic Basics
python -m module3.demo_pydantic_features schema      # Schema Design
python -m module3.demo_pydantic_features parsing     # Output Parsing
python -m module3.demo_pydantic_features composition # Model Composition
python -m module3.demo_pydantic_features validation  # Advanced Validation Patterns

# You can also run individual demo files directly
python -m module3.demo_pydantic_basics
python -m module3.demo_schema_design
python -m module3.demo_output_parsing
python -m module3.demo_user_profile
python -m module3.demo_model_composition
```

## 📋 Implementation Notes

- The implementations use Pydantic v2 for improved performance and features
- The output parsing examples integrate with the Groq API for LLM-generated structured outputs
- The schema design patterns demonstrate versioning and backward compatibility
- The model composition examples show inheritance, mixins, and composition patterns
- The form assistant combines all concepts into a complete application

## 🔄 LLM Integration

> 💡 **Note**: Module 3 integrates with real LLMs through the Groq API for generating structured outputs that can be validated and processed reliably.

## 🧪 Example Usage

Here's a simple example of how to use Pydantic for data validation:

```python
# Example code snippet showing basic usage
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from datetime import datetime

class User(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    bio: Optional[str] = None

try:
    # Valid user
    user = User(id=1, name="John Doe", email="john@example.com", tags=["customer", "premium"])
    print(user.model_dump_json(indent=2))

    # Invalid user (email missing)
    invalid_user = User(id=2, name="Jane Doe")
except ValidationError as e:
    print(f"Validation error: {e}")
```

And here's how to use the output parsing with LLMs:

```python
# Example code snippet showing output parsing
from module3.code.output_parsing import StructuredOutputParser, UserProfileSchema

# Create a parser
parser = StructuredOutputParser(output_schema=UserProfileSchema)

# Parse LLM output
llm_output = """
{
  "name": "Alice Johnson",
  "age": 28,
  "occupation": "Software Engineer",
  "interests": ["programming", "hiking", "photography"]
}
"""

try:
    profile = parser.parse(llm_output)
    print(f"Parsed profile: {profile}")
except Exception as e:
    print(f"Parsing error: {e}")
```

## 🛠️ Extending the Code

Here are some ideas for extending or customizing the implementations:

1. Add more complex validation rules using Pydantic validators
2. Implement schema migration tools for evolving data models
3. Create domain-specific output parsers for specialized applications
4. Add persistence layer to store validated data
5. Implement a web API around the form assistant

## 📚 Related Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Type Hints Guide](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [JSON Schema Documentation](https://json-schema.org/learn/getting-started-step-by-step)
- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
