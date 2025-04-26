# 🔄 Module 3: Structured Data Validation - Lesson 4.8: Validation Integration with LLM Systems 🤖

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔄 Understand the unique validation challenges presented by LLM systems
- 🛠️ Build robust validation pipelines for LLM-generated content
- 📊 Implement multi-format validators to handle various output structures
- 🧠 Create uncertainty-aware validation systems for probabilistic outputs
- 🔁 Design feedback loops between validation and LLM systems
- 📝 Develop validation-aware prompt templates to improve output quality

---

## 📚 Introduction to LLM Validation Integration

In this lesson, we'll explore how to integrate validation systems with Large Language Models (LLMs). While validation is important for all software systems, it presents unique challenges and opportunities when working with LLMs. We'll learn how to create robust validation pipelines that work with the probabilistic nature of LLMs, implement feedback loops to improve performance, and use validation as part of the learning process.

## 🧩 The Challenge of LLM Validation

LLMs present several unique validation challenges:

1. **Probabilistic Outputs**: LLMs generate different outputs for the same input
2. **Uncertainty**: LLMs may express varying degrees of confidence in their outputs
3. **Hallucinations**: LLMs can generate plausible but incorrect information
4. **Context Limitations**: LLMs may miss important context for validation
5. **Format Inconsistency**: LLMs may not consistently follow output format instructions

> 💡 **Key Insight**: Unlike traditional software systems where inputs and outputs are deterministic, LLMs introduce probabilistic elements that require more flexible, robust validation approaches that can handle uncertainty and variability.

---

## 🛠️ Connecting Validation with LLM-Generated Content

### 🔄 Validation Pipeline for LLM Outputs

```python
from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, List, Dict, Any, Literal, Union, TypeVar, Generic, Type
import json
from datetime import datetime

T = TypeVar('T', bound=BaseModel)

class LLMResponse(BaseModel):
    raw_response: str
    parsed_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = []
    is_valid: bool = False

class LLMValidator(Generic[T]):
    """Generic validator for LLM-generated content."""

    def __init__(self, model_class: Type[T], retry_prompt: Optional[str] = None):
        self.model_class = model_class
        self.retry_prompt = retry_prompt

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text, handling various formats."""
        # Try to find JSON-like structure with regex
        import re
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = re.findall(json_pattern, text)

        if not matches:
            return None

        # Try each match until we find valid JSON
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def validate(self, llm_response: str) -> LLMResponse:
        """Validate LLM response against the model."""
        result = LLMResponse(raw_response=llm_response)

        # Try to extract structured data
        extracted_data = self.extract_json(llm_response)
        result.parsed_data = extracted_data

        if not extracted_data:
            result.validation_errors.append("Could not extract structured data from response")
            return result

        # Validate against model
        try:
            validated_data = self.model_class(**extracted_data)
            result.is_valid = True
        except ValidationError as e:
            for error in e.errors():
                result.validation_errors.append(f"{error['loc']}: {error['msg']}")

        return result

    def generate_retry_prompt(self, validation_result: LLMResponse) -> Optional[str]:
        """Generate a retry prompt based on validation errors."""
        if validation_result.is_valid or not self.retry_prompt:
            return None

        error_details = "\n".join([f"- {error}" for error in validation_result.validation_errors])

        retry_prompt = f"{self.retry_prompt}\n\nThe previous response had the following issues:\n{error_details}\n\nPlease provide a corrected response."

        return retry_prompt

# Example usage with a weather forecast model
class WeatherForecast(BaseModel):
    location: str
    date: datetime
    temperature: float = Field(ge=-100, le=150)  # Reasonable temperature range in Fahrenheit
    condition: str
    precipitation_chance: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def validate_condition(self):
        """Validate that condition is a known weather condition."""
        valid_conditions = [
            "sunny", "partly cloudy", "cloudy", "rainy", "stormy",
            "snowy", "foggy", "windy", "clear"
        ]
        if self.condition.lower() not in valid_conditions:
            raise ValueError(f"Unknown weather condition: {self.condition}")
        return self

# Usage
weather_validator = LLMValidator(
    model_class=WeatherForecast,
    retry_prompt="Please provide a weather forecast in JSON format with location, date, temperature (°F), condition, and precipitation_chance (0-1)."
)

# Example LLM response
llm_response = """
I've checked the forecast for tomorrow. Here's what I found:

{
  "location": "New York",
  "date": "2023-12-25T00:00:00",
  "temperature": 45.5,
  "condition": "partly cloudy",
  "precipitation_chance": 0.2
}

Hope that helps!
"""

validation_result = weather_validator.validate(llm_response)
print(f"Is valid: {validation_result.is_valid}")
if validation_result.validation_errors:
    print("Validation errors:")
    for error in validation_result.validation_errors:
        print(f"- {error}")

    retry_prompt = weather_validator.generate_retry_prompt(validation_result)
    if retry_prompt:
        print(f"\nRetry prompt:\n{retry_prompt}")
```

### 📊 Handling Multiple Output Formats

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union, TypeVar, Generic, Type
import json
from datetime import datetime

class FormatHandler(BaseModel):
    """Base class for format handlers."""
    format_name: str

    def can_handle(self, text: str) -> bool:
        """Check if this handler can process the given text."""
        raise NotImplementedError

    def extract_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text."""
        raise NotImplementedError

class JSONFormatHandler(FormatHandler):
    format_name: str = "JSON"

    def can_handle(self, text: str) -> bool:
        """Check if text contains JSON."""
        return "{" in text and "}" in text

    def extract_data(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from text."""
        import re
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = re.findall(json_pattern, text)

        if not matches:
            raise ValueError("No JSON object found in text")

        # Try each match until we find valid JSON
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not parse JSON from text")

class KeyValueFormatHandler(FormatHandler):
    format_name: str = "Key-Value"

    def can_handle(self, text: str) -> bool:
        """Check if text contains key-value pairs."""
        import re
        kv_pattern = r'(\w+):\s*(.+)'
        matches = re.findall(kv_pattern, text)
        return len(matches) > 0

    def extract_data(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text."""
        import re
        kv_pattern = r'(\w+):\s*(.+)'
        matches = re.findall(kv_pattern, text)

        if not matches:
            raise ValueError("No key-value pairs found in text")

        result = {}
        for key, value in matches:
            # Try to convert value to appropriate type
            try:
                # Try as number
                if '.' in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Try as boolean
                if value.lower() in ('true', 'yes'):
                    result[key] = True
                elif value.lower() in ('false', 'no'):
                    result[key] = False
                else:
                    # Keep as string
                    result[key] = value.strip()

        return result

class MultiFormatValidator(Generic[T]):
    """Validator that can handle multiple input formats."""

    def __init__(self, model_class: Type[T], format_handlers: Optional[List[FormatHandler]] = None):
        self.model_class = model_class
        self.format_handlers = format_handlers or [
            JSONFormatHandler(),
            KeyValueFormatHandler()
        ]

    def validate(self, text: str) -> LLMResponse:
        """Validate text using appropriate format handler."""
        result = LLMResponse(raw_response=text)

        # Try each handler
        for handler in self.format_handlers:
            if handler.can_handle(text):
                try:
                    extracted_data = handler.extract_data(text)
                    result.parsed_data = extracted_data

                    # Validate against model
                    try:
                        validated_data = self.model_class(**extracted_data)
                        result.is_valid = True
                        break
                    except ValidationError as e:
                        for error in e.errors():
                            result.validation_errors.append(f"{error['loc']}: {error['msg']}")
                except ValueError as e:
                    result.validation_errors.append(f"{handler.format_name} extraction error: {str(e)}")

        if not result.parsed_data:
            result.validation_errors.append("Could not extract data with any available format handler")

        return result

# Usage
multi_validator = MultiFormatValidator(model_class=WeatherForecast)

# JSON format
json_response = """
Here's the weather forecast:

{
  "location": "Boston",
  "date": "2023-12-26T00:00:00",
  "temperature": 32.5,
  "condition": "snowy",
  "precipitation_chance": 0.7
}
"""

# Key-value format
kv_response = """
Weather Forecast:
location: Chicago
date: 2023-12-26T00:00:00
temperature: 28.3
condition: windy
precipitation_chance: 0.1
"""

json_result = multi_validator.validate(json_response)
print(f"JSON format valid: {json_result.is_valid}")
if not json_result.is_valid:
    print("Errors:", json_result.validation_errors)

kv_result = multi_validator.validate(kv_response)
print(f"Key-value format valid: {kv_result.is_valid}")
if not kv_result.is_valid:
    print("Errors:", kv_result.validation_errors)
```

---

## 🧠 Handling Validation in the Context of Uncertainty

LLMs often express uncertainty in their responses. We can incorporate this into our validation:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union
import re
from datetime import datetime

class UncertainValue(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: Optional[List[Any]] = None

class UncertainResponse(BaseModel):
    content: Dict[str, Any]
    uncertain_fields: Dict[str, UncertainValue] = {}

    def get_value(self, field: str, confidence_threshold: float = 0.0):
        """Get a value, considering uncertainty."""
        if field in self.uncertain_fields and self.uncertain_fields[field].confidence < confidence_threshold:
            return None

        if field in self.uncertain_fields:
            return self.uncertain_fields[field].value

        return self.content.get(field)

class UncertaintyParser:
    """Parse uncertainty expressions in LLM responses."""

    def __init__(self):
        self.uncertainty_patterns = [
            (r"I'm not sure, but (?:I think |maybe |possibly |perhaps )?(?:it's|it is|that's|that is) (.*)", 0.5),
            (r"(?:I think|I believe|Probably|Likely|It seems) (.*)", 0.7),
            (r"(?:I'm confident|I'm certain|Definitely|Certainly|Without doubt) (.*)", 0.9),
            (r"(?:it might be|it could be|possibly|perhaps) (.*)", 0.3),
        ]

    def parse_uncertainty(self, text: str) -> Dict[str, UncertainValue]:
        """Extract uncertain values from text."""
        uncertain_fields = {}

        # Look for field-specific uncertainty
        field_uncertainty_pattern = r"(?:For|Regarding|About) the ([\w\s]+), (.*)"
        field_matches = re.findall(field_uncertainty_pattern, text)

        for field_name, field_text in field_matches:
            field_name = field_name.strip().lower().replace(" ", "_")

            # Check for uncertainty expressions
            for pattern, confidence in self.uncertainty_patterns:
                match = re.search(pattern, field_text)
                if match:
                    value_text = match.group(1)

                    # Try to parse the value
                    value = self._parse_value(value_text)

                    # Look for alternatives
                    alternatives = self._find_alternatives(field_text)

                    uncertain_fields[field_name] = UncertainValue(
                        value=value,
                        confidence=confidence,
                        alternatives=alternatives if alternatives else None
                    )
                    break

        return uncertain_fields

    def _parse_value(self, text: str) -> Any:
        """Parse a value from text, attempting to infer type."""
        text = text.strip()

        # Try as number
        try:
            if '.' in text:
                return float(text)
            return int(text)
        except ValueError:
            pass

        # Try as boolean
        if text.lower() in ('true', 'yes'):
            return True
        if text.lower() in ('false', 'no'):
            return False

        # Return as string
        return text

    def _find_alternatives(self, text: str) -> Optional[List[Any]]:
        """Find alternative values in text."""
        alternatives_pattern = r"(?:but it could also be|alternatives include|or maybe|or possibly) (.*)"
        match = re.search(alternatives_pattern, text)

        if not match:
            return None

        alternatives_text = match.group(1)

        # Split by commas or 'or'
        parts = re.split(r',\s*|\s+or\s+', alternatives_text)

        return [self._parse_value(part) for part in parts if part.strip()]

# Usage
uncertainty_parser = UncertaintyParser()

llm_response = """
Here's what I found about the weather:

For the temperature, I think it's around 45.5 degrees, but it could also be 44 or 47 depending on the exact location.

Regarding the condition, it's likely partly cloudy, but it might change to cloudy later.

For the precipitation chance, I'm not sure, but I think it's about 0.3.
"""

uncertain_fields = uncertainty_parser.parse_uncertainty(llm_response)

# Create an uncertain response
response = UncertainResponse(
    content={
        "location": "New York",
        "date": "2023-12-25T00:00:00",
        "temperature": 45.5,
        "condition": "partly cloudy",
        "precipitation_chance": 0.3
    },
    uncertain_fields=uncertain_fields
)

# Get values with different confidence thresholds
print("Temperature (any confidence):", response.get_value("temperature"))
print("Temperature (confidence > 0.8):", response.get_value("temperature", 0.8))
print("Condition (any confidence):", response.get_value("condition"))
print("Precipitation (any confidence):", response.get_value("precipitation_chance"))
```

---

## 🔁 Feedback Loops Between Validation and LLM Systems

Creating feedback loops to improve LLM outputs:

```python
from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, List, Dict, Any, Literal, Union, TypeVar, Generic, Type
import json
from datetime import datetime

class ValidationFeedback(BaseModel):
    """Feedback about validation results for improving LLM outputs."""
    original_prompt: str
    original_response: str
    validation_result: LLMResponse
    improved_prompt: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    @model_validator(mode='after')
    def generate_improved_prompt(self):
        """Generate an improved prompt based on validation errors."""
        if self.validation_result.is_valid or self.retry_count >= self.max_retries:
            return self

        # Extract schema information
        schema_info = self._extract_schema_info()

        # Format error information
        error_info = self._format_error_info()

        # Generate improved prompt
        self.improved_prompt = f"""
I need you to provide information in a specific format. My previous request was:

{self.original_prompt}

Your previous response had these validation issues:
{error_info}

Please provide a new response that follows this schema:
{schema_info}

Make sure all fields are present and valid.
"""

        return self

    def _extract_schema_info(self) -> str:
        """Extract schema information from validation errors."""
        # This is a simplified example; in practice, you would extract this from the model
        schema_parts = []

        for error in self.validation_result.validation_errors:
            if "missing" in error.lower():
                field = error.split(":")[0].strip()
                schema_parts.append(f"- {field}: Required")
            elif "not a valid" in error.lower():
                parts = error.split(":")
                if len(parts) >= 2:
                    field = parts[0].strip()
                    expected_type = parts[1].strip().split(" ")[-1]
                    schema_parts.append(f"- {field}: Must be a valid {expected_type}")

        return "\n".join(schema_parts)

    def _format_error_info(self) -> str:
        """Format error information for the prompt."""
        return "\n".join([f"- {error}" for error in self.validation_result.validation_errors])

class FeedbackLoop:
    """Implements a feedback loop between validation and LLM."""

    def __init__(self, validator: Union[LLMValidator, MultiFormatValidator], llm_function):
        self.validator = validator
        self.llm_function = llm_function
        self.feedback_history = []

    def process_with_feedback(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Process a prompt with validation feedback loop."""
        retry_count = 0

        while retry_count <= max_retries:
            # Get response from LLM
            llm_response = self.llm_function(prompt)

            # Validate response
            validation_result = self.validator.validate(llm_response)

            # Create feedback
            feedback = ValidationFeedback(
                original_prompt=prompt,
                original_response=llm_response,
                validation_result=validation_result,
                retry_count=retry_count,
                max_retries=max_retries
            )

            self.feedback_history.append(feedback)

            # If valid or max retries reached, return result
            if validation_result.is_valid or retry_count >= max_retries:
                return {
                    "response": llm_response,
                    "validation_result": validation_result,
                    "retry_count": retry_count,
                    "is_valid": validation_result.is_valid
                }

            # Update prompt and retry
            prompt = feedback.improved_prompt
            retry_count += 1

        # Should not reach here, but just in case
        return {
            "response": llm_response,
            "validation_result": validation_result,
            "retry_count": retry_count,
            "is_valid": False
        }

# Simulated LLM function for demonstration
def simulated_llm(prompt: str) -> str:
    """Simulate an LLM response for demonstration purposes."""
    if "weather" in prompt.lower():
        if "validation issues" in prompt.lower():
            # Improved response after feedback
            return """
Here's the corrected weather forecast:

{
  "location": "New York",
  "date": "2023-12-25T00:00:00",
  "temperature": 45.5,
  "condition": "partly cloudy",
  "precipitation_chance": 0.2
}
"""
        else:
            # Initial response with issues
            return """
Here's the weather forecast:

{
  "location": "New York",
  "date": "2023-12-25T00:00:00",
  "temperature": 45.5,
  "condition": "nice",  # Invalid condition
  "precipitation_chance": 1.2  # Invalid probability
}
"""

    return "I don't know how to respond to that."

# Usage
weather_validator = LLMValidator(model_class=WeatherForecast)
feedback_loop = FeedbackLoop(validator=weather_validator, llm_function=simulated_llm)

result = feedback_loop.process_with_feedback(
    prompt="What's the weather forecast for New York tomorrow?"
)

print(f"Final result valid: {result['is_valid']}")
print(f"Retry count: {result['retry_count']}")
print(f"Final response: {result['response']}")
```

---

## 📝 Prompt Engineering for Better Validation

Improving validation through better prompts:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class PromptTemplate(BaseModel):
    """Template for generating prompts that lead to validatable responses."""
    template: str
    required_variables: List[str] = []
    example_output: Optional[str] = None
    format_instructions: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Format the template with the provided variables."""
        # Check that all required variables are provided
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")

        # Format the template
        prompt = self.template.format(**kwargs)

        # Add format instructions if available
        if self.format_instructions:
            prompt += f"\n\n{self.format_instructions}"

        # Add example output if available
        if self.example_output:
            prompt += f"\n\nExample output format:\n{self.example_output}"

        return prompt

class ValidationAwarePromptTemplate(PromptTemplate):
    """Prompt template that includes validation requirements."""
    validation_model: Optional[Type[BaseModel]] = None
    include_schema: bool = True

    def format(self, **kwargs) -> str:
        """Format the template with validation information."""
        prompt = super().format(**kwargs)

        if self.validation_model and self.include_schema:
            # Add schema information
            schema = self._get_schema_description()
            prompt += f"\n\nYour response must conform to this schema:\n{schema}"

        return prompt

    def _get_schema_description(self) -> str:
        """Get a human-readable description of the schema."""
        if not self.validation_model:
            return ""

        schema = self.validation_model.model_json_schema()

        # Create a simplified description
        description = []

        if "title" in schema:
            description.append(f"# {schema['title']}")

        if "description" in schema:
            description.append(f"{schema['description']}")

        description.append("Fields:")

        for field_name, field_info in schema.get("properties", {}).items():
            field_desc = f"- {field_name}"

            if "type" in field_info:
                field_desc += f" ({field_info['type']})"

            if "description" in field_info:
                field_desc += f": {field_info['description']}"

            if field_name in schema.get("required", []):
                field_desc += " [REQUIRED]"

            description.append(field_desc)

        return "\n".join(description)

# Example usage with weather forecast
class WeatherForecastWithDesc(BaseModel):
    """Weather forecast for a specific location and date."""
    location: str = Field(description="The city or location name")
    date: datetime = Field(description="The date of the forecast")
    temperature: float = Field(ge=-100, le=150, description="Temperature in Fahrenheit")
    condition: str = Field(description="Weather condition (sunny, partly cloudy, cloudy, rainy, stormy, snowy, foggy, windy, clear)")
    precipitation_chance: float = Field(ge=0, le=1, description="Probability of precipitation (0-1)")

# Create a validation-aware prompt template
weather_prompt = ValidationAwarePromptTemplate(
    template="What's the weather forecast for {location} on {date}?",
    required_variables=["location", "date"],
    validation_model=WeatherForecastWithDesc,
    example_output="""
{
  "location": "New York",
  "date": "2023-12-25T00:00:00",
  "temperature": 45.5,
  "condition": "partly cloudy",
  "precipitation_chance": 0.2
}
"""
)

# Format the prompt
formatted_prompt = weather_prompt.format(
    location="Boston",
    date="tomorrow"
)

print(formatted_prompt)
```

---

## 💪 Practice Exercises

1. **Create a Form-Filling Validation Pipeline**: Build a validation pipeline for an LLM-based form-filling assistant that can extract and validate user information from natural language inputs.

2. **Implement a Multi-Format Validator**: Develop a validator that can handle JSON, YAML, and key-value pair formats for the same data model.

3. **Build a Feedback Loop System**: Create a system that tracks common validation errors and automatically improves prompts to reduce those errors.

4. **Design an Uncertainty-Aware Validator**: Implement a validation system for a financial advisor agent that handles probabilistic predictions.

5. **Create a Healthcare Prompt Template**: Build a validation-aware prompt template system for a healthcare agent that ensures all medical information includes appropriate disclaimers and safety checks.

---

## 🔍 Key Concepts to Remember

1. **LLM-Specific Validation**: Handling probabilistic outputs and uncertainty in LLM responses
2. **Multi-Format Validation**: Creating robust validators that can handle various output structures
3. **Feedback Loops**: Using validation results to improve LLM outputs over time
4. **Uncertainty Handling**: Incorporating confidence levels and alternatives into validation
5. **Prompt Engineering**: Designing prompts that lead to more validatable outputs
6. **Integrated Validation**: Building validation into every step of the LLM interaction process

---

## 📚 Additional Resources

- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Validation Documentation](https://docs.pydantic.dev/latest/usage/validators/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Handling Uncertainty in LLMs](https://arxiv.org/abs/2304.00612)
- [Evaluating LLM Outputs](https://huggingface.co/blog/evaluating-llm-outputs)
- [JSON Schema for LLM Outputs](https://json-schema.org/understanding-json-schema/)

---

## 🚀 Next Steps

Congratulations on completing the lessons on validation in agent systems! In the upcoming mini-project, we'll apply everything we've learned to build a Form-Filling Assistant that can extract, validate, and process information from unstructured text inputs. This project will bring together concepts from all the lessons in this module, from basic Pydantic validation to complex LLM integration.

---

> 💡 **Note on LLM Integration**: This lesson represents the culmination of our validation journey, bringing together all the concepts we've learned and applying them specifically to LLM-based systems. The techniques covered here are essential for building reliable, robust agent systems that can handle the inherent uncertainty and variability of LLM outputs while still maintaining data integrity and consistency.

---

Happy coding! 🤖
