"""
Output Parsers - Structured Output Parsing for LLMs
------------------------------------------------
This module demonstrates techniques for parsing and validating LLM outputs using Pydantic.
"""

import json
import re
from typing import List, Optional, Dict, Any, Union, Callable, TypeVar
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Basic Pydantic models for parsing LLM outputs
# --------------------------------------------

class Person(BaseModel):
    """Basic person model for demonstration."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's professional skills")


class ContactForm(BaseModel):
    """Contact form model for form extraction example."""
    full_name: str = Field(description="The person's full name")
    email: str = Field(description="The person's email address")
    phone: Optional[str] = Field(None, description="The person's phone number")
    address: Optional[str] = Field(None, description="The person's physical address")
    inquiry_type: str = Field(description="The type of inquiry (e.g., support, sales, information)")
    message: str = Field(description="The content of the inquiry message")
    
    @field_validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        """Validate phone number format if provided."""
        if v is not None and not re.match(r"^\+?[\d\s\-\(\)]+$", v):
            raise ValueError("Invalid phone number format")
        return v


# Basic Output Parsing Functions
# ----------------------------

def parse_json_output(output_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output text.
    
    Args:
        output_text: Text output from an LLM
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Try to find JSON in the output
    try:
        # Look for JSON-like structure
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = output_text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON errors
        try:
            # Replace single quotes with double quotes
            fixed_json = output_text.replace("'", '"')
            start_idx = fixed_json.find('{')
            end_idx = fixed_json.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = fixed_json[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
    
    # Try to find JSON in code blocks
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', output_text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except:
            pass
    
    # If all parsing attempts fail
    raise ValueError("Could not parse JSON from LLM output")


def parse_llm_output(output_text: str, model_class: type) -> BaseModel:
    """
    Parse LLM output into a Pydantic model, handling common errors.
    
    Args:
        output_text: Text output from an LLM
        model_class: Pydantic model class to parse into
        
    Returns:
        Instance of the model_class
        
    Raises:
        ValueError: If output cannot be parsed into the model
    """
    try:
        # Try to parse JSON from the output
        data = parse_json_output(output_text)
        
        # Validate with Pydantic model
        return model_class(**data)
    except Exception as e:
        raise ValueError(f"Could not parse LLM output as {model_class.__name__}: {str(e)}")


# Output Parser Classes
# -------------------

class PydanticOutputParser:
    """
    Parser for converting LLM outputs to Pydantic models.
    Similar to LangChain's PydanticOutputParser but simplified.
    """
    
    def __init__(self, pydantic_object: type):
        """
        Initialize the parser with a Pydantic model class.
        
        Args:
            pydantic_object: Pydantic model class to parse into
        """
        self.pydantic_object = pydantic_object
    
    def parse(self, text: str) -> BaseModel:
        """
        Parse text into the Pydantic model.
        
        Args:
            text: Text to parse
            
        Returns:
            Instance of the Pydantic model
            
        Raises:
            ValueError: If text cannot be parsed into the model
        """
        return parse_llm_output(text, self.pydantic_object)
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the LLM on how to format its output.
        
        Returns:
            Formatting instructions as a string
        """
        schema = self.pydantic_object.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        return f"""
        You must format your output as a JSON instance that conforms to the JSON schema below.

        {schema_str}
        
        The JSON output should contain only the required fields and should be valid JSON.
        """


class StructuredOutputParser:
    """
    Parser for structured outputs with custom schemas.
    Similar to LangChain's StructuredOutputParser but simplified.
    """
    
    def __init__(self, response_schemas: List[Dict[str, str]]):
        """
        Initialize the parser with response schemas.
        
        Args:
            response_schemas: List of schema dictionaries with 'name' and 'description' keys
        """
        self.response_schemas = response_schemas
    
    @classmethod
    def from_response_schemas(cls, response_schemas: List[Dict[str, str]]):
        """
        Create a parser from response schemas.
        
        Args:
            response_schemas: List of schema dictionaries with 'name' and 'description' keys
            
        Returns:
            StructuredOutputParser instance
        """
        return cls(response_schemas)
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text into a dictionary based on the response schemas.
        
        Args:
            text: Text to parse
            
        Returns:
            Dictionary with parsed values
            
        Raises:
            ValueError: If text cannot be parsed
        """
        try:
            return parse_json_output(text)
        except ValueError as e:
            raise ValueError(f"Could not parse structured output: {str(e)}")
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the LLM on how to format its output.
        
        Returns:
            Formatting instructions as a string
        """
        schema_str = json.dumps({s["name"]: s["description"] for s in self.response_schemas}, indent=2)
        
        return f"""
        You must format your output as a JSON object with the following keys:
        
        {schema_str}
        
        The JSON output should contain only these keys and should be valid JSON.
        """


# Advanced Parsing Strategies
# -------------------------

T = TypeVar('T')

def parse_with_retry(
    llm_call: Callable[[str], str],
    parser: Union[PydanticOutputParser, StructuredOutputParser],
    text: str,
    max_retries: int = 3
) -> T:
    """
    Try to parse LLM output, retrying with more explicit instructions if it fails.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        parser: Parser to use for parsing the output
        text: Text to process
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed output
        
    Raises:
        ValueError: If parsing fails after all retries
    """
    prompt_template = """
    Extract information from the text below:
    
    {text}
    
    {format_instructions}
    
    {retry_instructions}
    """
    
    retry_instructions = ""
    
    for attempt in range(max_retries):
        # Format the prompt
        prompt = prompt_template.format(
            text=text,
            format_instructions=parser.get_format_instructions(),
            retry_instructions=retry_instructions
        )
        
        # Call the LLM
        output = llm_call(prompt)
        
        try:
            return parser.parse(output)
        except Exception as e:
            if attempt < max_retries - 1:
                # Add more explicit instructions for the next attempt
                retry_instructions = f"""
                The previous response could not be parsed correctly. Error: {e}
                
                Please make sure your response strictly follows the format instructions.
                Double-check that:
                1. All required fields are included
                2. The JSON format is valid
                3. Field types match the expected types (e.g., numbers for age)
                """
            else:
                raise ValueError(f"Failed to parse output after {max_retries} attempts: {e}")


def two_stage_parsing(
    llm_call: Callable[[str], str],
    text: str,
    model_class: type
) -> T:
    """
    First extract structured data, then validate with Pydantic.
    This approach gives the LLM more flexibility in the initial extraction.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        text: Text to process
        model_class: Pydantic model class to parse into
        
    Returns:
        Instance of the model_class
        
    Raises:
        ValueError: If parsing fails
    """
    # Stage 1: Extract information in a flexible format
    extraction_prompt = f"""
    Extract the following information from the text:
    
    {model_class.model_json_schema()}
    
    Text: {text}
    
    Provide the information in JSON format.
    """
    
    initial_output = llm_call(extraction_prompt)
    
    # Stage 2: Refine and validate the extracted information
    validation_prompt = f"""
    I've extracted the following information:
    
    {initial_output}
    
    Please format this as valid JSON that conforms to the following schema:
    
    {model_class.model_json_schema()}
    
    Ensure all required fields are present and correctly typed.
    """
    
    refined_output = llm_call(validation_prompt)
    
    # Parse with Pydantic
    try:
        data = parse_json_output(refined_output)
        return model_class(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse refined output: {e}")


def parse_with_fallbacks(
    llm_call: Callable[[str], str],
    text: str,
    parsers: Dict[str, Callable[[Callable[[str], str], str], T]]
) -> Union[T, Dict[str, Any]]:
    """
    Try multiple parsing strategies in sequence.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        text: Text to process
        parsers: Dictionary mapping parser names to parser functions
        
    Returns:
        Parsed output or error information
    """
    errors = []
    
    for parser_name, parser_func in parsers.items():
        try:
            return parser_func(llm_call, text)
        except Exception as e:
            errors.append(f"{parser_name}: {str(e)}")
    
    # If all parsers fail, return a structured error
    return {
        "success": False,
        "errors": errors,
        "raw_text": text
    }


def parse_with_human_fallback(
    llm_call: Callable[[str], str],
    text: str,
    parser: Union[PydanticOutputParser, StructuredOutputParser],
    human_input_func: Callable[[str, str, str], Optional[str]] = None
) -> Optional[Union[T, Dict[str, Any]]]:
    """
    Try to parse automatically, but fall back to human review if needed.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        text: Text to process
        parser: Parser to use for parsing the output
        human_input_func: Function to get input from a human
        
    Returns:
        Parsed output or None if skipped
    """
    prompt = f"""
    Extract information from the text below:
    
    {text}
    
    {parser.get_format_instructions()}
    """
    
    try:
        llm_output = llm_call(prompt)
        return parser.parse(llm_output)
    except Exception as e:
        error_message = f"Automatic parsing failed: {e}"
        
        if human_input_func:
            human_input = human_input_func(text, llm_output, error_message)
            
            if human_input is None or human_input.lower() == 'skip':
                return None
            else:
                # Assume the human provides valid JSON
                try:
                    data = json.loads(human_input)
                    return data
                except:
                    return None
        else:
            # Default implementation if no human_input_func provided
            print(error_message)
            print(f"Original text: {text}")
            print(f"LLM output: {llm_output}")
            
            human_input = input("Please correct the parsing issue or type 'skip' to ignore: ")
            
            if human_input.lower() == 'skip':
                return None
            else:
                # Assume the human provides valid JSON
                try:
                    return json.loads(human_input)
                except:
                    return None


# Practical Example: Form Extraction
# --------------------------------

class FormExtractor:
    """Extract structured form data from unstructured text."""
    
    def __init__(self, llm_call: Callable[[str], str], form_model: type):
        """
        Initialize the form extractor.
        
        Args:
            llm_call: Function that calls the LLM with a prompt
            form_model: Pydantic model class for the form
        """
        self.llm_call = llm_call
        self.form_model = form_model
        self.parser = PydanticOutputParser(pydantic_object=form_model)
    
    def extract(self, text: str, max_retries: int = 2) -> Union[BaseModel, Dict[str, Any]]:
        """
        Extract form data from text.
        
        Args:
            text: Text to extract form data from
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted form data or error information
        """
        try:
            return parse_with_retry(
                llm_call=self.llm_call,
                parser=self.parser,
                text=text,
                max_retries=max_retries
            )
        except Exception as e:
            # Fall back to two-stage parsing
            try:
                return two_stage_parsing(
                    llm_call=self.llm_call,
                    text=text,
                    model_class=self.form_model
                )
            except Exception as e2:
                return {
                    "success": False,
                    "errors": [str(e), str(e2)],
                    "raw_text": text
                }


# Simulated LLM for testing
# -----------------------

def simulate_llm_call(prompt: str) -> str:
    """
    Simulate an LLM call for testing purposes.
    
    Args:
        prompt: Prompt to send to the LLM
        
    Returns:
        Simulated LLM response
    """
    # This is a very simple simulation that just returns predefined responses
    # In a real application, this would call an actual LLM API
    
    if "person" in prompt.lower() and "john" in prompt.lower():
        return """
        ```json
        {
          "name": "John Doe",
          "age": 35,
          "occupation": "software engineer",
          "skills": ["Python", "JavaScript", "SQL"]
        }
        ```
        """
    
    if "contact form" in prompt.lower() and "sarah" in prompt.lower():
        return """
        {
          "full_name": "Sarah Johnson",
          "email": "sarah.johnson@example.com",
          "phone": "(555) 123-4567",
          "address": "123 Main St, Apt 4B, Boston, MA 02108",
          "inquiry_type": "support",
          "message": "I'd like to inquire about your premium support package. I'm specifically interested in the data export capabilities and API integration options."
        }
        """
    
    # Default response for unknown prompts
    return """
    {
      "error": "I don't have enough information to provide a structured response."
    }
    """


# Demonstration functions
# ---------------------

def demonstrate_basic_parsing():
    """Demonstrate basic parsing techniques."""
    # Simulate LLM output
    llm_output = """
    Based on the text, here's the information:
    
    ```json
    {
      "name": "John Doe",
      "age": 35,
      "occupation": "software engineer",
      "skills": ["Python", "JavaScript", "SQL"]
    }
    ```
    """
    
    # Parse with basic function
    try:
        data = parse_json_output(llm_output)
        print(f"Parsed JSON: {data}")
        
        # Parse into Pydantic model
        person = Person(**data)
        print(f"Parsed Person: {person}")
    except Exception as e:
        print(f"Parsing error: {e}")


def demonstrate_pydantic_parser():
    """Demonstrate PydanticOutputParser."""
    # Create parser
    parser = PydanticOutputParser(pydantic_object=Person)
    
    # Get format instructions
    instructions = parser.get_format_instructions()
    print(f"Format instructions:\n{instructions}\n")
    
    # Simulate LLM call with instructions
    prompt = f"""
    Extract information about the person from this text:
    
    John Doe is a 35-year-old software engineer who knows Python, JavaScript, and SQL.
    
    {instructions}
    """
    
    llm_output = simulate_llm_call(prompt)
    print(f"LLM output:\n{llm_output}\n")
    
    # Parse the output
    try:
        person = parser.parse(llm_output)
        print(f"Parsed Person: {person}")
    except Exception as e:
        print(f"Parsing error: {e}")


def demonstrate_form_extraction():
    """Demonstrate form extraction."""
    # Create form extractor
    extractor = FormExtractor(
        llm_call=simulate_llm_call,
        form_model=ContactForm
    )
    
    # Example text
    text = """
    Hello,
    
    My name is Sarah Johnson and I'd like to inquire about your premium support package. 
    I've been using your product for about 6 months and have some questions about advanced features.
    
    You can reach me at sarah.johnson@example.com or call me at (555) 123-4567.
    My address is 123 Main St, Apt 4B, Boston, MA 02108.
    
    I'm specifically interested in the data export capabilities and API integration options.
    Could someone from your technical team contact me to discuss these features in detail?
    
    Thanks,
    Sarah
    """
    
    # Extract form data
    result = extractor.extract(text)
    print(f"Extracted form data: {result}")


if __name__ == "__main__":
    print("=== Basic Parsing ===")
    demonstrate_basic_parsing()
    
    print("\n=== Pydantic Parser ===")
    demonstrate_pydantic_parser()
    
    print("\n=== Form Extraction ===")
    demonstrate_form_extraction()
