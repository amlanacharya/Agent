"""
Standalone Demonstration Script for Structured Output Parsing
---------------------------------------------------------
This script demonstrates techniques for parsing and validating LLM outputs using Pydantic.
"""

import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# 1. Basic Pydantic Models
# ----------------------

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


# 2. Output Parsers
# ---------------

class PydanticOutputParser:
    """Parser for converting LLM outputs to Pydantic models."""
    
    def __init__(self, pydantic_object: type):
        """Initialize the parser with a Pydantic model class."""
        self.pydantic_object = pydantic_object
    
    def parse(self, text: str) -> BaseModel:
        """Parse text into the Pydantic model."""
        try:
            # Try to extract JSON from the text
            json_data = self._extract_json(text)
            
            # Validate with Pydantic model
            return self.pydantic_object(**json_data)
        except Exception as e:
            raise ValueError(f"Could not parse text as {self.pydantic_object.__name__}: {str(e)}")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except:
                pass
        
        # Try to find JSON-like structure
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        # Try to fix common JSON errors
        try:
            # Replace single quotes with double quotes
            fixed_json = text.replace("'", '"')
            start_idx = fixed_json.find('{')
            end_idx = fixed_json.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = fixed_json[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        # If all parsing attempts fail
        raise ValueError("Could not extract JSON from text")
    
    def get_format_instructions(self) -> str:
        """Get instructions for the LLM on how to format its output."""
        schema = self.pydantic_object.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        return f"""
        You must format your output as a JSON instance that conforms to the JSON schema below.

        {schema_str}
        
        The JSON output should contain only the required fields and should be valid JSON.
        """


# 3. Simulated LLM Responses
# ------------------------

def simulate_llm_response(prompt: str) -> str:
    """Simulate an LLM response for demonstration purposes."""
    if "person" in prompt.lower():
        if "json" in prompt.lower():
            # Clean JSON response
            return """
            {
              "name": "John Doe",
              "age": 35,
              "occupation": "software engineer",
              "skills": ["Python", "JavaScript", "SQL"]
            }
            """
        else:
            # Messy response with JSON embedded
            return """
            Based on the text, here's the information about the person:
            
            ```json
            {
              "name": "John Doe",
              "age": 35,
              "occupation": "software engineer",
              "skills": ["Python", "JavaScript", "SQL"]
            }
            ```
            
            Let me know if you need any clarification!
            """
    
    elif "contact form" in prompt.lower():
        if "json" in prompt.lower():
            # Clean JSON response
            return """
            {
              "full_name": "Sarah Johnson",
              "email": "sarah.johnson@example.com",
              "phone": "(555) 123-4567",
              "address": "123 Main St, Apt 4B, Boston, MA 02108",
              "inquiry_type": "support",
              "message": "I'd like to inquire about your premium support package."
            }
            """
        else:
            # Response with single quotes instead of double quotes
            return """
            I've extracted the contact form information:
            
            {
              'full_name': 'Sarah Johnson',
              'email': 'sarah.johnson@example.com',
              'phone': '(555) 123-4567',
              'address': '123 Main St, Apt 4B, Boston, MA 02108',
              'inquiry_type': 'support',
              'message': 'I\'d like to inquire about your premium support package.'
            }
            """
    
    elif "invalid" in prompt.lower():
        if "missing field" in prompt.lower():
            # Missing required field
            return """
            {
              "name": "John Doe",
              "age": 35,
              "occupation": "software engineer"
              // Missing skills field
            }
            """
        elif "wrong type" in prompt.lower():
            # Wrong type for a field
            return """
            {
              "name": "John Doe",
              "age": "thirty-five",
              "occupation": "software engineer",
              "skills": ["Python", "JavaScript", "SQL"]
            }
            """
        else:
            # Not JSON at all
            return "The person's name is John Doe, age 35, and works as a software engineer."
    
    else:
        # Default response
        return "I'm not sure how to respond to that prompt."


# 4. Retry Mechanism
# ----------------

def parse_with_retry(prompt: str, parser: PydanticOutputParser, max_retries: int = 3) -> BaseModel:
    """Try to parse LLM output, retrying with more explicit instructions if it fails."""
    retry_instructions = ""
    
    for attempt in range(max_retries):
        # Format the prompt with retry instructions
        full_prompt = f"""
        {prompt}
        
        {parser.get_format_instructions()}
        
        {retry_instructions}
        """
        
        # Get LLM response
        response = simulate_llm_response(full_prompt)
        print(f"Attempt {attempt + 1} response: {response[:100]}...\n")
        
        try:
            return parser.parse(response)
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
                print(f"Parsing failed: {e}")
                print("Retrying with more explicit instructions...\n")
            else:
                raise ValueError(f"Failed to parse output after {max_retries} attempts: {e}")


# 5. Two-Stage Parsing
# ------------------

def two_stage_parsing(prompt: str) -> BaseModel:
    """First extract structured data, then validate with Pydantic."""
    # Stage 1: Extract information in a flexible format
    extraction_prompt = f"""
    {prompt}
    
    Provide the information in JSON format.
    """
    
    initial_output = simulate_llm_response(extraction_prompt)
    print(f"Stage 1 output: {initial_output[:100]}...\n")
    
    # Stage 2: Refine and validate the extracted information
    validation_prompt = f"""
    I've extracted the following information:
    
    {initial_output}
    
    Please format this as valid JSON that conforms to the Person schema with fields:
    - name (string)
    - age (number)
    - occupation (string)
    - skills (array of strings)
    
    Ensure all fields are present and correctly typed.
    """
    
    refined_output = simulate_llm_response(validation_prompt + " json")
    print(f"Stage 2 output: {refined_output[:100]}...\n")
    
    # Parse with Pydantic
    parser = PydanticOutputParser(pydantic_object=Person)
    return parser.parse(refined_output)


def main():
    """Main demonstration function."""
    print("=== Structured Output Parsing Demonstration ===")
    
    # 1. Basic Parsing
    print("\n1. Basic Parsing")
    print("---------------")
    
    parser = PydanticOutputParser(pydantic_object=Person)
    
    print("Parsing clean JSON response:")
    try:
        clean_response = simulate_llm_response("Extract information about a person json")
        person = parser.parse(clean_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    print("\nParsing messy response with embedded JSON:")
    try:
        messy_response = simulate_llm_response("Extract information about a person")
        person = parser.parse(messy_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    print("\nParsing response with single quotes:")
    try:
        single_quotes_response = simulate_llm_response("Extract contact form information")
        contact_parser = PydanticOutputParser(pydantic_object=ContactForm)
        contact = contact_parser.parse(single_quotes_response)
        print(f"Successfully parsed: {contact}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    # 2. Handling Invalid Responses
    print("\n2. Handling Invalid Responses")
    print("---------------------------")
    
    print("Parsing response with missing field:")
    try:
        missing_field_response = simulate_llm_response("Extract information about a person invalid missing field")
        person = parser.parse(missing_field_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    print("\nParsing response with wrong type:")
    try:
        wrong_type_response = simulate_llm_response("Extract information about a person invalid wrong type")
        person = parser.parse(wrong_type_response)
        print(f"Successfully parsed: {person}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    # 3. Retry Mechanism
    print("\n3. Retry Mechanism")
    print("----------------")
    
    print("Parsing with retry:")
    try:
        # This will fail on the first attempt but succeed on the second
        person = parse_with_retry(
            prompt="Extract information about a person",
            parser=parser,
            max_retries=3
        )
        print(f"Successfully parsed after retry: {person}")
    except Exception as e:
        print(f"Parsing failed after retries: {e}")
    
    # 4. Two-Stage Parsing
    print("\n4. Two-Stage Parsing")
    print("------------------")
    
    print("Two-stage parsing:")
    try:
        person = two_stage_parsing("Extract information about a person named John Doe who is 35 years old")
        print(f"Successfully parsed with two-stage approach: {person}")
    except Exception as e:
        print(f"Two-stage parsing failed: {e}")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
