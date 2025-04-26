# 🚀 Module 3: Data Validation with Pydantic - Lesson 3: Structured Output Parsing 🔄

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand the challenges of working with unstructured LLM outputs
- 🧩 Implement various output parsing techniques using Pydantic
- 🔄 Create robust error handling and fallback strategies
- 📊 Build a practical form extraction agent with structured output
- 🛠️ Apply advanced parsing strategies for complex data scenarios

---

## 📚 Introduction to Structured Output Parsing

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

One of the most challenging aspects of working with Large Language Models (LLMs) is getting them to produce outputs in a consistent, structured format. In this lesson, we'll explore techniques for parsing and validating LLM outputs using Pydantic, ensuring that we can reliably extract structured data from natural language responses.

## 🧩 The Challenge of LLM Outputs

LLMs are trained to generate natural language, not structured data. This creates several challenges:

1. **Inconsistent formatting**: The model might format the same information differently each time
2. **Hallucinations**: The model might generate data that doesn't match your schema
3. **Missing information**: The model might omit required fields
4. **Extra information**: The model might include information you didn't ask for

Structured output parsing helps address these challenges by:

1. Providing clear instructions to the LLM about the expected output format
2. Validating the output against a predefined schema
3. Handling errors gracefully when the output doesn't match expectations

![Structured Output Parsing](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPrc2ngFZ6BTyww/giphy.gif)

## 🛠️ Basic Output Parsing Techniques

### 1. JSON Output Format

The simplest approach is to ask the LLM to format its response as JSON:

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt_template = """
Extract the following information from the text and return it as JSON:
- Name
- Age
- Occupation
- Skills (as a list)

Text: {text}

JSON Output:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

llm = OpenAI(temperature=0)
text = "John Doe is a 35-year-old software engineer who knows Python, JavaScript, and SQL."
result = llm.invoke(prompt.format(text=text))

print(result)
```

Example output:
```json
{
  "Name": "John Doe",
  "Age": 35,
  "Occupation": "software engineer",
  "Skills": ["Python", "JavaScript", "SQL"]
}
```

### 2. Parsing with Pydantic

Once you have JSON output, you can parse it with Pydantic:

```python
import json
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    Name: str
    Age: int
    Occupation: str
    Skills: List[str]

# Parse the LLM output
try:
    json_output = json.loads(result)
    person = Person(**json_output)
    print(f"Parsed person: {person}")
except Exception as e:
    print(f"Error parsing output: {e}")
```

### 3. Error Handling

LLMs don't always produce valid JSON. Handle parsing errors gracefully:

```python
def parse_llm_output(output_text, model_class):
    """Parse LLM output into a Pydantic model, handling common errors."""
    # Try to find JSON in the output
    try:
        # Look for JSON-like structure
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = output_text[start_idx:end_idx+1]
            data = json.loads(json_str)
            return model_class(**data)
    except json.JSONDecodeError:
        # Try to fix common JSON errors
        try:
            # Replace single quotes with double quotes
            fixed_json = output_text.replace("'", '"')
            data = json.loads(fixed_json)
            return model_class(**data)
        except:
            pass

    # If all parsing attempts fail
    raise ValueError(f"Could not parse LLM output as {model_class.__name__}")
```

## 🔄 LangChain Output Parsers

LangChain provides several output parsers that work well with Pydantic:

### 1. PydanticOutputParser

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Define your data model
class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's professional skills")

# Create a parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create a prompt template
prompt = PromptTemplate(
    template="Extract information from the text below:\n\n{text}\n\n{format_instructions}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Set up the LLM
llm = OpenAI(temperature=0)

# Process text
text = "John Doe is a 35-year-old software engineer who knows Python, JavaScript, and SQL."
_input = prompt.format_prompt(text=text)
output = llm.invoke(_input.to_string())

# Parse the output
try:
    person = parser.parse(output)
    print(person)
except Exception as e:
    print(f"Error parsing output: {e}")
```

### 2. StructuredOutputParser

For more flexibility in defining the output format:

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define the response schemas
response_schemas = [
    ResponseSchema(name="name", description="The person's full name"),
    ResponseSchema(name="age", description="The person's age in years"),
    ResponseSchema(name="occupation", description="The person's job or profession"),
    ResponseSchema(name="skills", description="Comma-separated list of the person's professional skills")
]

# Create a parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create a prompt template
prompt = PromptTemplate(
    template="Extract information from the text below:\n\n{text}\n\n{format_instructions}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Set up the LLM
llm = OpenAI(temperature=0)

# Process text
text = "John Doe is a 35-year-old software engineer who knows Python, JavaScript, and SQL."
_input = prompt.format_prompt(text=text)
output = llm.invoke(_input.to_string())

# Parse the output
try:
    parsed_output = parser.parse(output)
    print(parsed_output)

    # Convert to Pydantic model if needed
    from pydantic import BaseModel
    from typing import List

    class Person(BaseModel):
        name: str
        age: int
        occupation: str
        skills: List[str]

    # Convert skills from comma-separated string to list if needed
    if isinstance(parsed_output["skills"], str):
        parsed_output["skills"] = [s.strip() for s in parsed_output["skills"].split(",")]

    person = Person(**parsed_output)
    print(person)
except Exception as e:
    print(f"Error parsing output: {e}")
```

## 🧠 Advanced Parsing Strategies

### 1. Retry Logic for Failed Parsing

When parsing fails, you can retry with more explicit instructions:

```python
def parse_with_retry(llm, parser, text, max_retries=3):
    """Try to parse LLM output, retrying with more explicit instructions if it fails."""
    prompt_template = """
    Extract information from the text below:

    {text}

    {format_instructions}

    {retry_instructions}
    """

    retry_instructions = ""

    for attempt in range(max_retries):
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "retry_instructions": retry_instructions
            }
        )

        _input = prompt.format_prompt(text=text)
        output = llm.invoke(_input.to_string())

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
```

### 2. Two-Stage Parsing

For complex outputs, use a two-stage approach:

```python
def two_stage_parsing(llm, text):
    """
    First extract structured data, then validate with Pydantic.
    This approach gives the LLM more flexibility in the initial extraction.
    """
    # Stage 1: Extract information in a flexible format
    extraction_prompt = """
    Extract the following information from the text:
    - Name
    - Age
    - Occupation
    - Skills

    Text: {text}

    Provide the information in JSON format.
    """

    prompt = PromptTemplate(
        template=extraction_prompt,
        input_variables=["text"]
    )

    initial_output = llm.invoke(prompt.format(text=text))

    # Stage 2: Refine and validate the extracted information
    validation_prompt = """
    I've extracted the following information:

    {initial_output}

    Please format this as valid JSON with the following schema:
    {{
        "name": "string",
        "age": number,
        "occupation": "string",
        "skills": ["string", "string", ...]
    }}

    Ensure all fields are present and correctly typed.
    """

    prompt = PromptTemplate(
        template=validation_prompt,
        input_variables=["initial_output"]
    )

    refined_output = llm.invoke(prompt.format(initial_output=initial_output))

    # Parse with Pydantic
    try:
        # Try to extract JSON from the output
        import re
        import json

        json_match = re.search(r'```json\n(.*?)\n```', refined_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for JSON-like structure
            start_idx = refined_output.find('{')
            end_idx = refined_output.rfind('}')
            json_str = refined_output[start_idx:end_idx+1] if start_idx != -1 and end_idx != -1 else refined_output

        data = json.loads(json_str)

        # Validate with Pydantic
        from pydantic import BaseModel
        from typing import List

        class Person(BaseModel):
            name: str
            age: int
            occupation: str
            skills: List[str]

        return Person(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse refined output: {e}")
```

### 3. Function Calling for Structured Outputs

Modern LLMs like GPT-4 support function calling, which is ideal for structured outputs:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")
    skills: List[str] = Field(description="List of the person's professional skills")

# Define the function schema
function_schema = {
    "name": "extract_person_info",
    "description": "Extract information about a person from text",
    "parameters": Person.model_json_schema()
}

# Set up the model with function calling
model = ChatOpenAI(temperature=0, model="gpt-4").bind(
    functions=[function_schema],
    function_call={"name": "extract_person_info"}
)

# Process text
text = "John Doe is a 35-year-old software engineer who knows Python, JavaScript, and SQL."
message = HumanMessage(content=f"Extract person information from this text: {text}")
response = model.invoke([message])

# The response will contain the function call with structured data
function_call = response.additional_kwargs.get("function_call", {})
if function_call:
    import json
    arguments = json.loads(function_call.get("arguments", "{}"))
    person = Person(**arguments)
    print(person)
```

## 🧪 Handling Parsing Failures

Even with the best prompts, parsing can sometimes fail. Here's how to handle failures gracefully:

### 1. Fallback Strategies

```python
def parse_with_fallbacks(llm, text, parsers):
    """Try multiple parsing strategies in sequence."""
    errors = []

    for parser_name, parser_func in parsers.items():
        try:
            return parser_func(llm, text)
        except Exception as e:
            errors.append(f"{parser_name}: {str(e)}")

    # If all parsers fail, return a structured error
    return {
        "success": False,
        "errors": errors,
        "raw_text": text
    }

# Usage
parsers = {
    "pydantic_parser": lambda llm, text: parse_with_pydantic(llm, text),
    "structured_parser": lambda llm, text: parse_with_structured(llm, text),
    "function_calling": lambda llm, text: parse_with_function_calling(llm, text)
}

result = parse_with_fallbacks(llm, text, parsers)
```

### 2. Human-in-the-Loop

For critical applications, involve humans when parsing fails:

```python
def parse_with_human_fallback(llm, text, parser):
    """Try to parse automatically, but fall back to human review if needed."""
    try:
        return parser.parse(llm.invoke(text))
    except Exception as e:
        print(f"Automatic parsing failed: {e}")
        print(f"Original text: {text}")
        print(f"LLM output: {llm.invoke(text)}")

        # In a real application, this could be a UI prompt or a task queue
        human_input = input("Please correct the parsing issue or type 'skip' to ignore: ")

        if human_input.lower() == 'skip':
            return None
        else:
            # Assume the human provides valid JSON
            import json
            return json.loads(human_input)
```

## 🔍 Practical Example: Form Extraction Agent

Let's build a simple agent that extracts form data from unstructured text:

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re

# Define the form schema
class ContactForm(BaseModel):
    full_name: str = Field(description="The person's full name")
    email: str = Field(description="The person's email address")
    phone: Optional[str] = Field(None, description="The person's phone number")
    address: Optional[str] = Field(None, description="The person's physical address")
    inquiry_type: str = Field(description="The type of inquiry (e.g., support, sales, information)")
    message: str = Field(description="The content of the inquiry message")

    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator('phone')
    def validate_phone(cls, v):
        if v is not None and not re.match(r"^\+?[\d\s\-\(\)]+$", v):
            raise ValueError("Invalid phone number format")
        return v

# Create the parser
parser = PydanticOutputParser(pydantic_object=ContactForm)

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""
You are a form data extraction assistant. Extract the contact form information from the following text:

{text}

{format_instructions}

Make sure to extract all required fields. If a field is not present in the text, use your best judgment to determine if it should be left empty or inferred from context.
""")

# Set up the LLM
llm = ChatOpenAI(temperature=0)

# Create the extraction chain
from langchain.chains import LLMChain

extraction_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=parser
)

# Example usage
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

try:
    result = extraction_chain.invoke({"text": text, "format_instructions": parser.get_format_instructions()})
    print(result)
except Exception as e:
    print(f"Extraction failed: {e}")
    # Implement fallback strategy here
```

---

## 💪 Practice Exercises

1. **Create a Job Application Parser**:
   - Create a Pydantic model for a job application form with fields for personal information, education, work experience, and skills
   - Implement a PydanticOutputParser for your model
   - Test it with sample text containing job application information

2. **Implement Custom Validators**:
   - Add custom validators to ensure that dates are in the correct format
   - Validate email addresses and phone numbers
   - Implement a validator for education history to ensure chronological order

3. **Build a Retry Mechanism**:
   - Create a function that retries parsing with more specific instructions when it fails
   - Implement progressive guidance that provides more detailed format instructions on each retry
   - Test with deliberately malformed inputs

4. **Develop a Two-Stage Parser**:
   - Create a two-stage parsing approach for complex job applications with nested information
   - First stage extracts basic information, second stage refines and validates
   - Handle nested structures like education history and work experience

5. **Create a Fallback System**:
   - Implement multiple parsing strategies (JSON, structured, function calling)
   - Create a fallback system that tries each strategy in sequence
   - Add logging to track which strategy succeeded or why all failed

---

## 🔍 Key Concepts to Remember

1. **Structured Output Parsing**: Essential for reliable LLM-based data extraction and consistent results
2. **Pydantic Validation**: Provides powerful validation capabilities for ensuring LLM outputs match expected schemas
3. **Multiple Parsing Strategies**: Combining different approaches increases robustness and reliability
4. **Error Handling**: Fallback mechanisms and retry logic are crucial for production systems
5. **Function Calling**: When available, offers the most reliable structured output format for modern LLMs

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Advanced validation patterns for complex data scenarios
- Recursive validation for nested data structures
- Conditional validation based on field values
- Strategies for handling ambiguous or incomplete inputs
- Integration with agent systems for end-to-end workflows

---

## 📚 Resources

- [LangChain Output Parsers Documentation](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic Validation Documentation](https://docs.pydantic.dev/latest/usage/validators/)

---

## 🎯 Mini-Project Progress: Data Validation System

In this lesson, we've made progress on our data validation system by:
- Implementing structured output parsing for LLM responses
- Creating robust error handling mechanisms
- Building a practical form extraction agent
- Developing strategies for handling parsing failures

In the next lesson, we'll continue by:
- Expanding our validation capabilities for more complex scenarios
- Integrating our parsing system with the broader agent architecture
- Implementing advanced validation patterns for specialized use cases

---

> 💡 **Note on LLM Integration**: This lesson demonstrates integration with real LLMs for output parsing. The examples can be adapted to work with any LLM provider, including OpenAI, Anthropic, or open-source models.

---

Happy coding! 🚀
