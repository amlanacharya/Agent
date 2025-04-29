"""
Example script demonstrating how to use the Groq tool.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the Groq tool
from module6.code.tools.groq_tool import GroqTool

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{text.center(80)}")
    print("="*80)

def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-"*80)
    print(f"{text}")
    print("-"*80)

def main():
    """Main function to demonstrate the Groq tool."""
    print_header("Groq Tool Demo")
    
    # Create the tool
    groq_tool = GroqTool(
        model="llama3-8b-8192",
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"Initialized Groq tool with model: {groq_tool.model}")
    
    # Example 1: Simple text completion
    print_section("Example 1: Simple Text Completion")
    prompt = "Write a short poem about artificial intelligence."
    print(f"Prompt: {prompt}")
    
    response = groq_tool.complete(prompt)
    print("\nResponse:")
    print(response)
    
    # Example 2: Chat completion
    print_section("Example 2: Chat Completion")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "Tell me about the weather today."}
    ]
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    response = groq_tool.chat(messages)
    print("\nResponse:")
    print(response)
    
    # Example 3: Using different parameters
    print_section("Example 3: Using Different Parameters")
    prompt = "List 3 benefits of exercise."
    print(f"Prompt: {prompt}")
    print("Parameters: temperature=0.2, max_tokens=50")
    
    response = groq_tool.complete(prompt, temperature=0.2, max_tokens=50)
    print("\nResponse:")
    print(response)
    
    # Example 4: JSON generation
    print_section("Example 4: JSON Generation")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age", "occupation", "skills"]
    }
    
    prompt = "Extract information about John Doe, a 35-year-old software engineer who knows Python, JavaScript, and SQL."
    print(f"Prompt: {prompt}")
    print("Schema:", json.dumps(schema, indent=2))
    
    response = groq_tool.generate_json(prompt, schema)
    print("\nResponse:")
    print(json.dumps(response, indent=2))
    
    # Example 5: Using the execute method directly
    print_section("Example 5: Using the execute Method Directly")
    prompt = "What are some good books to read?"
    print(f"Prompt: {prompt}")
    
    response = groq_tool.execute(prompt=prompt)
    print("\nSuccess:", response.success)
    print("Result:", response.result)
    print("Metadata:", response.metadata)
    
    print_header("End of Demo")

if __name__ == "__main__":
    main()
