"""
Example script demonstrating how to use the OpenAI tool.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the OpenAI tool
from module6.code.tools.openai_tool import OpenAITool

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
    """Main function to demonstrate the OpenAI tool."""
    print_header("OpenAI Tool Demo")
    
    # Create the tool with a cheaper model
    openai_tool = OpenAITool(
        model="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"Initialized OpenAI tool with model: {openai_tool.model}")
    
    # Example 1: Simple text completion
    print_section("Example 1: Simple Text Completion")
    prompt = "Write a short poem about artificial intelligence."
    print(f"Prompt: {prompt}")
    
    response = openai_tool.complete(prompt)
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
    
    response = openai_tool.chat(messages)
    print("\nResponse:")
    print(response)
    
    # Example 3: Using different parameters
    print_section("Example 3: Using Different Parameters")
    prompt = "List 3 benefits of exercise."
    print(f"Prompt: {prompt}")
    print("Parameters: temperature=0.2, max_tokens=50")
    
    response = openai_tool.complete(prompt, temperature=0.2, max_tokens=50)
    print("\nResponse:")
    print(response)
    
    # Example 4: Using the execute method directly
    print_section("Example 4: Using the execute Method Directly")
    prompt = "What are some good books to read?"
    print(f"Prompt: {prompt}")
    
    response = openai_tool.execute(prompt=prompt)
    print("\nSuccess:", response.success)
    print("Result:", response.result)
    print("Metadata:", response.metadata)
    
    print_header("End of Demo")

if __name__ == "__main__":
    main()
