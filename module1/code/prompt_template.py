"""
Prompt Template Implementation
----------------------------
This file contains a basic implementation of prompt templates for AI agents.
Use this as a starting point for the exercises in Module 1, Lesson 2.
"""

class PromptTemplate:
    def __init__(self, template_text):
        """
        Initialize with a template string containing {placeholders}
        
        Args:
            template_text (str): The template string with {placeholders}
        """
        self.template = template_text
    
    def format(self, **kwargs):
        """
        Fill in the template with provided values
        
        Args:
            **kwargs: Key-value pairs for placeholders in the template
            
        Returns:
            str: The formatted prompt
            
        Raises:
            ValueError: If a required placeholder is missing
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required placeholder: {e}")
    
    @classmethod
    def from_examples(cls, instructions, examples, input_placeholder="input"):
        """
        Create a few-shot prompt template from examples
        
        Args:
            instructions (str): The task instructions
            examples (list): List of dictionaries with 'input' and 'output' keys
            input_placeholder (str): The name of the input placeholder
            
        Returns:
            PromptTemplate: A new template with examples included
        """
        examples_text = "\n".join([f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}" 
                                  for i, ex in enumerate(examples)])
        template = f"{instructions}\n\n{examples_text}\n\nInput: {{{input_placeholder}}}\nOutput:"
        return cls(template)


class PromptLibrary:
    def __init__(self):
        """Initialize with a default set of prompt templates"""
        self.templates = {
            "greeting": PromptTemplate(
                "You are a {assistant_type} assistant named {assistant_name}. "
                "Greet the user named {user_name} in a {tone} tone."
            ),
            
            "task_creation": PromptTemplate(
                "Create a new task with the following details:\n"
                "Description: {description}\n"
                "Due date: {due_date}\n"
                "Priority: {priority}\n"
                "Return a confirmation message in a {tone} tone."
            ),
            
            "task_query": PromptTemplate(
                "The user wants to know about their tasks. "
                "Current tasks in the system: {task_list}\n"
                "User query: {query}\n"
                "Provide a helpful response addressing their query."
            ),
            
            "error_handling": PromptTemplate(
                "The user request resulted in an error: {error_message}\n"
                "Explain the issue to the user in a {tone} tone and provide suggestions to resolve it."
            )
        }
    
    def get_template(self, template_name):
        """
        Retrieve a template by name
        
        Args:
            template_name (str): The name of the template to retrieve
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            ValueError: If the template doesn't exist
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found in library")
        return self.templates[template_name]
    
    def add_template(self, name, template):
        """
        Add a new template to the library
        
        Args:
            name (str): The name for the new template
            template (PromptTemplate): The template to add
        """
        self.templates[name] = template
        
    def format_prompt(self, template_name, **kwargs):
        """
        Format a specific template with provided values
        
        Args:
            template_name (str): The name of the template to format
            **kwargs: Key-value pairs for placeholders in the template
            
        Returns:
            str: The formatted prompt
        """
        template = self.get_template(template_name)
        return template.format(**kwargs)


# Example usage
if __name__ == "__main__":
    # Basic template usage
    greeting_template = PromptTemplate("Hello, {name}! How are you {feeling} today?")
    formatted = greeting_template.format(name="Alice", feeling="excited")
    print("Basic template example:")
    print(formatted)
    print("-" * 50)
    
    # Few-shot template example
    task_parser = PromptTemplate.from_examples(
        instructions="Extract task details from the user input.",
        examples=[
            {
                "input": "I need to call John tomorrow at 3pm",
                "output": "{'action': 'call', 'person': 'John', 'date': 'tomorrow', 'time': '3pm'}"
            },
            {
                "input": "Remind me to buy groceries on Friday",
                "output": "{'action': 'buy', 'item': 'groceries', 'date': 'Friday'}"
            }
        ]
    )
    formatted = task_parser.format(input="I have a meeting with Sarah on Monday at 10am")
    print("Few-shot template example:")
    print(formatted)
    print("-" * 50)
    
    # Prompt library example
    library = PromptLibrary()
    greeting = library.format_prompt(
        "greeting", 
        assistant_type="task management",
        assistant_name="TaskBot",
        user_name="Bob",
        tone="friendly"
    )
    print("Prompt library example:")
    print(greeting)
    print("-" * 50)
