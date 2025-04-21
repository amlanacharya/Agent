# 🧙‍♂️ Module 1: Agent Fundamentals - Lesson 2 📝

![Prompt Engineering](https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif)

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔮 Master the fundamentals of **prompt engineering**
- 📋 Create effective **prompt templates** for different agent tasks
- 🧩 Understand how to structure prompts for **consistent agent behavior**
- 🔄 Implement **prompt chaining** for complex reasoning

---

## 📚 Introduction to Prompt Engineering

![Magic Words](https://media.giphy.com/media/3o84U6421OOWegpQhq/giphy.gif)

Prompt engineering is the art and science of crafting inputs to AI models to elicit desired outputs. It's like learning to communicate with an alien intelligence that has its own way of understanding and responding to language.

> 💡 **Key Insight**: The quality of your agent's responses is directly tied to the quality of your prompts. Better prompts lead to better agent performance.

In the context of agentic systems, prompt engineering serves several critical functions:

1. **Guiding Agent Behavior**: Shaping how the agent interprets and responds to inputs
2. **Ensuring Consistency**: Maintaining reliable behavior across different interactions
3. **Enabling Complex Reasoning**: Breaking down complex tasks into manageable steps
4. **Defining Personality**: Establishing the agent's tone, style, and character

---

## 🧩 Prompt Engineering Fundamentals

### The Anatomy of an Effective Prompt

![Blueprint](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Effective prompts typically contain several key components:

| Component | Description | Example |
|-----------|-------------|---------|
| **Context** | Background information the agent needs | "You are a personal task manager assistant." |
| **Instructions** | Clear directions on what to do | "Create a new task with the following details:" |
| **Examples** | Demonstrations of expected behavior | "Input: 'Meeting tomorrow at 2pm' → Output: {task: 'Meeting', date: 'tomorrow', time: '2pm'}" |
| **Constraints** | Limitations or requirements | "Always confirm before deleting tasks." |
| **Output Format** | How the response should be structured | "Respond with a JSON object containing task details." |

### Basic Prompt Types

1. **Zero-shot Prompts**: No examples provided, just instructions
   ```
   Classify the following text as positive, negative, or neutral: "I love this product!"
   ```

2. **One-shot Prompts**: One example provided before the task
   ```
   Example: "The food was terrible." → Negative
   Classify the following text: "I love this product!"
   ```

3. **Few-shot Prompts**: Multiple examples provided
   ```
   Example 1: "The food was terrible." → Negative
   Example 2: "The service was okay." → Neutral
   Example 3: "I had an amazing time!" → Positive
   Classify the following text: "I love this product!"
   ```

---

## 🛠️ Implementing Prompt Templates

![Template](https://media.giphy.com/media/3oKIPtjEDHOVwzYrNS/giphy.gif)

Prompt templates allow you to create reusable prompt structures with placeholders for dynamic content. Let's implement a simple prompt template system for our agent:

```python
class PromptTemplate:
    def __init__(self, template_text):
        """Initialize with a template string containing {placeholders}"""
        self.template = template_text
    
    def format(self, **kwargs):
        """Fill in the template with provided values"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required placeholder: {e}")
    
    @classmethod
    def from_examples(cls, instructions, examples, input_placeholder="input"):
        """Create a few-shot prompt template from examples"""
        examples_text = "\n".join([f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}" 
                                  for i, ex in enumerate(examples)])
        template = f"{instructions}\n\n{examples_text}\n\nInput: {{{input_placeholder}}}\nOutput:"
        return cls(template)
```

### Using Prompt Templates in Your Agent

Let's enhance our agent from Lesson 1 to use prompt templates:

```python
from prompt_template import PromptTemplate

class PromptDrivenAgent:
    def __init__(self):
        self.state = {}
        
        # Define prompt templates
        self.greeting_template = PromptTemplate(
            "You are a helpful assistant named {assistant_name}. "
            "Greet the user named {user_name} in a {tone} tone."
        )
        
        self.task_parser_template = PromptTemplate.from_examples(
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
    
    def generate_greeting(self, user_name, tone="friendly"):
        """Generate a greeting using the template"""
        prompt = self.greeting_template.format(
            assistant_name="TaskBot",
            user_name=user_name,
            tone=tone
        )
        # In a real implementation, this would be sent to an LLM
        # For now, we'll simulate a response
        if tone == "friendly":
            return f"Hi {user_name}! How can I help you today?"
        elif tone == "formal":
            return f"Good day, {user_name}. How may I be of assistance?"
        else:
            return f"Hey {user_name}! What's up?"
    
    def parse_task(self, user_input):
        """Parse task details from user input using the template"""
        prompt = self.task_parser_template.format(input=user_input)
        # In a real implementation, this would be sent to an LLM
        # For now, we'll return a simulated response
        if "call" in user_input.lower():
            return "{'action': 'call', 'details': 'Extracted from input'}"
        elif "buy" in user_input.lower() or "purchase" in user_input.lower():
            return "{'action': 'buy', 'details': 'Extracted from input'}"
        else:
            return "{'action': 'unknown', 'raw_input': '" + user_input + "'}"
```

---

## 🧠 Advanced Prompt Engineering Techniques

### Chain-of-Thought Prompting

![Thinking Chain](https://media.giphy.com/media/3o7TKT6gL5B7Lzq3re/giphy.gif)

Chain-of-thought prompting encourages the agent to break down complex reasoning into steps:

```
To solve this problem, let's think step by step:

1. First, I need to understand what the user is asking for.
2. Then, I need to identify the key information needed.
3. Next, I'll process that information according to the rules.
4. Finally, I'll formulate a clear response.

User query: "I need to schedule a meeting with the marketing team next Tuesday at 2pm and send a reminder to everyone."
```

### Role-Based Prompting

Assigning a specific role to the agent can shape its behavior and responses:

```
You are an expert project manager with 15 years of experience in agile methodologies. 
Your communication style is concise, clear, and focused on actionable insights.
Your goal is to help the user organize their tasks effectively.

User query: "I have too many tasks and don't know where to start."
```

### Self-Reflection Prompting

Encouraging the agent to evaluate its own responses:

```
Answer the following question, then reflect on your answer to check for errors or ways to improve it.

Question: "What are the key factors to consider when implementing a microservice architecture?"
```

---

## 🛠️ Building a Prompt Library

![Library](https://media.giphy.com/media/3o85xGocUH8RYoDKKs/giphy.gif)

For a sophisticated agent, it's helpful to build a library of prompt templates for different tasks:

```python
class PromptLibrary:
    def __init__(self):
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
        """Retrieve a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found in library")
        return self.templates[template_name]
    
    def add_template(self, name, template):
        """Add a new template to the library"""
        self.templates[name] = template
        
    def format_prompt(self, template_name, **kwargs):
        """Format a specific template with provided values"""
        template = self.get_template(template_name)
        return template.format(**kwargs)
```

---

## 💪 Practice Exercises

![Practice](https://media.giphy.com/media/3oKIPrc2ngFZ6BTyww/giphy.gif)

1. **Create a Task Parser Template**:
   - Design a prompt template that extracts task details from natural language input
   - Include examples for different types of tasks (meetings, deadlines, reminders)
   - Test your template with various inputs

2. **Implement Role-Based Prompts**:
   - Create templates for different assistant personalities (professional, friendly, technical)
   - Implement a method to switch between roles based on user preferences
   - Test how the same query gets different responses with different roles

3. **Chain-of-Thought Implementation**:
   - Design a template that breaks down complex task planning into steps
   - Implement a method that uses this template to generate step-by-step plans
   - Test with a complex scenario like planning a project or event

---

## 🔍 Key Concepts to Remember

![Key Concepts](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

1. **Context Matters**: Always provide sufficient context in your prompts
2. **Be Specific**: Clear, specific instructions yield better results
3. **Examples Help**: Few-shot prompting often produces more consistent outputs
4. **Format Matters**: The structure and organization of your prompt affects the response
5. **Iterative Refinement**: Prompt engineering is an iterative process - test and refine

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- State management patterns for agents
- Implementing conversation memory
- Tracking and updating agent state
- Building a more sophisticated task manager

---

## 📚 Resources

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
- [OpenAI Prompt Engineering Best Practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

---

## 🎯 Mini-Project Progress: Personal Task Manager

![Task Manager](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

In this lesson, we've learned how to create prompt templates that will help our Personal Task Manager:
- Parse natural language task descriptions
- Extract key details like dates, times, and priorities
- Respond consistently to user queries
- Handle different types of task-related requests

In the next lesson, we'll implement state management to actually store and track these tasks!

---

Happy prompting! 🚀
