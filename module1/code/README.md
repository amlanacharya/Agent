# 🧩 Module 1: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 1: Agent Fundamentals. These examples demonstrate the core concepts of the sense-think-act loop, prompt engineering, and state management patterns.

## 🔍 File Descriptions

### Core Implementations
- **simple_agent.py**: Basic implementation of the sense-think-act loop with minimal components
- **prompt_template.py**: Implementation of prompt templates and a prompt library for structured agent interactions
- **prompt_driven_agent.py**: Agent that uses prompt templates for more sophisticated responses
- **state_management.py**: Implementation of different memory and state management patterns
- **stateful_agent.py**: Agent with state management capabilities for maintaining context
- **task_manager_agent.py**: Complete implementation of the personal task manager

### Test Scripts
- **test_simple_agent.py**: Tests for the basic agent implementation
- **test_prompt_driven_agent.py**: Tests for the prompt-driven agent
- **test_state_management.py**: Tests for the state management components
- **test_task_manager.py**: Tests for the task manager agent

### Demo Scripts
- **../demo_agents.py**: Interactive demonstration for all agent implementations with a menu interface

## 🚀 Running the Examples

You can run any of the examples directly from the command line:

```bash
# Run from the project root
python -m module1.code.simple_agent
python -m module1.code.prompt_driven_agent
python -m module1.code.stateful_agent
python -m module1.code.task_manager_agent
```

To run the tests:

```bash
# Run from the project root
python -m module1.code.test_simple_agent
python -m module1.code.test_prompt_driven_agent
python -m module1.code.test_state_management
python -m module1.code.test_task_manager
```

To run the interactive demos:

```bash
# Run from the project root
python -m module1.demo_agents

# Or run a specific agent demo directly
python -m module1.demo_agents simple    # Simple Agent
python -m module1.demo_agents prompt    # Prompt-Driven Agent
python -m module1.demo_agents stateful  # Stateful Agent
python -m module1.demo_agents task      # Task Manager Agent
```

## 📋 Implementation Notes

- The agents follow a consistent architecture based on the sense-think-act loop
- Each agent implementation builds on the previous one, adding new capabilities
- The state management patterns include conversation history, working memory, and task storage
- The task manager uses a combination of prompt engineering and state management techniques
- All implementations are designed to be modular and extensible

## 🔄 LLM Integration

> 💡 **Note**: Module 1 uses simulated LLM responses rather than integrating with real LLMs. This allows you to learn the fundamentals without dealing with API keys, rate limits, or costs.

## 🧪 Example Usage

Here's a simple example of how to use the SimpleAgent:

```python
# Example code snippet showing basic usage
from module1.code.simple_agent import SimpleAgent

# Create an instance
agent = SimpleAgent()

# Use the agent
response = agent.process_input("Hello, agent!")
print(response)
```

And here's how to use the TaskManagerAgent:

```python
# Example code snippet showing task manager usage
from module1.code.task_manager_agent import TaskManagerAgent

# Create an instance
task_manager = TaskManagerAgent()

# Add a task
response = task_manager.process_input("Add a task: Complete the project report by Friday with high priority")
print(response)

# Query tasks
response = task_manager.process_input("What are my high priority tasks?")
print(response)
```

## 🛠️ Extending the Code

Here are some ideas for extending or customizing the implementations:

1. Add more sophisticated intent recognition to the agents
2. Implement additional state management patterns
3. Create specialized agents for different domains
4. Add persistence to the task manager to save tasks between sessions
5. Implement a graphical user interface for the agents

## 📚 Related Resources

- [Python Documentation on Classes](https://docs.python.org/3/tutorial/classes.html)
- [Design Patterns in Python](https://refactoring.guru/design-patterns/python)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
