# 🧩 Module 1: Code Examples

![Code Banner](https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif)

## 📚 Overview

This directory contains all the code examples and implementations for Module 1: Agent Fundamentals. These examples demonstrate the core concepts of the sense-think-act loop, prompt engineering, and state management patterns.

## 🔍 File Descriptions

### Core Agent Implementations
- **simple_agent.py**: Basic implementation of the sense-think-act loop
- **prompt_driven_agent.py**: Agent that uses prompt templates for more sophisticated responses
- **stateful_agent.py**: Agent with state management capabilities
- **task_manager_agent.py**: Complete implementation of the personal task manager

### Supporting Components
- **prompt_template.py**: Implementation of prompt templates and a prompt library
- **state_management.py**: Implementation of different memory and state management patterns

### Test Scripts
- **test_simple_agent.py**: Tests for the basic agent
- **test_prompt_driven_agent.py**: Tests for the prompt-driven agent
- **test_state_management.py**: Tests for the state management components
- **test_task_manager.py**: Tests for the task manager agent

## 🚀 Running the Examples

You can run any of the examples directly from the command line:

```bash
# Run from the project root
python -m module1.code.simple_agent
python -m module1.code.prompt_driven_agent
python -m module1.code.stateful_agent
python -m module1.code.task_manager_agent

# Run the tests
python -m module1.code.test_simple_agent
python -m module1.code.test_prompt_driven_agent
python -m module1.code.test_state_management
python -m module1.code.test_task_manager
```

## 📝 Notes

- These implementations use simulated responses instead of actual LLM calls
- The focus is on architecture and patterns, not on LLM integration
- Later modules will cover integration with actual LLMs
