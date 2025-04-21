# 🤖 Module 1: Agent Fundamentals

![Agent Banner](https://media.giphy.com/media/3o7TKsQ8Xb3gcGEgZW/giphy.gif)

## 📚 Overview

Welcome to Module 1 of the Accelerated Agentic AI Mastery course! This module covers the fundamental concepts of AI agents, focusing on the core agent loop (sense-think-act), prompt engineering basics, and state management patterns.

> 💡 **Note**: Module 1 focuses on the architecture and patterns of agentic systems without requiring integration with actual Large Language Models (LLMs). We use simulated responses where an LLM would typically be used, allowing you to learn the fundamentals without dealing with API keys, rate limits, or costs. Later modules will cover actual LLM integration.

## 📂 Module Structure

```
module1/
├── lessons/                  # Lesson content in markdown format
│   ├── lesson1.md            # Lesson 1: The Sense-Think-Act Loop
│   ├── lesson2.md            # Lesson 2: Prompt Engineering Fundamentals
│   ├── lesson3.md            # Lesson 3: State Management Patterns
│   └── lesson4.md            # Lesson 4: Building the Personal Task Manager
├── code/                     # Code examples and implementations
│   ├── simple_agent.py       # Basic agent implementation
│   ├── test_simple_agent.py  # Test script for the simple agent
│   ├── prompt_template.py    # Prompt template implementation
│   ├── prompt_driven_agent.py # Agent using prompt templates
│   ├── state_management.py   # State management implementations
│   ├── stateful_agent.py     # Agent with state management
│   ├── task_manager_agent.py # Complete task manager implementation
│   └── test_task_manager.py  # Test script for the task manager
└── exercises/               # Practice exercises and solutions
    ├── exercise_solutions.py # Solutions for lesson 1 exercises
    ├── prompt_exercises.py   # Solutions for lesson 2 exercises
    └── state_exercises.py    # Solutions for lesson 3 exercises
```

## 🚀 Getting Started

1. Start by reading through the lessons in order:
   - **lessons/lesson1.md**: The Sense-Think-Act Loop
   - **lessons/lesson2.md**: Prompt Engineering Fundamentals
   - **lessons/lesson3.md**: State Management Patterns
   - **lessons/lesson4.md**: Building the Personal Task Manager

2. Examine the code examples for each lesson:
   - Lesson 1: **code/simple_agent.py**
   - Lesson 2: **code/prompt_template.py** and **code/prompt_driven_agent.py**
   - Lesson 3: **code/state_management.py** and **code/stateful_agent.py**
   - Lesson 4: **code/task_manager_agent.py**

3. Run the test scripts to see the agents in action:
   ```
   python module1/code/test_simple_agent.py
   python module1/code/test_prompt_agent.py
   python module1/code/test_state_management.py
   python module1/code/test_task_manager.py
   ```

4. Try the interactive demos by running:
   ```
   python module1/code/simple_agent.py
   python module1/code/prompt_driven_agent.py
   python module1/code/stateful_agent.py
   python module1/code/task_manager_agent.py
   ```

5. Complete the practice exercises at the end of each lesson

6. Check your solutions against the exercise files (but try to solve them yourself first!):
   - Lesson 1: **exercises/exercise_solutions.py**
   - Lesson 2: **exercises/prompt_exercises.py**
   - Lesson 3: **exercises/state_exercises.py**

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the core agent loop (sense-think-act)
- Master prompt engineering fundamentals
- Learn basic state management patterns
- Build a simple personal task manager agent

## 🧪 Practice Exercises

The lesson includes several practice exercises to help reinforce your learning:
1. Extending the SimpleAgent with new capabilities
2. Implementing conversation history tracking
3. Enhancing input processing with basic intent recognition

## 📝 Mini-Project: Personal Task Manager

Throughout this module, you'll be building a Personal Task Manager agent that can:
- Accept natural language commands to create, update, and delete tasks
- Store tasks with priority levels and due dates
- Respond to queries about task status
- Provide daily summaries of pending tasks

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Python Documentation](https://docs.python.org/3/)

## 🤔 Need Help?

If you get stuck or have questions:
- Review the lesson material again
- Check the example solutions
- Experiment with different approaches
- Discuss with fellow students

Happy learning! 🚀
