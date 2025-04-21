# 🚀 Module 1: Agent Fundamentals - Lesson 4 📋

![Task Manager](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🏗️ Build a complete **Personal Task Manager** agent
- 🔄 Integrate the **sense-think-act** loop with **prompt templates** and **state management**
- 🧪 Implement **testing and validation** for your agent
- 🚀 Learn how to **extend and enhance** your agent with additional features

---

## 📚 Introduction to the Mini-Project

![Project Planning](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

In this final lesson of Module 1, we'll bring together everything we've learned to build a complete Personal Task Manager agent. This project will demonstrate how the core concepts of agent development work together in a real-world application.

> 💡 **Key Insight**: Building a complete agent requires careful integration of all components. The way these components interact is just as important as the components themselves.

Our Personal Task Manager will be able to:
- Accept natural language commands to create, update, and delete tasks
- Store tasks with priority levels and due dates
- Respond to queries about task status
- Provide daily summaries of pending tasks
- Remember user preferences for task organization

---

## 🏗️ Project Architecture

![Architecture](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

Before diving into implementation, let's understand the architecture of our task manager agent:

### System Components

```
TaskManagerAgent
│
├── Core Agent Loop
│   ├── Sense: Parse natural language input
│   ├── Think: Determine actions based on intent
│   └── Act: Execute actions and generate responses
│
├── Prompt Templates
│   ├── Task parsing templates
│   ├── Response generation templates
│   └── Query handling templates
│
├── State Management
│   ├── Task storage and retrieval
│   ├── User preferences
│   └── Conversation history
│
└── Utilities
    ├── Date and time handling
    ├── Natural language processing
    └── Validation and error handling
```

### Data Flow

1. User provides natural language input
2. Input is processed by the sense function
3. Intent is determined and relevant data extracted
4. State is updated based on the intent
5. Response is generated using appropriate templates
6. Response is returned to the user

---

## 🛠️ Implementation Steps

We'll build our task manager in stages, focusing on one component at a time:

1. **Core Agent Structure**: Set up the basic agent framework
2. **Task Management**: Implement task CRUD operations
3. **Query Handling**: Add the ability to query tasks
4. **User Preferences**: Implement preference management
5. **Conversation Context**: Add memory of previous interactions
6. **Testing and Validation**: Ensure everything works correctly

Let's get started!

### Step 1: Core Agent Structure

![Building Blocks](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

First, we'll set up the basic structure of our TaskManagerAgent class, incorporating the sense-think-act loop:

```python
class TaskManagerAgent:
    def __init__(self, storage_dir="task_manager_data"):
        """Initialize the task manager agent"""
        # Set up state management
        self.state_manager = AgentStateManager(storage_dir=storage_dir)

        # Set up prompt templates
        self.prompt_library = self._initialize_prompt_library()

        # Internal state for reasoning
        self.internal_state = {
            "current_intent": None,
            "reasoning_steps": [],
            "last_action": None
        }

    def sense(self, user_input):
        """Process user input to extract intent and entities"""
        # Update conversation history
        self.state_manager.update_conversation("user", user_input)

        # Detect intent
        intent = self._detect_intent(user_input)

        # Extract entities based on intent
        entities = self._extract_entities(user_input, intent)

        # Return processed input
        return {
            "text": user_input,
            "intent": intent,
            "entities": entities,
            "timestamp": time.time()
        }

    def think(self, processed_input):
        """Determine actions based on intent and entities"""
        # Update internal state
        self.internal_state["current_intent"] = processed_input["intent"]
        self.internal_state["reasoning_steps"] = []

        # Add initial reasoning step
        self._add_reasoning_step(f"Detected intent: {processed_input['intent']}")

        # Handle different intents
        if processed_input["intent"] == "create_task":
            return self._handle_create_task(processed_input)
        elif processed_input["intent"] == "update_task":
            return self._handle_update_task(processed_input)
        elif processed_input["intent"] == "delete_task":
            return self._handle_delete_task(processed_input)
        elif processed_input["intent"] == "query_tasks":
            return self._handle_query_tasks(processed_input)
        elif processed_input["intent"] == "set_preference":
            return self._handle_set_preference(processed_input)
        else:
            # Default handling for unknown intents
            return self._handle_unknown_intent(processed_input)

    def act(self, action_plan):
        """Execute the action plan and generate a response"""
        # Update conversation with agent's response
        self.state_manager.update_conversation("agent", action_plan["response"])

        # Save state
        self.state_manager.save_state()

        # Update internal state
        self.internal_state["last_action"] = action_plan["action_type"]

        # Return the response
        return action_plan["response"]

    def agent_loop(self, user_input):
        """Run the full sense-think-act cycle"""
        processed_input = self.sense(user_input)
        action_plan = self.think(processed_input)
        response = self.act(action_plan)
        return response

    # Helper methods will be implemented in the next steps
```

### Step 2: Task Management

![Task Management](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Next, we'll implement the methods for creating, updating, and deleting tasks:

```python
def _handle_create_task(self, processed_input):
    """Handle task creation intent"""
    entities = processed_input["entities"]

    # Extract task details
    task = {
        "description": entities.get("description", processed_input["text"]),
        "priority": entities.get("priority", "medium"),
        "status": "pending",
        "created_at": time.time()
    }

    # Add due date if provided
    if "due_date" in entities:
        task["due_date"] = entities["due_date"]

    # Add tags if provided
    if "tags" in entities:
        task["tags"] = entities["tags"]

    # Add task to state
    task_id = self.state_manager.add_task(task)

    # Add reasoning step
    self._add_reasoning_step(f"Created task with ID {task_id}: {task['description']}")

    # Generate response
    response = f"I've added your task: {task['description']}"
    if "due_date" in task:
        response += f" (due: {task['due_date']})"

    return {
        "action_type": "create_task",
        "task_id": task_id,
        "response": response
    }

def _handle_update_task(self, processed_input):
    """Handle task update intent"""
    entities = processed_input["entities"]

    # Check if task ID is provided
    if "task_id" not in entities:
        self._add_reasoning_step("No task ID provided for update")
        return {
            "action_type": "request_clarification",
            "response": "Which task would you like to update? Please specify the task."
        }

    # Extract updates
    updates = {}
    if "priority" in entities:
        updates["priority"] = entities["priority"]
    if "status" in entities:
        updates["status"] = entities["status"]
    if "description" in entities:
        updates["description"] = entities["description"]
    if "due_date" in entities:
        updates["due_date"] = entities["due_date"]

    # Update task
    task_id = entities["task_id"]
    success = self.state_manager.update_task(task_id, **updates)

    if success:
        self._add_reasoning_step(f"Updated task {task_id} with {updates}")
        return {
            "action_type": "update_task",
            "task_id": task_id,
            "response": f"I've updated the task with your changes."
        }
    else:
        self._add_reasoning_step(f"Failed to find task {task_id}")
        return {
            "action_type": "error",
            "response": "I couldn't find that task. Can you try again with a different task?"
        }

def _handle_delete_task(self, processed_input):
    """Handle task deletion intent"""
    entities = processed_input["entities"]

    # Check if task ID is provided
    if "task_id" not in entities:
        self._add_reasoning_step("No task ID provided for deletion")
        return {
            "action_type": "request_clarification",
            "response": "Which task would you like to delete? Please specify the task."
        }

    # Delete task
    task_id = entities["task_id"]
    success = self.state_manager.delete_task(task_id)

    if success:
        self._add_reasoning_step(f"Deleted task {task_id}")
        return {
            "action_type": "delete_task",
            "task_id": task_id,
            "response": "I've deleted that task for you."
        }
    else:
        self._add_reasoning_step(f"Failed to find task {task_id}")
        return {
            "action_type": "error",
            "response": "I couldn't find that task to delete. Can you try again with a different task?"
        }
```

### Step 3: Query Handling

![Searching](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

Now, let's implement the ability to query tasks and handle unknown intents:

```python
def _handle_query_tasks(self, processed_input):
    """Handle task query intent"""
    entities = processed_input["entities"]

    # Determine query type
    query_type = entities.get("query_type", "all")
    self._add_reasoning_step(f"Query type: {query_type}")

    # Apply filters based on query type and entities
    filters = {}

    if query_type == "priority":
        if "priority" in entities:
            filters["priority"] = entities["priority"]
    elif query_type == "status":
        if "status" in entities:
            filters["status"] = entities["status"]
    elif query_type == "date":
        if "date" in entities:
            # This would require more sophisticated date handling
            pass
    elif query_type == "tag":
        if "tag" in entities:
            filters["tags"] = entities["tag"]

    # Get tasks with filters
    tasks = self.state_manager.get_tasks(filters)
    self._add_reasoning_step(f"Found {len(tasks)} tasks matching filters")

    # Format response based on tasks
    if not tasks:
        response = "You don't have any tasks matching those criteria."
    else:
        response = self._format_task_list(tasks)

    return {
        "action_type": "query_tasks",
        "tasks": tasks,
        "response": response
    }

def _handle_unknown_intent(self, processed_input):
    """Handle unknown or general intents"""
    self._add_reasoning_step("Handling unknown intent with general information")

    # Get task count for context
    tasks = self.state_manager.get_tasks()
    task_count = len(tasks)

    # Generate helpful response
    response = (
        "I'm your task manager assistant. I can help you create, update, delete, and query tasks. "
        f"Currently, you have {task_count} tasks. "
        "Try saying things like:\n"
        "- Add a new task: Finish the report by Friday\n"
        "- Show my high priority tasks\n"
        "- Mark the report task as completed\n"
        "- Delete the meeting task"
    )

    return {
        "action_type": "provide_help",
        "response": response
    }

def _format_task_list(self, tasks):
    """Format a list of tasks for display"""
    # Sort tasks by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(tasks, key=lambda t: priority_order.get(t.get("priority"), 3))

    # Format the list
    task_list = "Here are your tasks:\n"
    for i, task in enumerate(sorted_tasks):
        # Add emoji based on priority
        priority_emoji = "🔴" if task.get("priority") == "high" else "🟡" if task.get("priority") == "medium" else "🟢"

        # Add emoji based on status
        status_emoji = "✅" if task.get("status") == "completed" else "🔄" if task.get("status") == "in_progress" else "⏳"

        # Format due date if present
        due_str = f" (Due: {task.get('due_date')})" if "due_date" in task else ""

        task_list += f"{i+1}. {status_emoji} {task['description']} {priority_emoji}{due_str}\n"

    return task_list
```

### Step 4: User Preferences

![Preferences](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Let's implement preference management:

```python
def _handle_set_preference(self, processed_input):
    """Handle setting user preferences"""
    entities = processed_input["entities"]

    # Check if preference key and value are provided
    if "preference_key" not in entities or "preference_value" not in entities:
        self._add_reasoning_step("Incomplete preference information provided")
        return {
            "action_type": "request_clarification",
            "response": "What preference would you like to set? Please specify both the preference and value."
        }

    # Extract preference details
    pref_key = entities["preference_key"]
    pref_value = entities["preference_value"]

    # Update preference
    preferences = {pref_key: pref_value}
    self.state_manager.update_user_profile(preferences=preferences)

    self._add_reasoning_step(f"Updated preference {pref_key} to {pref_value}")

    return {
        "action_type": "set_preference",
        "preference": {pref_key: pref_value},
        "response": f"I've updated your {pref_key} preference to {pref_value}."
    }

def _initialize_prompt_library(self):
    """Initialize the prompt library with templates"""
    library = PromptLibrary()

    # Add task parsing template
    library.add_template(
        "task_parser",
        PromptTemplate.from_examples(
            instructions="Extract task details from the user input.",
            examples=[
                {
                    "input": "I need to finish the report by Friday",
                    "output": "{'description': 'finish the report', 'due_date': 'Friday', 'priority': 'medium'}"
                },
                {
                    "input": "Add a high priority task to call John tomorrow",
                    "output": "{'description': 'call John', 'due_date': 'tomorrow', 'priority': 'high'}"
                }
            ]
        )
    )

    # Add preference parsing template
    library.add_template(
        "preference_parser",
        PromptTemplate.from_examples(
            instructions="Extract user preference settings from the input.",
            examples=[
                {
                    "input": "I prefer dark theme",
                    "output": "{'preference_key': 'theme', 'preference_value': 'dark'}"
                },
                {
                    "input": "Sort my tasks by priority",
                    "output": "{'preference_key': 'task_sort_order', 'preference_value': 'priority'}"
                }
            ]
        )
    )

    return library
```

### Step 5: Intent and Entity Detection

![Detection](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Now, let's implement the methods for detecting intent and extracting entities:

```python
def _detect_intent(self, text):
    """Detect the user's intent from input text"""
    text = text.lower()

    # Task creation intents
    if any(phrase in text for phrase in ["add task", "create task", "new task", "remind me to"]):
        return "create_task"

    # Task update intents
    elif any(phrase in text for phrase in ["update task", "change task", "modify task", "mark task"]):
        return "update_task"

    # Task deletion intents
    elif any(phrase in text for phrase in ["delete task", "remove task", "cancel task"]):
        return "delete_task"

    # Task query intents
    elif any(phrase in text for phrase in ["show tasks", "list tasks", "what are my tasks", "find tasks"]):
        return "query_tasks"

    # Preference setting intents
    elif any(phrase in text for phrase in ["set preference", "change preference", "update settings", "i prefer"]):
        return "set_preference"

    # Default to general intent
    else:
        return "general"

def _extract_entities(self, text, intent):
    """Extract entities based on the detected intent"""
    entities = {}

    # Different extraction logic based on intent
    if intent == "create_task":
        entities = self._extract_task_entities(text)
    elif intent == "update_task":
        entities = self._extract_update_entities(text)
    elif intent == "delete_task":
        entities = self._extract_task_id(text)
    elif intent == "query_tasks":
        entities = self._extract_query_entities(text)
    elif intent == "set_preference":
        entities = self._extract_preference_entities(text)

    return entities

def _extract_task_entities(self, text):
    """Extract task details from text"""
    # In a real implementation, this would use more sophisticated NLP
    # or call an LLM with the task_parser prompt template

    entities = {"description": text}

    # Simple extraction of priority
    if "high priority" in text.lower() or "urgent" in text.lower():
        entities["priority"] = "high"
    elif "low priority" in text.lower() or "not urgent" in text.lower():
        entities["priority"] = "low"
    else:
        entities["priority"] = "medium"

    # Simple extraction of due date
    if "today" in text.lower():
        entities["due_date"] = "today"
    elif "tomorrow" in text.lower():
        entities["due_date"] = "tomorrow"
    elif "next week" in text.lower():
        entities["due_date"] = "next week"

    return entities

# Additional entity extraction methods would be implemented similarly
```

### Step 6: Testing and Validation

![Testing](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Finally, let's add some utility methods and complete our implementation:

```python
def _add_reasoning_step(self, step):
    """Add a reasoning step to the agent's internal state"""
    self.internal_state["reasoning_steps"].append({
        "step": step,
        "timestamp": time.time()
    })

def get_reasoning_steps(self):
    """Get the agent's reasoning steps for the last interaction"""
    return self.internal_state["reasoning_steps"]

def get_tasks_summary(self):
    """Get a summary of the user's tasks"""
    tasks = self.state_manager.get_tasks()

    # Count tasks by status
    status_counts = {}
    for task in tasks:
        status = task.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Count tasks by priority
    priority_counts = {}
    for task in tasks:
        priority = task.get("priority", "medium")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return {
        "total_tasks": len(tasks),
        "by_status": status_counts,
        "by_priority": priority_counts
    }

def get_daily_summary(self):
    """Generate a daily summary of tasks"""
    tasks = self.state_manager.get_tasks()

    # Filter for pending and in-progress tasks
    active_tasks = [t for t in tasks if t.get("status") in ["pending", "in_progress"]]

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(active_tasks, key=lambda t: priority_order.get(t.get("priority"), 3))

    # Generate summary
    if not sorted_tasks:
        return "You have no active tasks for today. Great job!"

    summary = f"Daily Summary: You have {len(sorted_tasks)} active tasks.\n\n"

    # Add high priority tasks first
    high_priority = [t for t in sorted_tasks if t.get("priority") == "high"]
    if high_priority:
        summary += "High Priority:\n"
        for i, task in enumerate(high_priority):
            summary += f"- {task['description']}\n"
        summary += "\n"

    # Add other tasks
    other_tasks = [t for t in sorted_tasks if t.get("priority") != "high"]
    if other_tasks:
        summary += "Other Tasks:\n"
        for i, task in enumerate(other_tasks):
            priority = "Medium" if task.get("priority") == "medium" else "Low"
            summary += f"- {task['description']} ({priority})\n"

    return summary
```

With these implementations, we now have a complete TaskManagerAgent that can handle task creation, updates, deletions, and queries, as well as manage user preferences and provide summaries.

---

## 🧪 Testing Our Task Manager

![Testing](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Now that we've built our task manager, let's create a test script to verify that it works correctly:

```python
import time
from task_manager_agent import TaskManagerAgent

def test_task_manager():
    """Test the TaskManagerAgent with various inputs"""
    # Create a clean test directory
    test_dir = "test_task_manager_data"

    # Initialize the agent
    agent = TaskManagerAgent(storage_dir=test_dir)

    print("=== Task Manager Agent Test ===\n")

    # Test task creation
    print("Testing task creation...")
    inputs = [
        "Add a new task: Finish the project proposal by Friday",
        "I need to call John tomorrow, high priority",
        "Remind me to buy groceries next week"
    ]

    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")

    # Test task queries
    print("\nTesting task queries...")
    query = "Show me all my tasks"
    print(f"\nUser: {query}")
    response = agent.agent_loop(query)
    print(f"Agent: {response}")

    # Test task updates
    print("\nTesting task updates...")
    update = "Mark the project proposal as completed"
    print(f"\nUser: {update}")
    response = agent.agent_loop(update)
    print(f"Agent: {response}")

    # Test preference setting
    print("\nTesting preference setting...")
    preference = "I prefer to sort tasks by priority"
    print(f"\nUser: {preference}")
    response = agent.agent_loop(preference)
    print(f"Agent: {response}")

    # Test daily summary
    print("\nTesting daily summary...")
    summary = agent.get_daily_summary()
    print(f"Daily Summary:\n{summary}")

    # Test task statistics
    print("\nTask Statistics:")
    stats = agent.get_tasks_summary()
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"By status: {stats['by_status']}")
    print(f"By priority: {stats['by_priority']}")

if __name__ == "__main__":
    test_task_manager()
```

## 🚀 Extending the Task Manager

![Extensions](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Our task manager is already quite capable, but there are many ways we could extend it:

1. **Natural Language Processing**: Integrate with a real LLM to improve entity extraction and intent detection
2. **Date Handling**: Add more sophisticated date parsing for due dates
3. **Recurring Tasks**: Implement support for recurring tasks (daily, weekly, monthly)
4. **Categories and Tags**: Add support for categorizing tasks and using tags
5. **Notifications**: Implement a notification system for upcoming deadlines
6. **Multi-user Support**: Extend the system to support multiple users
7. **Visualization**: Add methods to visualize task distribution and progress
8. **Integration**: Connect with external calendars, email, or other productivity tools

Here's an example of how we might implement recurring tasks:

```python
def _handle_recurring_task(self, processed_input):
    """Handle creation of recurring tasks"""
    entities = processed_input["entities"]

    # Extract task details
    task = {
        "description": entities.get("description", processed_input["text"]),
        "priority": entities.get("priority", "medium"),
        "status": "pending",
        "created_at": time.time(),
        "is_recurring": True,
        "recurrence_pattern": entities.get("recurrence_pattern", "daily"),
        "recurrence_count": 0
    }

    # Add the task
    task_id = self.state_manager.add_task(task)

    # Add reasoning step
    self._add_reasoning_step(f"Created recurring task with ID {task_id}: {task['description']}")

    # Generate response
    response = f"I've added your recurring task: {task['description']} (repeats {task['recurrence_pattern']})"

    return {
        "action_type": "create_recurring_task",
        "task_id": task_id,
        "response": response
    }
```

## 💡 Best Practices

As you continue to develop your agent, keep these best practices in mind:

1. **Separation of Concerns**: Keep the sense-think-act loop clean by delegating specific functionality to helper methods
2. **Error Handling**: Implement robust error handling to gracefully manage unexpected inputs
3. **Testing**: Create comprehensive tests for each component of your agent
4. **Documentation**: Document your code thoroughly, especially the interfaces between components
5. **Incremental Development**: Build one feature at a time and test thoroughly before moving on
6. **User Feedback**: Incorporate mechanisms to learn from user interactions
7. **Performance Monitoring**: Add logging and monitoring to identify bottlenecks

> 💡 **Important Note on LLMs in Module 1**
>
> In Module 1, we've created a **simulation** of how agents would work with LLMs, but we're not actually integrating with real LLMs like GPT-4, Claude, or others. We've used simplified rule-based logic that simulates what an LLM might return.
>
> Throughout the code, you'll see comments indicating where a real implementation would use an LLM. For example:
> ```python
> # In a real implementation, this would use more sophisticated NLP
> # or call an LLM with the task_parser prompt template
> ```
>
> This approach allows us to focus on the fundamental architecture of agents (sense-think-act loop, prompt engineering, state management) without adding the complexity of LLM integration, API dependencies, or potential costs.
>
> In later modules, we'll explore actual LLM integration, showing how to connect to LLM APIs, handle responses and errors, manage tokens and costs, and implement more sophisticated prompt engineering techniques specific to different LLMs.

---

## 🎯 Mini-Project Completion

![Completion](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Congratulations! You've completed the Personal Task Manager mini-project. This project has demonstrated how to:

- Implement the sense-think-act loop in a practical application
- Use prompt templates to guide agent behavior
- Manage state effectively for a stateful agent
- Handle natural language inputs and generate appropriate responses
- Test and validate agent functionality

You now have a solid foundation in agent fundamentals that you can build upon in future modules.

---

## 📚 Resources

- [Python Documentation](https://docs.python.org/3/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Task Management Best Practices](https://todoist.com/productivity-methods)

---

## 🚀 Next Steps

In the next module, we'll explore Memory Systems in more depth, including:
- Different memory types (working, short-term, long-term)
- Vector database fundamentals
- Retrieval patterns for contextual memory

We'll build on the foundation established in this module to create even more sophisticated agents!

---

Happy coding! 🚀

