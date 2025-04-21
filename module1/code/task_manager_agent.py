"""
Task Manager Agent Implementation
------------------------------
This file contains a complete implementation of a Personal Task Manager agent
that integrates the sense-think-act loop with prompt templates and state management.
"""

import time
import json
import os
from .state_management import AgentStateManager
from .prompt_template import PromptTemplate, PromptLibrary

class TaskManagerAgent:
    def __init__(self, storage_dir="task_manager_data"):
        """
        Initialize the task manager agent
        
        Args:
            storage_dir (str): Directory for persistent storage
        """
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
        """
        Process user input to extract intent and entities
        
        Args:
            user_input (str): The raw input from the user
            
        Returns:
            dict: Processed input with intent and entities
        """
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
        """
        Determine actions based on intent and entities
        
        Args:
            processed_input (dict): The processed input from the sense phase
            
        Returns:
            dict: Action plan with response
        """
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
        """
        Execute the action plan and generate a response
        
        Args:
            action_plan (dict): The plan created in the think phase
            
        Returns:
            str: The response to be shown to the user
        """
        # Update conversation with agent's response
        self.state_manager.update_conversation("agent", action_plan["response"])
        
        # Save state
        self.state_manager.save_state()
        
        # Update internal state
        self.internal_state["last_action"] = action_plan["action_type"]
        
        # Return the response
        return action_plan["response"]
    
    def agent_loop(self, user_input):
        """
        Run the full sense-think-act cycle
        
        Args:
            user_input (str): Raw input from the user
            
        Returns:
            str: The agent's response
        """
        processed_input = self.sense(user_input)
        action_plan = self.think(processed_input)
        response = self.act(action_plan)
        return response
    
    # Task handling methods
    
    def _handle_create_task(self, processed_input):
        """
        Handle task creation intent
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for task creation
        """
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
        """
        Handle task update intent
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for task update
        """
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
        """
        Handle task deletion intent
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for task deletion
        """
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
    
    def _handle_query_tasks(self, processed_input):
        """
        Handle task query intent
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for task query
        """
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
    
    def _handle_set_preference(self, processed_input):
        """
        Handle setting user preferences
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for preference setting
        """
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
    
    def _handle_unknown_intent(self, processed_input):
        """
        Handle unknown or general intents
        
        Args:
            processed_input (dict): Processed user input
            
        Returns:
            dict: Action plan for general response
        """
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
    
    # Intent and entity detection
    
    def _detect_intent(self, text):
        """
        Detect the user's intent from input text
        
        Args:
            text (str): The user's input text
            
        Returns:
            str: The detected intent
        """
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
        """
        Extract entities based on the detected intent
        
        Args:
            text (str): The user's input text
            intent (str): The detected intent
            
        Returns:
            dict: Extracted entities
        """
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
        """
        Extract task details from text
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted task entities
        """
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
    
    def _extract_update_entities(self, text):
        """
        Extract task update details from text
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted update entities
        """
        entities = {}
        
        # Extract task ID (simplified - in a real implementation, this would be more sophisticated)
        # For this example, we'll just use the first task if available
        tasks = self.state_manager.get_tasks()
        if tasks:
            entities["task_id"] = tasks[0]["id"]
        
        # Extract status updates
        if "completed" in text.lower() or "done" in text.lower():
            entities["status"] = "completed"
        elif "in progress" in text.lower() or "started" in text.lower():
            entities["status"] = "in_progress"
        
        # Extract priority updates
        if "high priority" in text.lower() or "urgent" in text.lower():
            entities["priority"] = "high"
        elif "medium priority" in text.lower():
            entities["priority"] = "medium"
        elif "low priority" in text.lower() or "not urgent" in text.lower():
            entities["priority"] = "low"
        
        return entities
    
    def _extract_task_id(self, text):
        """
        Extract task ID from text
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted task ID
        """
        entities = {}
        
        # Simplified implementation - in a real system, this would use more sophisticated NLP
        # For this example, we'll just use the first task if available
        tasks = self.state_manager.get_tasks()
        if tasks:
            entities["task_id"] = tasks[0]["id"]
        
        return entities
    
    def _extract_query_entities(self, text):
        """
        Extract query parameters from text
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted query entities
        """
        entities = {"query_type": "all"}
        
        # Determine query type and parameters
        if "high priority" in text.lower():
            entities["query_type"] = "priority"
            entities["priority"] = "high"
        elif "medium priority" in text.lower():
            entities["query_type"] = "priority"
            entities["priority"] = "medium"
        elif "low priority" in text.lower():
            entities["query_type"] = "priority"
            entities["priority"] = "low"
        elif "completed" in text.lower() or "done" in text.lower():
            entities["query_type"] = "status"
            entities["status"] = "completed"
        elif "in progress" in text.lower():
            entities["query_type"] = "status"
            entities["status"] = "in_progress"
        elif "pending" in text.lower():
            entities["query_type"] = "status"
            entities["status"] = "pending"
        
        return entities
    
    def _extract_preference_entities(self, text):
        """
        Extract preference settings from text
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted preference entities
        """
        entities = {}
        
        # Extract theme preferences
        if "dark theme" in text.lower() or "dark mode" in text.lower():
            entities["preference_key"] = "theme"
            entities["preference_value"] = "dark"
        elif "light theme" in text.lower() or "light mode" in text.lower():
            entities["preference_key"] = "theme"
            entities["preference_value"] = "light"
        
        # Extract sort preferences
        elif "sort by priority" in text.lower():
            entities["preference_key"] = "task_sort_order"
            entities["preference_value"] = "priority"
        elif "sort by deadline" in text.lower() or "sort by due date" in text.lower():
            entities["preference_key"] = "task_sort_order"
            entities["preference_value"] = "deadline"
        elif "sort by creation" in text.lower():
            entities["preference_key"] = "task_sort_order"
            entities["preference_value"] = "creation"
        
        # Extract notification preferences
        elif "daily notifications" in text.lower():
            entities["preference_key"] = "notification_frequency"
            entities["preference_value"] = "daily"
        elif "weekly notifications" in text.lower():
            entities["preference_key"] = "notification_frequency"
            entities["preference_value"] = "weekly"
        
        return entities
    
    # Utility methods
    
    def _initialize_prompt_library(self):
        """
        Initialize the prompt library with templates
        
        Returns:
            PromptLibrary: Initialized prompt library
        """
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
    
    def _format_task_list(self, tasks):
        """
        Format a list of tasks for display
        
        Args:
            tasks (list): List of task dictionaries
            
        Returns:
            str: Formatted task list
        """
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
    
    def _add_reasoning_step(self, step):
        """
        Add a reasoning step to the agent's internal state
        
        Args:
            step (str): The reasoning step to add
        """
        self.internal_state["reasoning_steps"].append({
            "step": step,
            "timestamp": time.time()
        })
    
    # Public utility methods
    
    def get_reasoning_steps(self):
        """
        Get the agent's reasoning steps for the last interaction
        
        Returns:
            list: Reasoning steps
        """
        return self.internal_state["reasoning_steps"]
    
    def get_tasks_summary(self):
        """
        Get a summary of the user's tasks
        
        Returns:
            dict: Task summary statistics
        """
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
        """
        Generate a daily summary of tasks
        
        Returns:
            str: Formatted daily summary
        """
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


# Example usage
if __name__ == "__main__":
    # Create a clean test directory
    test_dir = "task_manager_test_data"
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # Initialize the agent
    agent = TaskManagerAgent(storage_dir=test_dir)
    
    print("Task Manager Agent Demo (type 'exit' to quit)")
    print("Try commands like:")
    print("- Add a new task: Finish the report by Friday")
    print("- Show my tasks")
    print("- Mark the report task as completed")
    print("- I prefer dark theme")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting task manager...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
        
        # Uncomment to see reasoning steps
        # print("\nReasoning steps:")
        # for step in agent.get_reasoning_steps():
        #     print(f"- {step['step']}")
