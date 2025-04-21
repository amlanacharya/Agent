"""
Stateful Agent Implementation
--------------------------
This file contains an implementation of an agent with comprehensive state management.
Use this as a reference for Module 1, Lesson 3.
"""

import time
import json
import os
from state_management import AgentStateManager

class StatefulAgent:
    def __init__(self, storage_dir="agent_data"):
        """
        Initialize the agent with state management
        
        Args:
            storage_dir (str): Directory for persistent storage
        """
        # Initialize state manager
        self.state_manager = AgentStateManager(storage_dir=storage_dir)
        
        # Track agent's internal state
        self.internal_state = {
            "current_goal": None,
            "reasoning_steps": [],
            "last_action": None
        }
    
    def sense(self, user_input):
        """
        Process user input and update state
        
        Args:
            user_input (str): The raw input from the user
            
        Returns:
            dict: Processed input with metadata
        """
        # Update conversation state
        self.state_manager.update_conversation("user", user_input)
        
        # Process the input
        processed_input = {
            "text": user_input,
            "timestamp": time.time(),
            "detected_intent": self._detect_intent(user_input)
        }
        
        # Update internal state
        self.internal_state["last_action"] = "sense"
        
        return processed_input
    
    def think(self, processed_input):
        """
        Generate a response based on input and state
        
        Args:
            processed_input (dict): The processed input from the sense phase
            
        Returns:
            dict: Action plan with response type and content
        """
        # Clear previous reasoning steps
        self.internal_state["reasoning_steps"] = []
        
        # Set current goal based on intent
        self.internal_state["current_goal"] = processed_input["detected_intent"]
        
        # Add reasoning step
        self._add_reasoning_step(f"Detected intent: {processed_input['detected_intent']}")
        
        # Get recent conversation for context
        recent_messages = self.state_manager.get_conversation_history(5)
        self._add_reasoning_step(f"Considered {len(recent_messages)} recent messages for context")
        
        # Determine how to respond based on intent and state
        if processed_input["detected_intent"] == "add_task":
            # Extract task details and add to state
            task = self._extract_task_details(processed_input["text"])
            self._add_reasoning_step(f"Extracted task details: {task['description']}")
            
            task_id = self.state_manager.add_task(task)
            self._add_reasoning_step(f"Added task with ID: {task_id}")
            
            response = {
                "type": "task_confirmation",
                "content": f"I've added your task: {task['description']}",
                "task_id": task_id
            }
        
        elif processed_input["detected_intent"] == "query_tasks":
            # Retrieve relevant tasks
            tasks = self.state_manager.get_tasks()
            self._add_reasoning_step(f"Retrieved {len(tasks)} tasks")
            
            response = {
                "type": "task_list",
                "content": self._format_task_list(tasks),
                "tasks": tasks
            }
        
        elif processed_input["detected_intent"] == "update_task":
            # Extract task ID and updates
            task_id, updates = self._extract_task_updates(processed_input["text"])
            self._add_reasoning_step(f"Extracted updates for task {task_id}: {updates}")
            
            if task_id and updates:
                success = self.state_manager.update_task(task_id, **updates)
                if success:
                    self._add_reasoning_step("Successfully updated task")
                    response = {
                        "type": "task_update_confirmation",
                        "content": f"I've updated the task with your changes.",
                        "task_id": task_id
                    }
                else:
                    self._add_reasoning_step("Failed to find task with that ID")
                    response = {
                        "type": "error",
                        "content": "I couldn't find that task. Can you try again with a different task?"
                    }
            else:
                self._add_reasoning_step("Could not determine which task to update")
                response = {
                    "type": "clarification_request",
                    "content": "I'm not sure which task you want to update. Can you specify the task more clearly?"
                }
        
        elif processed_input["detected_intent"] == "delete_task":
            # Extract task ID
            task_id = self._extract_task_id(processed_input["text"])
            self._add_reasoning_step(f"Extracted task ID for deletion: {task_id}")
            
            if task_id:
                success = self.state_manager.delete_task(task_id)
                if success:
                    self._add_reasoning_step("Successfully deleted task")
                    response = {
                        "type": "task_deletion_confirmation",
                        "content": "I've deleted that task for you.",
                        "task_id": task_id
                    }
                else:
                    self._add_reasoning_step("Failed to find task with that ID")
                    response = {
                        "type": "error",
                        "content": "I couldn't find that task to delete. Can you try again with a different task?"
                    }
            else:
                self._add_reasoning_step("Could not determine which task to delete")
                response = {
                    "type": "clarification_request",
                    "content": "I'm not sure which task you want to delete. Can you specify the task more clearly?"
                }
        
        elif processed_input["detected_intent"] == "set_preference":
            # Extract preference updates
            preference_updates = self._extract_preferences(processed_input["text"])
            self._add_reasoning_step(f"Extracted preference updates: {preference_updates}")
            
            if preference_updates:
                self.state_manager.update_user_profile(preferences=preference_updates)
                self._add_reasoning_step("Updated user preferences")
                
                response = {
                    "type": "preference_confirmation",
                    "content": f"I've updated your preferences.",
                    "updates": preference_updates
                }
            else:
                self._add_reasoning_step("Could not determine preference updates")
                response = {
                    "type": "clarification_request",
                    "content": "I'm not sure what preferences you want to change. Can you be more specific?"
                }
        
        else:
            # Default response for general queries
            self._add_reasoning_step("No specific intent matched, providing general response")
            
            response = {
                "type": "general",
                "content": "I'm your task management assistant. I can help you add, update, delete, and query tasks, as well as set your preferences."
            }
        
        # Update internal state
        self.internal_state["last_action"] = "think"
        
        return response
    
    def act(self, response):
        """
        Execute the response and update state
        
        Args:
            response (dict): The response plan from the think phase
            
        Returns:
            str: The response to be shown to the user
        """
        # Update conversation state with agent's response
        self.state_manager.update_conversation("agent", response["content"])
        
        # Save state after each interaction
        self.state_manager.save_state()
        
        # Update internal state
        self.internal_state["last_action"] = "act"
        
        # Return the response content
        return response["content"]
    
    def agent_loop(self, user_input):
        """
        Run the full sense-think-act cycle with state management
        
        Args:
            user_input (str): Raw input from the user
            
        Returns:
            str: The agent's response
        """
        processed_input = self.sense(user_input)
        response = self.think(processed_input)
        output = self.act(response)
        return output
    
    def get_state_summary(self):
        """
        Get a summary of the agent's current state
        
        Returns:
            dict: A summary of the agent's state
        """
        return {
            "user_profile": self.state_manager.user_profile,
            "task_count": len(self.state_manager.get_tasks()),
            "conversation_length": len(self.state_manager.get_conversation_history()),
            "current_goal": self.internal_state["current_goal"],
            "reasoning_steps": self.internal_state["reasoning_steps"]
        }
    
    # Helper methods
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
            return "add_task"
        
        # Task query intents
        elif any(phrase in text for phrase in ["show tasks", "list tasks", "what are my tasks", "pending tasks"]):
            return "query_tasks"
        
        # Task update intents
        elif any(phrase in text for phrase in ["update task", "change task", "modify task", "mark task"]):
            return "update_task"
        
        # Task deletion intents
        elif any(phrase in text for phrase in ["delete task", "remove task", "cancel task"]):
            return "delete_task"
        
        # Preference setting intents
        elif any(phrase in text for phrase in ["set preference", "change preference", "update settings"]):
            return "set_preference"
        
        # Default to general intent
        else:
            return "general"
    
    def _extract_task_details(self, text):
        """
        Extract task details from user input
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted task details
        """
        # In a real implementation, this would use more sophisticated NLP
        # For now, we'll use a simple implementation
        description = text
        priority = "medium"
        status = "pending"
        
        # Extract priority if mentioned
        if "urgent" in text.lower() or "important" in text.lower():
            priority = "high"
        elif "low priority" in text.lower() or "not urgent" in text.lower():
            priority = "low"
        
        # Extract status if mentioned
        if "in progress" in text.lower() or "started" in text.lower():
            status = "in_progress"
        elif "completed" in text.lower() or "done" in text.lower():
            status = "completed"
        
        return {
            "description": description,
            "created_at": time.time(),
            "priority": priority,
            "status": status
        }
    
    def _extract_task_updates(self, text):
        """
        Extract task ID and updates from user input
        
        Args:
            text (str): The user's input text
            
        Returns:
            tuple: (task_id, updates_dict) or (None, None) if extraction fails
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP
        
        # For this example, we'll assume the first task is being updated
        # In a real implementation, you would extract the specific task ID
        tasks = self.state_manager.get_tasks()
        if not tasks:
            return None, None
        
        task_id = tasks[0]["id"]
        updates = {}
        
        # Extract priority updates
        if "high priority" in text.lower() or "urgent" in text.lower():
            updates["priority"] = "high"
        elif "medium priority" in text.lower():
            updates["priority"] = "medium"
        elif "low priority" in text.lower():
            updates["priority"] = "low"
        
        # Extract status updates
        if "in progress" in text.lower() or "started" in text.lower():
            updates["status"] = "in_progress"
        elif "completed" in text.lower() or "done" in text.lower():
            updates["status"] = "completed"
        elif "not started" in text.lower() or "pending" in text.lower():
            updates["status"] = "pending"
        
        return (task_id, updates) if updates else (None, None)
    
    def _extract_task_id(self, text):
        """
        Extract task ID from user input
        
        Args:
            text (str): The user's input text
            
        Returns:
            str: The extracted task ID or None if extraction fails
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP
        
        # For this example, we'll assume the first task is being referenced
        # In a real implementation, you would extract the specific task ID
        tasks = self.state_manager.get_tasks()
        if not tasks:
            return None
        
        return tasks[0]["id"]
    
    def _extract_preferences(self, text):
        """
        Extract preference updates from user input
        
        Args:
            text (str): The user's input text
            
        Returns:
            dict: Extracted preference updates
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP
        
        preferences = {}
        
        # Extract notification frequency
        if "daily notifications" in text.lower():
            preferences["notification_frequency"] = "daily"
        elif "weekly notifications" in text.lower():
            preferences["notification_frequency"] = "weekly"
        elif "no notifications" in text.lower():
            preferences["notification_frequency"] = "none"
        
        # Extract sort order
        if "sort by deadline" in text.lower():
            preferences["task_sort_order"] = "deadline"
        elif "sort by priority" in text.lower():
            preferences["task_sort_order"] = "priority"
        elif "sort by creation" in text.lower():
            preferences["task_sort_order"] = "creation"
        
        # Extract theme
        if "dark theme" in text.lower() or "dark mode" in text.lower():
            preferences["theme"] = "dark"
        elif "light theme" in text.lower() or "light mode" in text.lower():
            preferences["theme"] = "light"
        
        return preferences
    
    def _format_task_list(self, tasks):
        """
        Format a list of tasks for display
        
        Args:
            tasks (list): List of task dictionaries
            
        Returns:
            str: Formatted task list
        """
        if not tasks:
            return "You don't have any tasks yet."
        
        task_list = "Here are your tasks:\n"
        for i, task in enumerate(tasks):
            status_emoji = "🔄" if task["status"] == "in_progress" else "✅" if task["status"] == "completed" else "⏳"
            priority_emoji = "🔴" if task["priority"] == "high" else "🟡" if task["priority"] == "medium" else "🟢"
            
            task_list += f"{i+1}. {status_emoji} {task['description']} {priority_emoji}\n"
        
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


# Example usage
if __name__ == "__main__":
    # Create a clean test directory
    test_dir = "agent_test_data"
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # Create the agent
    agent = StatefulAgent(storage_dir=test_dir)
    
    print("Stateful Agent Demo (type 'exit' to quit)")
    print("Try commands like:")
    print("- 'Add a new task: Finish the report by Friday'")
    print("- 'Show my tasks'")
    print("- 'Update the task to high priority'")
    print("- 'Delete the task'")
    print("- 'Set my preference to dark theme'")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
        
        # Uncomment to see the agent's reasoning steps
        # print("\nReasoning steps:")
        # for step in agent.internal_state["reasoning_steps"]:
        #     print(f"- {step['step']}")
        # print()
