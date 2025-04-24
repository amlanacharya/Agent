"""
Test Script for Lesson 3 Exercise Solutions
----------------------------------------
This script demonstrates how to test the state management exercise solutions from Lesson 3.
"""

import os
import shutil
import time

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/exercises directory
    from state_exercises import ConversationMemorySystem, UserProfileManager, TaskStateManager
except ImportError:
    # When running from the project root
    from module1.exercises.state_exercises import ConversationMemorySystem, UserProfileManager, TaskStateManager

def test_conversation_memory():
    """Test the ConversationMemorySystem implementation"""
    print("=== Conversation Memory System Test ===")
    
    # Create a new conversation memory system
    conversation = ConversationMemorySystem(max_history=10)
    
    # Add messages with topics
    conversation.add_message("user", "Hello, I need help with my tasks", ["greeting", "tasks"])
    conversation.add_message("agent", "I can help you manage your tasks. What would you like to do?", ["tasks"])
    conversation.add_message("user", "I need to create a report for tomorrow", ["tasks", "reports", "deadlines"])
    conversation.add_message("agent", "I'll add that to your task list. Anything else?", ["tasks", "confirmation"])
    conversation.add_message("user", "Also remind me to call John", ["tasks", "reminders"])
    
    # Test getting conversation history
    history = conversation.get_conversation_history()
    print(f"Conversation history length: {len(history)}")
    
    # Test getting messages by topic
    task_messages = conversation.get_messages_by_topic("tasks")
    report_messages = conversation.get_messages_by_topic("reports")
    print(f"Messages about tasks: {len(task_messages)}")
    print(f"Messages about reports: {len(report_messages)}")
    
    # Test conversation summary
    summary = conversation.summarize_conversation()
    print("\nConversation Summary:")
    print(f"Message count: {summary['message_count']}")
    print(f"Role distribution: {summary['role_distribution']}")
    print("Top topics:")
    for topic, count in summary['top_topics']:
        print(f"  - {topic}: {count}")
    
    # Test search (if implemented)
    if hasattr(conversation, 'search_conversations'):
        results = conversation.search_conversations("report")
        print(f"\nSearch results for 'report': {len(results)}")
    
    print("-" * 50)

def test_user_profile_manager():
    """Test the UserProfileManager implementation"""
    print("\n=== User Profile Manager Test ===")
    
    # Create a temporary file for testing
    test_file = "test_profiles.json"
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Create a new profile manager
    profile_manager = UserProfileManager(storage_path=test_file)
    
    # Create users
    profile_manager.create_user("user1", "Alice")
    profile_manager.create_user("user2", "Bob", {"theme": "dark"})
    
    # Set current user
    profile_manager.set_current_user("user1")
    
    # Update preferences
    profile_manager.update_preferences({"notification_frequency": "weekly"})
    
    # Log behaviors
    profile_manager.log_behavior("app_usage", {"time_of_day": "morning"})
    profile_manager.log_behavior("task_creation", {"priority": "high"})
    profile_manager.log_behavior("notification_interaction", {"action": "dismissed"})
    
    # Get user data
    user_data = profile_manager.get_user_data()
    print(f"User name: {user_data['name']}")
    print(f"Preferences: {user_data['preferences']}")
    
    # Get suggestions
    suggestions = profile_manager.get_suggestions()
    print("\nPreference suggestions:")
    for pref, details in suggestions.items():
        print(f"  - {pref}: {details['value']} (Confidence: {details['confidence']})")
        print(f"    Reason: {details['reason']}")
    
    # Switch user
    profile_manager.set_current_user("user2")
    user2_data = profile_manager.get_user_data()
    print(f"\nSwitched to user: {user2_data['name']}")
    print(f"Theme preference: {user2_data['preferences'].get('theme')}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("-" * 50)

def test_task_state_manager():
    """Test the TaskStateManager implementation"""
    print("\n=== Task State Manager Test ===")
    
    # Create a temporary file for testing
    test_file = "test_tasks.json"
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Create a new task manager
    task_manager = TaskStateManager(storage_path=test_file)
    
    # Add tasks
    task1_id = task_manager.add_task("Complete project proposal", priority="high", due_date="2023-12-15")
    task2_id = task_manager.add_task("Schedule team meeting", priority="medium", due_date="2023-12-10")
    task3_id = task_manager.add_task("Review documentation", priority="low")
    
    # Get all tasks
    all_tasks = task_manager.get_tasks()
    print(f"Total tasks: {len(all_tasks)}")
    
    # Get tasks by priority
    high_priority = task_manager.get_tasks(filters={"priority": "high"})
    print(f"High priority tasks: {len(high_priority)}")
    
    # Update a task
    task_manager.update_task(task1_id, status="in_progress", notes="Started working on this")
    
    # Get task by ID
    task1 = task_manager.get_task(task1_id)
    print(f"\nTask details:")
    print(f"  Description: {task1['description']}")
    print(f"  Priority: {task1['priority']}")
    print(f"  Status: {task1['status']}")
    print(f"  Notes: {task1.get('notes', 'None')}")
    
    # Delete a task
    task_manager.delete_task(task3_id)
    remaining_tasks = task_manager.get_tasks()
    print(f"\nRemaining tasks after deletion: {len(remaining_tasks)}")
    
    # Test persistence
    task_manager.save_tasks()
    new_manager = TaskStateManager(storage_path=test_file)
    loaded_tasks = new_manager.get_tasks()
    print(f"Tasks loaded from file: {len(loaded_tasks)}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("-" * 50)

if __name__ == "__main__":
    test_conversation_memory()
    test_user_profile_manager()
    test_task_state_manager()
