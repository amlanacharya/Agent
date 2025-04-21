"""
Test Script for State Management
-----------------------------
This script demonstrates how to test the state management implementations.
"""

import os
import json
import time
import shutil

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/code directory
    from state_management import ShortTermMemory, LongTermMemory, EpisodicMemory, AgentStateManager
except ImportError:
    # When running from the project root
    from module1.code.state_management import ShortTermMemory, LongTermMemory, EpisodicMemory, AgentStateManager

def test_short_term_memory():
    """Test the ShortTermMemory class"""
    print("=== Testing ShortTermMemory ===")
    
    # Create a memory with capacity of 3
    memory = ShortTermMemory(capacity=3)
    
    # Add items
    memory.add("Item 1")
    memory.add("Item 2")
    memory.add("Item 3")
    
    # Check capacity enforcement
    print(f"Memory after adding 3 items: {memory.memory}")
    
    # Add one more item (should remove the oldest)
    memory.add("Item 4")
    print(f"Memory after adding a 4th item: {memory.memory}")
    
    # Test get_recent
    print(f"2 most recent items: {memory.get_recent(2)}")
    
    # Test clear
    memory.clear()
    print(f"Memory after clearing: {memory.memory}")

def test_long_term_memory():
    """Test the LongTermMemory class"""
    print("\n=== Testing LongTermMemory ===")
    
    # Create a test directory
    test_dir = "test_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a memory with a test file
    memory_path = os.path.join(test_dir, "test_long_term.json")
    memory = LongTermMemory(storage_path=memory_path)
    
    # Store some data
    memory.store("user_profiles", "alice", {"name": "Alice", "preferences": {"theme": "dark"}})
    memory.store("frequent_tasks", "weekly_report", {"count": 5, "last_created": "2023-06-01"})
    
    # Retrieve data
    alice_profile = memory.retrieve("user_profiles", "alice")
    print(f"Retrieved Alice's profile: {alice_profile}")
    
    all_profiles = memory.retrieve("user_profiles")
    print(f"All profiles: {all_profiles}")
    
    # Test default value
    unknown_data = memory.retrieve("unknown_category", "unknown_key", "default_value")
    print(f"Unknown data with default: {unknown_data}")
    
    # Verify file was created
    print(f"Memory file exists: {os.path.exists(memory_path)}")
    
    # Clean up
    shutil.rmtree(test_dir)

def test_episodic_memory():
    """Test the EpisodicMemory class"""
    print("\n=== Testing EpisodicMemory ===")
    
    # Create a memory
    memory = EpisodicMemory()
    
    # Add some interactions
    memory.add_interaction("Hello, how can you help me?")
    memory.update_last_response("I can help you manage your tasks.")
    
    memory.add_interaction("Can you create a task for me?")
    memory.update_last_response("Sure, what task would you like to create?")
    
    # Print current session
    current_session = memory.get_current_session()
    print(f"Current session ID: {current_session['id']}")
    print(f"Interactions in current session: {len(current_session['interactions'])}")
    
    # End the session
    memory.end_session(summary="User asked about task creation")
    
    # Start a new session
    memory.add_interaction("I need to create a report")
    memory.update_last_response("I'll help you create a report task")
    
    # Print sessions
    print(f"Total sessions: {len(memory.sessions) + 1}")  # +1 for current session
    print(f"First session summary: {memory.sessions[0]['summary']}")

def test_agent_state_manager():
    """Test the AgentStateManager class"""
    print("\n=== Testing AgentStateManager ===")
    
    # Create a test directory
    test_dir = "test_agent_state"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create a state manager
    state_manager = AgentStateManager(storage_dir=test_dir)
    
    # Test conversation updates
    state_manager.update_conversation("user", "Hello, can you help me?")
    state_manager.update_conversation("agent", "Yes, I can help you manage your tasks.")
    
    # Test user profile updates
    state_manager.update_user_profile(
        name="Bob",
        preferences={"theme": "dark", "notification_frequency": "weekly"}
    )
    
    # Test task management
    task1_id = state_manager.add_task({
        "description": "Complete the report",
        "priority": "high",
        "status": "pending"
    })
    
    task2_id = state_manager.add_task({
        "description": "Schedule meeting",
        "priority": "medium",
        "status": "pending"
    })
    
    # Update a task
    state_manager.update_task(task1_id, status="in_progress")
    
    # Get tasks
    all_tasks = state_manager.get_tasks()
    high_priority_tasks = state_manager.get_tasks(filters={"priority": "high"})
    
    # Print state information
    print(f"Conversation history length: {len(state_manager.get_conversation_history())}")
    print(f"User name: {state_manager.user_profile['name']}")
    print(f"User preferences: {state_manager.user_profile['preferences']}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"High priority tasks: {len(high_priority_tasks)}")
    
    # Save state
    state_manager.save_state()
    
    # Verify files were created
    print(f"Files created in {test_dir}: {os.listdir(test_dir)}")
    
    # Create a new state manager to test loading
    new_state_manager = AgentStateManager(storage_dir=test_dir)
    
    # Verify state was loaded
    print(f"Loaded user name: {new_state_manager.user_profile['name']}")
    print(f"Loaded tasks count: {len(new_state_manager.get_tasks())}")
    
    # Clean up
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_short_term_memory()
    test_long_term_memory()
    test_episodic_memory()
    test_agent_state_manager()
