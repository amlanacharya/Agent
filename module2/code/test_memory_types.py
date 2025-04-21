"""
Test Script for Memory Types
---------------------------
This script demonstrates and tests the different memory types
implemented in memory_types.py.
"""

import os
import time
from memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem

def test_working_memory():
    """Test working memory functionality"""
    print("\n=== Testing Working Memory ===")
    
    # Create working memory
    memory = WorkingMemory()
    print("Initial state:", memory)
    
    # Set context
    context = {"user": "Alice", "query": "What's the weather today?"}
    memory.set_context(context)
    print("After setting context:", memory)
    
    # Get context
    retrieved_context = memory.get_context()
    print("Retrieved context:", retrieved_context)
    
    # Clear memory
    memory.clear()
    print("After clearing:", memory)
    
    print("Working memory test completed.")

def test_short_term_memory():
    """Test short-term memory functionality"""
    print("\n=== Testing Short-Term Memory ===")
    
    # Create short-term memory with capacity of 3
    memory = ShortTermMemory(capacity=3)
    print("Initial state:", memory)
    
    # Add items
    memory.add("First item")
    memory.add("Second item")
    print("After adding 2 items:", memory)
    print("Memory contents:", memory.get_recent())
    
    # Add more items to test capacity limit
    memory.add("Third item")
    memory.add("Fourth item")  # This should push out "First item"
    print("After adding 2 more items:", memory)
    print("Memory contents:", memory.get_recent())
    
    # Test get_recent with limit
    print("Most recent 2 items:", memory.get_recent(2))
    
    # Clear memory
    memory.clear()
    print("After clearing:", memory)
    
    print("Short-term memory test completed.")

def test_long_term_memory():
    """Test long-term memory functionality"""
    print("\n=== Testing Long-Term Memory ===")
    
    # Create a test file path
    test_file = "test_long_term_memory.json"
    
    # Remove the test file if it exists
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Create long-term memory
    memory = LongTermMemory(file_path=test_file)
    print("Initial state:", memory)
    
    # Store items
    memory.store("user_name", "Alice")
    memory.store("preferences", {"theme": "dark", "notifications": True})
    print("After storing items:", memory)
    
    # Retrieve items
    user_name = memory.retrieve("user_name")
    preferences = memory.retrieve("preferences")
    print(f"Retrieved user_name: {user_name}")
    print(f"Retrieved preferences: {preferences}")
    
    # Test non-existent key
    unknown = memory.retrieve("unknown_key")
    print(f"Retrieved unknown_key: {unknown}")
    
    # Forget an item
    memory.forget("user_name")
    print("After forgetting user_name:", memory)
    print(f"Retrieved user_name after forgetting: {memory.retrieve('user_name')}")
    
    # Get all items
    all_items = memory.get_all()
    print("All items:", all_items)
    
    # Clear memory
    memory.clear()
    print("After clearing:", memory)
    
    # Create a new instance to test loading from file
    print("Creating new instance to test persistence...")
    memory.store("test_persistence", "This should be loaded by the new instance")
    del memory
    
    new_memory = LongTermMemory(file_path=test_file)
    print("New instance state:", new_memory)
    print(f"Retrieved test_persistence: {new_memory.retrieve('test_persistence')}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("Long-term memory test completed.")

def test_episodic_memory():
    """Test episodic memory functionality"""
    print("\n=== Testing Episodic Memory ===")
    
    # Create episodic memory
    memory = EpisodicMemory()
    print("Initial state:", memory)
    
    # Record episodes
    episode1_id = memory.record_episode(
        episode_type="conversation",
        content={"user": "Hello", "agent": "Hi there!"},
        metadata={"sentiment": "positive"}
    )
    
    # Wait a moment to ensure different timestamps
    time.sleep(0.1)
    
    episode2_id = memory.record_episode(
        episode_type="action",
        content={"action": "search", "query": "weather"},
        metadata={"success": True}
    )
    
    time.sleep(0.1)
    
    episode3_id = memory.record_episode(
        episode_type="conversation",
        content={"user": "Thanks", "agent": "You're welcome!"},
        metadata={"sentiment": "positive"}
    )
    
    print(f"Recorded episodes with IDs: {episode1_id}, {episode2_id}, {episode3_id}")
    print("Memory state:", memory)
    
    # Get episode by ID
    episode = memory.get_episode(episode1_id)
    print(f"Episode {episode1_id}:", episode)
    
    # Get episodes by type
    conversation_episodes = memory.get_episodes_by_type("conversation")
    print("Conversation episodes:", conversation_episodes)
    
    # Get episodes in timeframe
    # Get the timestamp from the first episode
    start_time = memory.get_episode(episode1_id)["timestamp"]
    # Get episodes from that time onwards
    recent_episodes = memory.get_episodes_in_timeframe(start_time)
    print("Recent episodes:", recent_episodes)
    
    # Clear memory
    memory.clear()
    print("After clearing:", memory)
    
    print("Episodic memory test completed.")

def test_agent_memory_system():
    """Test the integrated agent memory system"""
    print("\n=== Testing Agent Memory System ===")
    
    # Create a test directory
    test_dir = "test_agent_memory"
    
    # Create agent memory system
    memory = AgentMemorySystem(storage_dir=test_dir)
    print("Initial state:", memory)
    
    # Process interactions
    memory.process_interaction(
        user_input="Hello, my name is Bob.",
        agent_response="Hi Bob! How can I help you today?"
    )
    
    memory.process_interaction(
        user_input="What's the weather like?",
        agent_response="I don't have real-time weather data, but I can help you find that information."
    )
    
    print("After processing interactions:", memory)
    
    # Get conversation context
    context = memory.get_conversation_context()
    print("Conversation context:")
    for turn in context:
        print(f"User: {turn['user_input']}")
        print(f"Agent: {turn['agent_response']}")
        print()
    
    # Store and retrieve facts
    memory.store_fact("user_name", "Bob")
    memory.store_fact("user_interests", ["weather", "news"])
    
    print(f"Retrieved user_name: {memory.retrieve_fact('user_name')}")
    print(f"Retrieved user_interests: {memory.retrieve_fact('user_interests')}")
    
    # Save and get user preferences
    memory.save_user_preference("theme", "dark")
    memory.save_user_preference("notifications", True)
    
    preferences = memory.get_user_preferences()
    print("User preferences:", preferences)
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Agent memory system test completed.")

def run_all_tests():
    """Run all memory tests"""
    print("=== Memory Types Tests ===")
    
    test_working_memory()
    test_short_term_memory()
    test_long_term_memory()
    test_episodic_memory()
    test_agent_memory_system()
    
    print("\nAll memory tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
