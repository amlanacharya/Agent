"""
Demo for Module 2: Memory Systems
---------------------------
This script demonstrates the functionality of different memory systems from Module 2.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Import the modules to be demonstrated
# Adjust the import path based on how you're running the script
try:
    # When running from the module2 directory
    from code.memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem
    from code.vector_store import SimpleVectorDB, Document
    from code.retrieval_agent import RetrievalAgent
    from code.knowledge_base import KnowledgeBase
    from code.kb_agent import KnowledgeBaseAssistant
except ImportError:
    # When running from the project root
    from module2.code.memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem
    from module2.code.vector_store import SimpleVectorDB, Document
    from module2.code.retrieval_agent import RetrievalAgent
    from module2.code.knowledge_base import KnowledgeBase
    from module2.code.kb_agent import KnowledgeBaseAssistant


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def demo_working_memory() -> None:
    """Demonstrate the WorkingMemory class."""
    print_subheader("Working Memory Demo")
    
    # Create a working memory instance
    memory = WorkingMemory()
    
    print("1. Adding items to working memory...")
    memory.add("User asked about the weather.")
    print(f"Current context: {memory.get()}")
    
    print("\n2. Adding more context...")
    memory.add("User is planning a trip this weekend.")
    print(f"Current context: {memory.get()}")
    
    print("\n3. Clearing working memory...")
    memory.clear()
    print(f"After clearing: {memory.get()}")
    
    print("\nWorking memory provides immediate context for the current interaction.")


def demo_short_term_memory() -> None:
    """Demonstrate the ShortTermMemory class."""
    print_subheader("Short-Term Memory Demo")
    
    # Create a short-term memory with capacity of 3
    memory = ShortTermMemory(capacity=3)
    
    print("1. Adding items to short-term memory...")
    memory.add("User asked about the weather.")
    memory.add("Agent provided weather forecast.")
    memory.add("User asked about weekend activities.")
    
    print(f"All items: {memory.get_all()}")
    print(f"Most recent 2 items: {memory.get_recent(2)}")
    
    print("\n2. Adding one more item (should remove oldest due to capacity)...")
    memory.add("Agent suggested outdoor activities.")
    
    print(f"All items after adding 4th item: {memory.get_all()}")
    print("Notice the first item was removed to maintain capacity of 3.")
    
    print("\nShort-term memory maintains recent interactions with a capacity limit.")


def demo_long_term_memory() -> None:
    """Demonstrate the LongTermMemory class."""
    print_subheader("Long-Term Memory Demo")
    
    # Create a test directory
    test_dir = "test_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a memory with a test file
    memory_path = os.path.join(test_dir, "test_long_term.json")
    memory = LongTermMemory(storage_path=memory_path)
    
    print("1. Storing information in different categories...")
    memory.store("user_profiles", "alice", {"name": "Alice", "preferences": {"theme": "dark"}})
    memory.store("frequent_tasks", "weekly_report", {"count": 5, "last_created": "2023-06-01"})
    
    print("2. Retrieving specific information...")
    alice_profile = memory.retrieve("user_profiles", "alice")
    print(f"Alice's profile: {alice_profile}")
    
    print("\n3. Retrieving all information in a category...")
    all_profiles = memory.retrieve("user_profiles")
    print(f"All profiles: {all_profiles}")
    
    print("\n4. Searching for information...")
    search_results = memory.search("alice")
    print(f"Search results for 'alice': {search_results}")
    
    print("\nLong-term memory persists information across sessions.")
    
    # Clean up
    if os.path.exists(memory_path):
        os.remove(memory_path)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)


def demo_episodic_memory() -> None:
    """Demonstrate the EpisodicMemory class."""
    print_subheader("Episodic Memory Demo")
    
    # Create an episodic memory
    memory = EpisodicMemory()
    
    print("1. Adding episodes (interaction sequences)...")
    # Add a weather conversation episode
    memory.add_episode({
        "id": "weather-convo-1",
        "timestamp": datetime.now().isoformat(),
        "topic": "weather",
        "interactions": [
            {"role": "user", "content": "What's the weather like today?"},
            {"role": "agent", "content": "It's sunny and 75°F."},
            {"role": "user", "content": "Will it rain this weekend?"},
            {"role": "agent", "content": "There's a 30% chance of rain on Saturday."}
        ]
    })
    
    # Add a task management episode
    memory.add_episode({
        "id": "task-convo-1",
        "timestamp": datetime.now().isoformat(),
        "topic": "tasks",
        "interactions": [
            {"role": "user", "content": "Remind me to call John tomorrow."},
            {"role": "agent", "content": "I've added a reminder to call John tomorrow."},
            {"role": "user", "content": "Also add a grocery shopping task for Saturday."},
            {"role": "agent", "content": "Added grocery shopping to your tasks for Saturday."}
        ]
    })
    
    print("2. Retrieving all episodes...")
    all_episodes = memory.get_episodes()
    print(f"Number of episodes: {len(all_episodes)}")
    
    print("\n3. Retrieving episodes by topic...")
    weather_episodes = memory.get_episodes(lambda e: e.get("topic") == "weather")
    print(f"Weather episodes: {len(weather_episodes)}")
    
    print("\n4. Getting a specific episode...")
    task_episode = memory.get_episode_by_id("task-convo-1")
    if task_episode:
        print(f"Task episode interactions: {len(task_episode['interactions'])}")
        print(f"First interaction: {task_episode['interactions'][0]}")
    
    print("\nEpisodic memory stores complete interaction sequences for future reference.")


def demo_agent_memory_system() -> None:
    """Demonstrate the AgentMemorySystem class."""
    print_subheader("Agent Memory System Demo")
    
    # Create a test directory
    test_dir = "test_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create an agent memory system
    memory_system = AgentMemorySystem(storage_dir=test_dir)
    
    print("1. Adding interactions...")
    memory_system.add_interaction("user", "My name is Alice.")
    memory_system.add_interaction("agent", "Nice to meet you, Alice!")
    memory_system.add_interaction("user", "I prefer dark mode in applications.")
    memory_system.add_interaction("agent", "I'll remember your preference for dark mode.")
    
    print("\n2. Getting current context...")
    context = memory_system.get_context()
    print(f"Current context: {context}")
    
    print("\n3. Storing knowledge...")
    memory_system.store_knowledge("user_preferences", "theme", "dark")
    memory_system.store_knowledge("user_info", "name", "Alice")
    
    print("\n4. Retrieving knowledge...")
    theme = memory_system.retrieve_knowledge("user_preferences", "theme")
    print(f"User theme preference: {theme}")
    
    print("\n5. Searching knowledge...")
    search_results = memory_system.search_knowledge("Alice")
    print(f"Search results for 'Alice': {search_results}")
    
    print("\nThe Agent Memory System integrates all memory types into a unified system.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def demo_vector_store() -> None:
    """Demonstrate the SimpleVectorDB class."""
    print_subheader("Vector Store Demo")
    
    # Create a vector store
    vector_db = SimpleVectorDB()
    
    print("1. Adding documents to the vector store...")
    vector_db.add_document("doc1", "Python is a programming language.", {"type": "language"})
    vector_db.add_document("doc2", "JavaScript is used for web development.", {"type": "language"})
    vector_db.add_document("doc3", "Machine learning is a subset of AI.", {"type": "concept"})
    vector_db.add_document("doc4", "Python is widely used in data science.", {"type": "application"})
    
    print("\n2. Searching for similar documents...")
    results = vector_db.search("programming languages", top_k=2)
    print("Results for 'programming languages':")
    for doc_id, score in results:
        doc = vector_db.get_document(doc_id)
        print(f"  - {doc.text} (Score: {score:.2f})")
    
    print("\n3. Filtering by metadata...")
    language_docs = vector_db.filter_documents(lambda meta: meta.get("type") == "language")
    print("Documents about languages:")
    for doc_id in language_docs:
        doc = vector_db.get_document(doc_id)
        print(f"  - {doc.text}")
    
    print("\nVector stores enable semantic search based on meaning rather than exact matches.")


def demo_retrieval_agent() -> None:
    """Demonstrate the RetrievalAgent class."""
    print_subheader("Retrieval Agent Demo")
    
    # Create a vector store and add some documents
    vector_db = SimpleVectorDB()
    vector_db.add_document("fact1", "The Earth orbits around the Sun.", {"category": "astronomy"})
    vector_db.add_document("fact2", "Water boils at 100 degrees Celsius at sea level.", {"category": "physics"})
    vector_db.add_document("fact3", "Photosynthesis is how plants make food from sunlight.", {"category": "biology"})
    vector_db.add_document("fact4", "The Moon orbits around the Earth.", {"category": "astronomy"})
    
    # Create a retrieval agent
    agent = RetrievalAgent(vector_db)
    
    print("1. Processing a query...")
    response = agent.process_query("Tell me about space and planets.")
    print(f"Query: Tell me about space and planets.")
    print(f"Response: {response}")
    
    print("\n2. Processing another query...")
    response = agent.process_query("How do plants get energy?")
    print(f"Query: How do plants get energy?")
    print(f"Response: {response}")
    
    print("\nThe Retrieval Agent uses the vector store to find relevant information for queries.")


def demo_knowledge_base() -> None:
    """Demonstrate the KnowledgeBase class."""
    print_subheader("Knowledge Base Demo")
    
    # Create a test directory
    test_dir = "test_kb"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base
    kb = KnowledgeBase(storage_dir=test_dir)
    
    print("1. Adding facts to the knowledge base...")
    kb.add_fact("The Earth is the third planet from the Sun.", {"category": "astronomy"})
    kb.add_fact("Python was created by Guido van Rossum.", {"category": "programming"})
    kb.add_fact("The Pacific Ocean is the largest ocean on Earth.", {"category": "geography"})
    
    print("\n2. Searching the knowledge base...")
    results = kb.search("planet Earth", top_k=2)
    print("Results for 'planet Earth':")
    for fact, score in results:
        print(f"  - {fact} (Score: {score:.2f})")
    
    print("\n3. Getting facts by category...")
    astronomy_facts = kb.get_facts_by_category("astronomy")
    print("Astronomy facts:")
    for fact in astronomy_facts:
        print(f"  - {fact}")
    
    print("\n4. Saving and loading the knowledge base...")
    kb.save()
    
    # Create a new knowledge base and load the saved data
    kb2 = KnowledgeBase(storage_dir=test_dir)
    kb2.load()
    
    print("\nFacts after loading:")
    all_facts = kb2.get_all_facts()
    for fact in all_facts:
        print(f"  - {fact}")
    
    print("\nThe Knowledge Base combines vector search with metadata filtering for fact retrieval.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def demo_kb_assistant() -> None:
    """Demonstrate the KnowledgeBaseAssistant class."""
    print_subheader("Knowledge Base Assistant Demo")
    
    # Create a test directory
    test_dir = "test_kb_assistant"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(storage_dir=test_dir)
    
    print("1. Adding knowledge to the assistant...")
    assistant.add_knowledge("The Earth is approximately 4.5 billion years old.")
    assistant.add_knowledge("The human brain contains approximately 86 billion neurons.")
    assistant.add_knowledge("The Great Wall of China is over 13,000 miles long.")
    
    print("\n2. Asking questions...")
    questions = [
        "How old is the Earth?",
        "Tell me about the human brain.",
        "What is the longest structure built by humans?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = assistant.answer_question(question)
        print(f"A: {answer}")
    
    print("\n3. Conversation with context...")
    conversation = [
        ("user", "What's the oldest planet in our solar system?"),
        ("assistant", "Based on current scientific understanding, all planets in our solar system formed around the same time, about 4.6 billion years ago. The Earth is approximately 4.5 billion years old."),
        ("user", "Tell me more about Earth's age.")
    ]
    
    for role, message in conversation:
        print(f"\n{role.capitalize()}: {message}")
        if role == "user":
            response = assistant.process_message(message)
            print(f"Assistant: {response}")
    
    print("\nThe Knowledge Base Assistant combines all components into a complete system.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def run_all_demos() -> None:
    """Run all demonstration functions."""
    print_header("DEMONSTRATION OF MODULE 2: MEMORY SYSTEMS")
    print("This script demonstrates the functionality of different memory systems.")
    print("Follow along to see how each component works and how they can be used together.")
    
    # Run each demo with a pause between
    demo_working_memory()
    time.sleep(1)
    
    demo_short_term_memory()
    time.sleep(1)
    
    demo_long_term_memory()
    time.sleep(1)
    
    demo_episodic_memory()
    time.sleep(1)
    
    demo_agent_memory_system()
    time.sleep(1)
    
    demo_vector_store()
    time.sleep(1)
    
    demo_retrieval_agent()
    time.sleep(1)
    
    demo_knowledge_base()
    time.sleep(1)
    
    demo_kb_assistant()
    
    print_header("DEMONSTRATION COMPLETE")
    print("You've now seen all the key components of the memory systems in Module 2.")
    print("For more details, refer to the documentation and source code.")


def main():
    """Main function to run the demo with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Module 2 memory systems")
    parser.add_argument("component", nargs="?", choices=[
        "working", "short", "long", "episodic", "system", 
        "vector", "retrieval", "kb", "assistant", "all"
    ], default="all", help="Component to demonstrate")
    
    args = parser.parse_args()
    
    if args.component == "working":
        print_header("WORKING MEMORY DEMO")
        demo_working_memory()
    elif args.component == "short":
        print_header("SHORT-TERM MEMORY DEMO")
        demo_short_term_memory()
    elif args.component == "long":
        print_header("LONG-TERM MEMORY DEMO")
        demo_long_term_memory()
    elif args.component == "episodic":
        print_header("EPISODIC MEMORY DEMO")
        demo_episodic_memory()
    elif args.component == "system":
        print_header("AGENT MEMORY SYSTEM DEMO")
        demo_agent_memory_system()
    elif args.component == "vector":
        print_header("VECTOR STORE DEMO")
        demo_vector_store()
    elif args.component == "retrieval":
        print_header("RETRIEVAL AGENT DEMO")
        demo_retrieval_agent()
    elif args.component == "kb":
        print_header("KNOWLEDGE BASE DEMO")
        demo_knowledge_base()
    elif args.component == "assistant":
        print_header("KNOWLEDGE BASE ASSISTANT DEMO")
        demo_kb_assistant()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
