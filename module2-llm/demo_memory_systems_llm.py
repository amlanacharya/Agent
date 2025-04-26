"""
Demo for Module 2-LLM: Memory Systems with Groq API
---------------------------
This script demonstrates the functionality of different LLM-enhanced memory systems from Module 2-LLM.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Import the modules to be demonstrated
# Adjust the import path based on how you're running the script
try:
    # When running from the module2-llm directory
    from code.groq_client import GroqClient
    from code.memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem
    from code.vector_store import VectorStore, Document
    from code.retrieval_agent import RetrievalAgent
    from code.knowledge_base import KnowledgeBase
    from code.kb_agent import KnowledgeBaseAssistant
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient
    from module2_llm.code.memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem
    from module2_llm.code.vector_store import VectorStore, Document
    from module2_llm.code.retrieval_agent import RetrievalAgent
    from module2_llm.code.knowledge_base import KnowledgeBase
    from module2_llm.code.kb_agent import KnowledgeBaseAssistant


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


def demo_groq_client() -> None:
    """Demonstrate the GroqClient class."""
    print_subheader("Groq Client Demo")
    
    print("Note: This demo requires a valid Groq API key.")
    print("If you don't have one, the demo will show simulated responses.")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nNo API key found. Using simulated responses.")
        
        # Simulate text generation
        print("\n1. Simulating text generation...")
        prompt = "Explain what a vector database is in simple terms."
        print(f"Prompt: {prompt}")
        print("Response: A vector database is like a smart filing cabinet that stores information based on meaning rather than exact words. It converts text into numbers (vectors) that represent the meaning, so when you ask a question, it can find information that's conceptually similar, not just matching the exact words you used.")
        
        # Simulate embedding generation
        print("\n2. Simulating embedding generation...")
        text = "Vector databases are essential for semantic search."
        print(f"Text: {text}")
        print("Embedding: [0.023, -0.112, 0.043, ...] (1536-dimensional vector)")
    else:
        # Create a client
        client = GroqClient(api_key)
        
        # Generate text
        print("\n1. Generating text with Groq API...")
        prompt = "Explain what a vector database is in simple terms."
        print(f"Prompt: {prompt}")
        response = client.generate_text(prompt)
        print(f"Response: {response}")
        
        # Generate embedding
        print("\n2. Generating embedding with Groq API...")
        text = "Vector databases are essential for semantic search."
        print(f"Text: {text}")
        embedding = client.generate_embedding(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5]}...")
    
    print("\nThe Groq Client provides access to LLM capabilities for text generation and embeddings.")


def demo_llm_working_memory() -> None:
    """Demonstrate the LLM-enhanced WorkingMemory class."""
    print_subheader("LLM-Enhanced Working Memory Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a working memory instance
    memory = WorkingMemory(api_key=api_key)
    
    print("1. Adding items to working memory...")
    memory.add("User asked about the weather in New York.")
    memory.add("User mentioned they're planning a trip this weekend.")
    memory.add("User is concerned about rain affecting their plans.")
    print(f"Current context: {memory.get()}")
    
    print("\n2. Summarizing context with LLM...")
    summary = memory.summarize_context()
    print(f"LLM-generated summary: {summary}")
    
    print("\n3. Clearing working memory...")
    memory.clear()
    print(f"After clearing: {memory.get()}")
    
    print("\nLLM-enhanced working memory provides context summarization capabilities.")


def demo_llm_short_term_memory() -> None:
    """Demonstrate the LLM-enhanced ShortTermMemory class."""
    print_subheader("LLM-Enhanced Short-Term Memory Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a short-term memory with capacity of 5
    memory = ShortTermMemory(capacity=5, api_key=api_key)
    
    print("1. Adding conversation turns to short-term memory...")
    memory.add({"role": "user", "content": "What's the weather like in New York?"})
    memory.add({"role": "assistant", "content": "It's currently 72°F and sunny in New York."})
    memory.add({"role": "user", "content": "Will it rain this weekend?"})
    memory.add({"role": "assistant", "content": "There's a 30% chance of rain on Saturday, but Sunday should be clear."})
    memory.add({"role": "user", "content": "I'm planning to visit Central Park. Is that a good idea?"})
    
    print(f"All items: {memory.get_all()}")
    
    print("\n2. Extracting key information with LLM...")
    key_info = memory.extract_key_information("weather")
    print(f"Key information about weather: {key_info}")
    
    print("\n3. Summarizing recent conversation with LLM...")
    summary = memory.summarize_recent(3)
    print(f"Summary of last 3 turns: {summary}")
    
    print("\nLLM-enhanced short-term memory provides information extraction and summarization.")


def demo_llm_long_term_memory() -> None:
    """Demonstrate the LLM-enhanced LongTermMemory class."""
    print_subheader("LLM-Enhanced Long-Term Memory Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a test directory
    test_dir = "test_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a memory with a test file
    memory_path = os.path.join(test_dir, "test_long_term.json")
    memory = LongTermMemory(storage_path=memory_path, api_key=api_key)
    
    print("1. Extracting facts from text with LLM...")
    text = "Alice is a software engineer who lives in San Francisco. She enjoys hiking on weekends and has a golden retriever named Max."
    facts = memory.extract_facts(text)
    print(f"Extracted facts: {facts}")
    
    print("\n2. Storing information with metadata...")
    memory.store("user_profiles", "alice", {
        "name": "Alice",
        "occupation": "Software Engineer",
        "location": "San Francisco",
        "hobbies": ["hiking"],
        "pets": [{"type": "dog", "breed": "golden retriever", "name": "Max"}]
    })
    
    print("\n3. Semantic search with LLM-generated embeddings...")
    search_results = memory.semantic_search("people who like outdoor activities", top_k=1)
    print(f"Semantic search results: {search_results}")
    
    print("\nLLM-enhanced long-term memory provides fact extraction and semantic search capabilities.")
    
    # Clean up
    if os.path.exists(memory_path):
        os.remove(memory_path)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)


def demo_llm_episodic_memory() -> None:
    """Demonstrate the LLM-enhanced EpisodicMemory class."""
    print_subheader("LLM-Enhanced Episodic Memory Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create an episodic memory
    memory = EpisodicMemory(api_key=api_key)
    
    print("1. Adding episodes (interaction sequences)...")
    # Add a weather conversation episode
    memory.add_episode({
        "id": "weather-convo-1",
        "timestamp": datetime.now().isoformat(),
        "topic": "weather",
        "interactions": [
            {"role": "user", "content": "What's the weather like today?"},
            {"role": "assistant", "content": "It's sunny and 75°F."},
            {"role": "user", "content": "Will it rain this weekend?"},
            {"role": "assistant", "content": "There's a 30% chance of rain on Saturday."}
        ]
    })
    
    # Add a task management episode
    memory.add_episode({
        "id": "task-convo-1",
        "timestamp": datetime.now().isoformat(),
        "topic": "tasks",
        "interactions": [
            {"role": "user", "content": "Remind me to call John tomorrow."},
            {"role": "assistant", "content": "I've added a reminder to call John tomorrow."},
            {"role": "user", "content": "Also add a grocery shopping task for Saturday."},
            {"role": "assistant", "content": "Added grocery shopping to your tasks for Saturday."}
        ]
    })
    
    print("\n2. Analyzing episodes with LLM...")
    analysis = memory.analyze_episode("weather-convo-1")
    print(f"Analysis of weather conversation: {analysis}")
    
    print("\n3. Finding relevant episodes with LLM...")
    relevant_episodes = memory.find_relevant_episodes("weekend plans")
    print(f"Episodes relevant to 'weekend plans': {[e['id'] for e in relevant_episodes]}")
    
    print("\nLLM-enhanced episodic memory provides conversation analysis and relevance matching.")


def demo_llm_agent_memory_system() -> None:
    """Demonstrate the LLM-enhanced AgentMemorySystem class."""
    print_subheader("LLM-Enhanced Agent Memory System Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a test directory
    test_dir = "test_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create an agent memory system
    memory_system = AgentMemorySystem(storage_dir=test_dir, api_key=api_key)
    
    print("1. Adding interactions...")
    memory_system.add_interaction("user", "My name is Alice and I live in San Francisco.")
    memory_system.add_interaction("assistant", "Nice to meet you, Alice! How do you like living in San Francisco?")
    memory_system.add_interaction("user", "I love it here, especially the parks and the food scene.")
    memory_system.add_interaction("assistant", "San Francisco does have amazing parks and restaurants!")
    
    print("\n2. Getting enhanced context with LLM...")
    context = memory_system.get_enhanced_context("user preferences")
    print(f"Enhanced context for 'user preferences': {context}")
    
    print("\n3. Extracting entities with LLM...")
    entities = memory_system.extract_entities()
    print(f"Extracted entities: {entities}")
    
    print("\nThe LLM-enhanced Agent Memory System integrates all memory types with LLM capabilities.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def demo_llm_vector_store() -> None:
    """Demonstrate the LLM-enhanced VectorStore class."""
    print_subheader("LLM-Enhanced Vector Store Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a vector store
    vector_db = VectorStore(api_key=api_key)
    
    print("1. Adding documents with real embeddings...")
    vector_db.add_document("doc1", "Python is a programming language with simple syntax and powerful libraries.", {"type": "language"})
    vector_db.add_document("doc2", "JavaScript is used for web development and runs in browsers.", {"type": "language"})
    vector_db.add_document("doc3", "Machine learning is a subset of artificial intelligence focused on data-driven algorithms.", {"type": "concept"})
    vector_db.add_document("doc4", "Python is widely used in data science and machine learning applications.", {"type": "application"})
    
    print("\n2. Semantic search with real embeddings...")
    results = vector_db.search("programming languages for AI", top_k=2)
    print("Results for 'programming languages for AI':")
    for doc_id, score in results:
        doc = vector_db.get_document(doc_id)
        print(f"  - {doc.text} (Score: {score:.2f})")
    
    print("\n3. Hybrid search with keyword boosting...")
    results = vector_db.hybrid_search("Python applications", top_k=2)
    print("Results for hybrid search 'Python applications':")
    for doc_id, score in results:
        doc = vector_db.get_document(doc_id)
        print(f"  - {doc.text} (Score: {score:.2f})")
    
    print("\nLLM-enhanced vector stores use real embeddings for more accurate semantic search.")


def demo_llm_retrieval_agent() -> None:
    """Demonstrate the LLM-enhanced RetrievalAgent class."""
    print_subheader("LLM-Enhanced Retrieval Agent Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a vector store and add some documents
    vector_db = VectorStore(api_key=api_key)
    vector_db.add_document("fact1", "The Earth orbits around the Sun at an average distance of 93 million miles.", {"category": "astronomy"})
    vector_db.add_document("fact2", "Water boils at 100 degrees Celsius at sea level, but at lower temperatures at higher altitudes.", {"category": "physics"})
    vector_db.add_document("fact3", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into oxygen and glucose.", {"category": "biology"})
    vector_db.add_document("fact4", "The Moon orbits around the Earth and is about 238,855 miles away on average.", {"category": "astronomy"})
    
    # Create a retrieval agent
    agent = RetrievalAgent(vector_db, api_key=api_key)
    
    print("1. Processing a query with LLM-enhanced retrieval...")
    response = agent.process_query("Tell me about celestial bodies in our solar system.")
    print(f"Query: Tell me about celestial bodies in our solar system.")
    print(f"Response: {response}")
    
    print("\n2. Query expansion with LLM...")
    expanded_query = agent.expand_query("How do plants make food?")
    print(f"Original query: How do plants make food?")
    print(f"Expanded query: {expanded_query}")
    
    print("\nThe LLM-enhanced Retrieval Agent uses query expansion and reranking for better results.")


def demo_llm_knowledge_base() -> None:
    """Demonstrate the LLM-enhanced KnowledgeBase class."""
    print_subheader("LLM-Enhanced Knowledge Base Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a test directory
    test_dir = "test_kb"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base
    kb = KnowledgeBase(storage_dir=test_dir, api_key=api_key)
    
    print("1. Adding facts with LLM-generated embeddings...")
    kb.add_fact("The Earth is the third planet from the Sun and the only astronomical object known to harbor life.", {"category": "astronomy"})
    kb.add_fact("Python was created by Guido van Rossum and first released in 1991 as a successor to the ABC programming language.", {"category": "programming"})
    kb.add_fact("The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the Earth's surface.", {"category": "geography"})
    
    print("\n2. Semantic search with LLM-generated embeddings...")
    results = kb.search("planets with life", top_k=1)
    print("Results for 'planets with life':")
    for fact, score in results:
        print(f"  - {fact} (Score: {score:.2f})")
    
    print("\n3. Fact verification with LLM...")
    verification = kb.verify_fact("The Earth is the fourth planet from the Sun.")
    print(f"Verification result: {verification}")
    
    print("\n4. Fact generation with LLM...")
    new_facts = kb.generate_related_facts("Python programming language", num_facts=2)
    print("Generated related facts about Python:")
    for fact in new_facts:
        print(f"  - {fact}")
    
    print("\nThe LLM-enhanced Knowledge Base adds fact verification and generation capabilities.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def demo_llm_kb_assistant() -> None:
    """Demonstrate the LLM-enhanced KnowledgeBaseAssistant class."""
    print_subheader("LLM-Enhanced Knowledge Base Assistant Demo")
    
    # Check if API key is available
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No API key found. Using simulated responses.")
    
    # Create a test directory
    test_dir = "test_kb_assistant"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(storage_dir=test_dir, api_key=api_key)
    
    print("1. Adding knowledge with LLM processing...")
    assistant.add_knowledge("The Earth is approximately 4.5 billion years old, having formed from the solar nebula along with the Sun and other planets.")
    assistant.add_knowledge("The human brain contains approximately 86 billion neurons, which are specialized cells that transmit information throughout the body.")
    assistant.add_knowledge("The Great Wall of China is over 13,000 miles long and was built over multiple dynasties, with the most famous sections built during the Ming Dynasty.")
    
    print("\n2. Asking questions with LLM-enhanced retrieval...")
    questions = [
        "How old is the Earth?",
        "Tell me about the human brain.",
        "What is the longest structure built by humans?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = assistant.answer_question(question)
        print(f"A: {answer}")
    
    print("\n3. Conversation with context and LLM reasoning...")
    conversation = [
        ("user", "What's the oldest planet in our solar system?"),
        ("assistant", "Based on current scientific understanding, all planets in our solar system formed around the same time, about 4.6 billion years ago. The Earth is approximately 4.5 billion years old, having formed from the solar nebula along with the Sun and other planets."),
        ("user", "Tell me more about Earth's formation.")
    ]
    
    for role, message in conversation:
        print(f"\n{role.capitalize()}: {message}")
        if role == "user":
            response = assistant.process_message(message)
            print(f"Assistant: {response}")
    
    print("\nThe LLM-enhanced Knowledge Base Assistant provides more sophisticated responses with reasoning.")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def run_all_demos() -> None:
    """Run all demonstration functions."""
    print_header("DEMONSTRATION OF MODULE 2-LLM: MEMORY SYSTEMS WITH GROQ API")
    print("This script demonstrates the functionality of different LLM-enhanced memory systems.")
    print("Follow along to see how each component works with real LLM integration.")
    
    # Run each demo with a pause between
    demo_groq_client()
    time.sleep(1)
    
    demo_llm_working_memory()
    time.sleep(1)
    
    demo_llm_short_term_memory()
    time.sleep(1)
    
    demo_llm_long_term_memory()
    time.sleep(1)
    
    demo_llm_episodic_memory()
    time.sleep(1)
    
    demo_llm_agent_memory_system()
    time.sleep(1)
    
    demo_llm_vector_store()
    time.sleep(1)
    
    demo_llm_retrieval_agent()
    time.sleep(1)
    
    demo_llm_knowledge_base()
    time.sleep(1)
    
    demo_llm_kb_assistant()
    
    print_header("DEMONSTRATION COMPLETE")
    print("You've now seen all the key components of the LLM-enhanced memory systems in Module 2-LLM.")
    print("For more details, refer to the documentation and source code.")


def main():
    """Main function to run the demo with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Module 2-LLM memory systems with Groq API")
    parser.add_argument("component", nargs="?", choices=[
        "groq", "working", "short", "long", "episodic", "system", 
        "vector", "retrieval", "kb", "assistant", "all"
    ], default="all", help="Component to demonstrate")
    
    args = parser.parse_args()
    
    if args.component == "groq":
        print_header("GROQ CLIENT DEMO")
        demo_groq_client()
    elif args.component == "working":
        print_header("LLM-ENHANCED WORKING MEMORY DEMO")
        demo_llm_working_memory()
    elif args.component == "short":
        print_header("LLM-ENHANCED SHORT-TERM MEMORY DEMO")
        demo_llm_short_term_memory()
    elif args.component == "long":
        print_header("LLM-ENHANCED LONG-TERM MEMORY DEMO")
        demo_llm_long_term_memory()
    elif args.component == "episodic":
        print_header("LLM-ENHANCED EPISODIC MEMORY DEMO")
        demo_llm_episodic_memory()
    elif args.component == "system":
        print_header("LLM-ENHANCED AGENT MEMORY SYSTEM DEMO")
        demo_llm_agent_memory_system()
    elif args.component == "vector":
        print_header("LLM-ENHANCED VECTOR STORE DEMO")
        demo_llm_vector_store()
    elif args.component == "retrieval":
        print_header("LLM-ENHANCED RETRIEVAL AGENT DEMO")
        demo_llm_retrieval_agent()
    elif args.component == "kb":
        print_header("LLM-ENHANCED KNOWLEDGE BASE DEMO")
        demo_llm_knowledge_base()
    elif args.component == "assistant":
        print_header("LLM-ENHANCED KNOWLEDGE BASE ASSISTANT DEMO")
        demo_llm_kb_assistant()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
