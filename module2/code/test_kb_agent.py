"""
Test Script for Knowledge Base Assistant
---------------------------------------
This script demonstrates the functionality of the knowledge base assistant,
including question answering, learning from statements, and handling uncertainty.
"""

import os
import shutil
import time

from module2.code.knowledge_base import KnowledgeBase, CitationManager, UncertaintyHandler
from module2.code.kb_agent import KnowledgeBaseAssistant


def print_separator(title):
    """Print a separator with a title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def test_knowledge_base():
    """Test the knowledge base functionality"""
    print_separator("Testing Knowledge Base")
    
    # Create a test directory
    test_dir = "test_kb"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Initialize knowledge base
    kb = KnowledgeBase(test_dir)
    print(f"Created knowledge base: {kb}")
    
    # Add knowledge
    print("\nAdding knowledge...")
    k1 = kb.add_knowledge(
        "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        {"source": "Wikipedia", "confidence": 1.0, "tags": ["AI", "definition"]}
    )
    print(f"Added knowledge with ID: {k1}")
    
    k2 = kb.add_knowledge(
        "Machine learning is a subset of AI that focuses on the ability of machines to learn from data.",
        {"source": "AI Textbook", "confidence": 0.95, "tags": ["AI", "machine learning"]}
    )
    print(f"Added knowledge with ID: {k2}")
    
    k3 = kb.add_knowledge(
        "Deep learning is a type of machine learning based on artificial neural networks.",
        {"source": "Research Paper", "confidence": 0.9, "tags": ["AI", "deep learning"]}
    )
    print(f"Added knowledge with ID: {k3}")
    
    # Add relationships
    print("\nAdding relationships...")
    kb.add_relationship(k1, k2, "has_subset")
    kb.add_relationship(k2, k3, "has_technique")
    print("Added relationships between knowledge entries")
    
    # Retrieve knowledge
    print("\nRetrieving knowledge...")
    results = kb.retrieve_knowledge("What is artificial intelligence?")
    print("Query: What is artificial intelligence?")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text']} (Similarity: {result['similarity']:.2f}, Confidence: {result['confidence']})")
    
    # Get related knowledge
    print("\nGetting related knowledge...")
    related = kb.get_related_knowledge(k1)
    print(f"Knowledge related to '{kb.get_knowledge(k1)['text']}':")
    for i, rel in enumerate(related):
        print(f"{i+1}. {rel['text']} (Relationship: {rel['relationship']})")
    
    # Get knowledge by tag
    print("\nGetting knowledge by tag...")
    tag_results = kb.get_knowledge_by_tag("machine learning")
    print("Knowledge with tag 'machine learning':")
    for i, result in enumerate(tag_results):
        print(f"{i+1}. {result['text']}")
    
    # Save and load
    print("\nSaving and loading knowledge base...")
    kb.save()
    print("Knowledge base saved")
    
    # Create a new instance and load
    kb2 = KnowledgeBase(test_dir)
    print(f"Loaded knowledge base: {kb2}")
    
    # Verify loaded data
    results2 = kb2.retrieve_knowledge("What is artificial intelligence?")
    print("Query after loading: What is artificial intelligence?")
    for i, result in enumerate(results2):
        print(f"{i+1}. {result['text']} (Similarity: {result['similarity']:.2f}, Confidence: {result['confidence']})")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\nTest directory cleaned up")


def test_citation_manager():
    """Test the citation manager functionality"""
    print_separator("Testing Citation Manager")
    
    # Create a test knowledge base
    test_dir = "test_citations"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    kb = KnowledgeBase(test_dir)
    
    # Add some knowledge
    k1 = kb.add_knowledge(
        "The Earth is the third planet from the Sun.",
        {"source": "NASA", "confidence": 1.0}
    )
    
    k2 = kb.add_knowledge(
        "The Earth orbits the Sun at an average distance of 149.6 million kilometers.",
        {"source": "https://solarsystem.nasa.gov/planets/earth/overview/", "confidence": 0.98}
    )
    
    # Create citation manager
    citation_manager = CitationManager(kb)
    
    # Test different citation styles
    knowledge = kb.get_knowledge(k1)
    print("Knowledge:", knowledge['text'])
    
    print("\nStandard citation:")
    print(citation_manager.format_citation(knowledge, "standard"))
    
    print("\nAcademic citation:")
    print(citation_manager.format_citation(knowledge, "academic"))
    
    # Test URL citation with the second knowledge entry
    knowledge2 = kb.get_knowledge(k2)
    print("\nURL citation:")
    print(citation_manager.format_citation(knowledge2, "url"))
    
    # Test adding citations to a response
    response = "The Earth is the third planet from the Sun and orbits at an average distance of 149.6 million kilometers."
    knowledge_entries = [knowledge, knowledge2]
    
    print("\nResponse with citations:")
    cited_response = citation_manager.add_citations_to_response(response, knowledge_entries)
    print(cited_response)
    
    # Clean up
    shutil.rmtree(test_dir)


def test_uncertainty_handler():
    """Test the uncertainty handler functionality"""
    print_separator("Testing Uncertainty Handler")
    
    # Create uncertainty handler
    uncertainty_handler = UncertaintyHandler(confidence_threshold=0.7)
    
    # Test with different confidence levels
    print("Testing with different confidence levels:")
    
    # High confidence
    high_confidence_results = [
        {"text": "The Earth is the third planet from the Sun.", "similarity": 0.95, "confidence": 0.9}
    ]
    print("\nHigh confidence (0.86):")
    print(uncertainty_handler.generate_response("What is Earth?", high_confidence_results))
    
    # Medium confidence
    medium_confidence_results = [
        {"text": "The Earth might be the third planet from the Sun.", "similarity": 0.8, "confidence": 0.7}
    ]
    print("\nMedium confidence (0.56):")
    print(uncertainty_handler.generate_response("What is Earth?", medium_confidence_results))
    
    # Low confidence
    low_confidence_results = [
        {"text": "The Earth is possibly a planet in the Solar System.", "similarity": 0.6, "confidence": 0.5}
    ]
    print("\nLow confidence (0.3):")
    print(uncertainty_handler.generate_response("What is Earth?", low_confidence_results))
    
    # No results
    print("\nNo results:")
    print(uncertainty_handler.generate_response("What is Zorblax?", []))
    
    # Test clarification requests
    print("\nTesting clarification requests:")
    
    # Should ask for clarification
    print("\nShould ask for clarification:")
    print(uncertainty_handler.should_ask_clarification("What is a quasar?", low_confidence_results))
    print(uncertainty_handler.generate_clarification_request("What is a quasar?", low_confidence_results))
    
    # Should not ask for clarification (high confidence)
    print("\nShould not ask for clarification (high confidence):")
    print(uncertainty_handler.should_ask_clarification("What is Earth?", high_confidence_results))
    
    # Should not ask for clarification (no results)
    print("\nShould not ask for clarification (no results):")
    print(uncertainty_handler.should_ask_clarification("What is Zorblax?", []))
    print(uncertainty_handler.generate_clarification_request("What is Zorblax?", []))


def test_kb_assistant():
    """Test the knowledge base assistant functionality"""
    print_separator("Testing Knowledge Base Assistant")
    
    # Create a test directory
    test_dir = "test_assistant"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    print(f"Created assistant: {assistant}")
    
    # Check initial settings
    print("\nInitial settings:")
    print(assistant.get_settings())
    
    # Update settings
    print("\nUpdating settings...")
    assistant.update_settings({
        "citation_style": "academic",
        "learning_mode": "active"
    })
    print("Updated settings:")
    print(assistant.get_settings())
    
    # Add some initial knowledge
    print("\nAdding initial knowledge...")
    assistant.learn_from_statement(
        "The Sun is a G-type main-sequence star at the center of the Solar System.",
        source="Astronomy Textbook",
        confidence=0.95
    )
    
    assistant.learn_from_statement(
        "The Solar System consists of the Sun and the objects that orbit it, including planets, moons, asteroids, and comets.",
        source="NASA",
        confidence=0.98
    )
    
    # Answer questions
    print("\nAnswering questions...")
    
    questions = [
        "What is the Sun?",
        "What is in the Solar System?",
        "What is Jupiter?",  # Unknown information
        "Is the Sun hot?"     # Uncertain information
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = assistant.answer_question(question)
        print(f"Response: {response}")
    
    # Test learning from statements
    print("\nLearning from statements...")
    
    statements = [
        "Jupiter is the largest planet in the Solar System.",
        "The Sun's surface temperature is about 5,500 degrees Celsius."
    ]
    
    for statement in statements:
        print(f"\nStatement: {statement}")
        response = assistant.learn_from_statement(statement)
        print(f"Response: {response}")
    
    # Test answering questions after learning
    print("\nAnswering questions after learning...")
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = assistant.answer_question(question)
        print(f"Response: {response}")
    
    # Test conversation history
    print("\nGetting conversation history...")
    history = assistant.get_conversation_history()
    print(f"Conversation history has {len(history)} entries")
    
    # Save the assistant state
    print("\nSaving assistant state...")
    assistant.save()
    print("Assistant state saved")
    
    # Create a new instance and verify persistence
    print("\nCreating new assistant instance...")
    assistant2 = KnowledgeBaseAssistant(test_dir)
    print(f"New assistant: {assistant2}")
    
    # Verify settings were loaded
    print("\nVerifying settings were loaded:")
    print(assistant2.get_settings())
    
    # Verify knowledge was loaded
    print("\nVerifying knowledge was loaded...")
    response = assistant2.answer_question("What is Jupiter?")
    print(f"Question: What is Jupiter?")
    print(f"Response: {response}")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\nTest directory cleaned up")


def interactive_demo():
    """Run an interactive demo of the knowledge base assistant"""
    print_separator("Interactive Knowledge Base Assistant Demo")
    
    # Create a demo directory
    demo_dir = "demo_assistant"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    
    # Initialize assistant
    assistant = KnowledgeBaseAssistant(demo_dir)
    
    # Add some initial knowledge
    assistant.learn_from_statement(
        "Python is a high-level, interpreted programming language known for its readability.",
        source="Python Documentation",
        confidence=1.0
    )
    
    assistant.learn_from_statement(
        "Python was created by Guido van Rossum and first released in 1991.",
        source="Python History",
        confidence=0.95
    )
    
    assistant.learn_from_statement(
        "JavaScript is a programming language commonly used for web development.",
        source="MDN Web Docs",
        confidence=0.9
    )
    
    print("Welcome to the Knowledge Base Assistant Demo!")
    print("You can ask questions, make statements, or type 'exit' to quit.")
    print("The assistant will learn from your statements and answer your questions.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ("exit", "quit", "bye"):
            break
        
        if user_input.lower() == "settings":
            print("\nCurrent settings:")
            print(assistant.get_settings())
            continue
        
        if user_input.lower() == "history":
            print("\nConversation history:")
            history = assistant.get_conversation_history()
            for i, entry in enumerate(history):
                role = entry["role"]
                content = entry["content"]
                print(f"{i+1}. {role.capitalize()}: {content}")
            continue
        
        # Process the input
        response = assistant.process_input(user_input)
        print(f"\nAssistant: {response}")
    
    # Save the assistant state
    assistant.save()
    print("\nThank you for using the Knowledge Base Assistant Demo!")
    print(f"Assistant state saved to {demo_dir}")


if __name__ == "__main__":
    # Run all tests
    test_knowledge_base()
    test_citation_manager()
    test_uncertainty_handler()
    test_kb_assistant()
    
    # Run interactive demo
    interactive_demo()
