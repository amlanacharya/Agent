"""
Test Script for Conversation Memory Analyzer
-------------------------------------------
This script demonstrates and tests the ConversationMemoryAnalyzer class
implemented in conversation_analyzer.py.
"""

import os
import time
import json
from module2.exercises.conversation_analyzer import ConversationMemoryAnalyzer, MockConversationMemory


class ExtendedMockConversationMemory:
    """An extended mock conversation memory with multiple conversations"""

    def __init__(self):
        """Initialize with multiple test conversations"""
        # Conversation about AI and machine learning
        self.ai_conversation = [
            {
                "turn_id": 0,
                "timestamp": time.time() - 7200,
                "user_input": "Hello, I'm interested in learning about artificial intelligence.",
                "agent_response": "Hi there! I'd be happy to help you learn about AI. Is there a specific aspect of artificial intelligence you're interested in?"
            },
            {
                "turn_id": 1,
                "timestamp": time.time() - 7100,
                "user_input": "I want to know more about machine learning algorithms.",
                "agent_response": "Machine learning algorithms are methods that allow computers to learn patterns from data and make predictions. The main types include supervised learning, unsupervised learning, and reinforcement learning. Would you like me to explain any of these in more detail?"
            },
            {
                "turn_id": 2,
                "timestamp": time.time() - 7000,
                "user_input": "Yes, can you tell me about supervised learning?",
                "agent_response": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. It's called 'supervised' because the process of learning is guided by the labeled examples. The algorithm makes predictions based on the input data and is corrected when its predictions are wrong. Over time, it improves its accuracy. Common supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks."
            },
            {
                "turn_id": 3,
                "timestamp": time.time() - 6900,
                "user_input": "What's the difference between supervised and unsupervised learning?",
                "agent_response": "The main difference is that supervised learning uses labeled data, while unsupervised learning works with unlabeled data. In supervised learning, the algorithm learns to map inputs to known outputs. In unsupervised learning, the algorithm tries to find patterns or structure in the data without specific guidance. Unsupervised learning is often used for clustering, dimensionality reduction, and anomaly detection."
            },
            {
                "turn_id": 4,
                "timestamp": time.time() - 6800,
                "user_input": "Can you give me an example of a real-world application of machine learning?",
                "agent_response": "There are many real-world applications of machine learning! Some examples include: 1) Recommendation systems on platforms like Netflix or Amazon, 2) Email spam filters, 3) Voice assistants like Siri or Alexa, 4) Medical diagnosis from images, 5) Fraud detection in banking, and 6) Self-driving cars. These applications use different types of machine learning algorithms depending on their specific needs."
            }
        ]

        # Conversation about programming
        self.programming_conversation = [
            {
                "turn_id": 0,
                "timestamp": time.time() - 3600,
                "user_input": "I need help with Python programming.",
                "agent_response": "I'd be happy to help with Python programming. What specific aspect are you having trouble with?"
            },
            {
                "turn_id": 1,
                "timestamp": time.time() - 3500,
                "user_input": "I'm trying to understand how to work with lists and dictionaries.",
                "agent_response": "Lists and dictionaries are two of the most important data structures in Python. Lists are ordered collections of items, while dictionaries store key-value pairs. Would you like me to explain how to create and manipulate them?"
            },
            {
                "turn_id": 2,
                "timestamp": time.time() - 3400,
                "user_input": "Yes, please show me some examples of both.",
                "agent_response": "Here are some examples of lists in Python:\n\n```python\n# Creating a list\nmy_list = [1, 2, 3, 4, 5]\n\n# Accessing elements (indexing starts at 0)\nfirst_element = my_list[0]  # 1\n\n# Adding elements\nmy_list.append(6)  # [1, 2, 3, 4, 5, 6]\n\n# Slicing\nsub_list = my_list[1:4]  # [2, 3, 4]\n```\n\nAnd here are examples of dictionaries:\n\n```python\n# Creating a dictionary\nmy_dict = {'name': 'John', 'age': 30, 'city': 'New York'}\n\n# Accessing values\nname = my_dict['name']  # 'John'\n\n# Adding or updating key-value pairs\nmy_dict['email'] = 'john@example.com'\n\n# Getting all keys and values\nkeys = my_dict.keys()\nvalues = my_dict.values()\n```"
            },
            {
                "turn_id": 3,
                "timestamp": time.time() - 3300,
                "user_input": "How do I iterate through a dictionary?",
                "agent_response": "There are several ways to iterate through a dictionary in Python:\n\n```python\nmy_dict = {'name': 'John', 'age': 30, 'city': 'New York'}\n\n# Iterate through keys\nfor key in my_dict:\n    print(key, my_dict[key])\n\n# Iterate through key-value pairs\nfor key, value in my_dict.items():\n    print(key, value)\n\n# Iterate through keys only\nfor key in my_dict.keys():\n    print(key)\n\n# Iterate through values only\nfor value in my_dict.values():\n    print(value)\n```\n\nThe most common and readable way is using the `items()` method, which returns key-value pairs."
            },
            {
                "turn_id": 4,
                "timestamp": time.time() - 3200,
                "user_input": "What about list comprehensions? I've heard they're useful.",
                "agent_response": "List comprehensions are a concise way to create lists in Python. They're very powerful and can often replace loops and map/filter functions. Here's the basic syntax:\n\n```python\n[expression for item in iterable if condition]\n```\n\nExamples:\n\n```python\n# Create a list of squares\nsquares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# Filter even numbers\neven_numbers = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]\n\n# Create a list of tuples\ncoordinates = [(x, y) for x in range(3) for y in range(2)]  # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]\n```\n\nThere are also dictionary comprehensions with similar syntax:\n\n```python\n{key_expr: value_expr for item in iterable if condition}\n```\n\nExample:\n\n```python\n# Create a dictionary of squares\nsquare_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}\n```"
            }
        ]

        # Short conversation about weather
        self.weather_conversation = [
            {
                "turn_id": 0,
                "timestamp": time.time() - 1800,
                "user_input": "What's the weather like today?",
                "agent_response": "I don't have access to real-time weather data. To get the current weather, you could check a weather website or app, or look outside if you're able to!"
            },
            {
                "turn_id": 1,
                "timestamp": time.time() - 1700,
                "user_input": "How do weather forecasts work?",
                "agent_response": "Weather forecasting uses computer models that analyze current conditions (temperature, pressure, humidity, wind) and apply physics equations to predict future weather. These models use data from satellites, weather stations, radar, and other sources. Meteorologists interpret these models to create forecasts. The accuracy decreases the further into the future you predict, which is why short-term forecasts are generally more reliable than long-term ones."
            }
        ]

        # Current conversation is the programming one
        self.turns = self.programming_conversation

        # All conversations
        self.all_conversations = [
            self.ai_conversation,
            self.programming_conversation,
            self.weather_conversation
        ]

    def get_turns(self):
        """Get current conversation turns"""
        return self.turns

    def get_all_conversations(self):
        """Get all conversations"""
        return self.all_conversations


def print_separator(title):
    """Print a section separator with title"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")


def main():
    """Main test function"""
    # Clean up any existing test files
    test_analysis_path = "test_analysis.json"
    if os.path.exists(test_analysis_path):
        os.remove(test_analysis_path)

    print_separator("CONVERSATION MEMORY ANALYZER TEST")

    # Create the memory system and analyzer
    memory = ExtendedMockConversationMemory()
    analyzer = ConversationMemoryAnalyzer(memory, test_analysis_path)

    # Test 1: Extract key topics
    print_separator("TEST 1: EXTRACT KEY TOPICS")
    topics = analyzer.extract_key_topics(memory.get_turns())

    print("Key topics in the programming conversation:")
    for i, topic in enumerate(topics):
        print(f"{i+1}. {topic['topic']} (frequency: {topic['frequency']})")
        if topic['context']:
            print(f"   Context: \"{topic['context'][0]}\"")

    # Test 2: Identify patterns
    print_separator("TEST 2: IDENTIFY PATTERNS")
    patterns = analyzer.identify_patterns(memory.get_turns())

    print("Conversation patterns:")
    print(f"- Question frequency: {patterns['question_frequency']:.2f}")
    print("- Common question types:")
    for q_type in patterns['common_question_types']:
        print(f"  * {q_type['type']}: {q_type['count']}")

    print(f"- Topic shifts: {len(patterns['topic_shifts'])}")
    if patterns['topic_shifts']:
        for shift in patterns['topic_shifts']:
            print(f"  * From {shift['from_topics']} to {shift['to_topics']} at turn {shift['turn_index']}")

    print(f"- Response length trend: {patterns['response_length_trend']}")

    print("- Repeated queries:")
    if patterns['repeated_queries']:
        for query in patterns['repeated_queries']:
            print(f"  * \"{query['query']}\" (repeated {query['count']} times)")
    else:
        print("  * None detected")

    # Test 3: Generate conversation summary
    print_separator("TEST 3: GENERATE CONVERSATION SUMMARY")
    summary = analyzer.generate_conversation_summary(memory.get_turns())

    print("Conversation summary:")
    print(f"- Summary text: \"{summary['summary']}\"")
    print(f"- Duration: {analyzer._format_duration(summary['duration'])}")
    print(f"- Turns: {summary['turns']}")
    print(f"- Topics: {', '.join(summary['topics'])}")

    # Test 4: Provide conversation insights
    print_separator("TEST 4: PROVIDE CONVERSATION INSIGHTS")
    insights = analyzer.provide_conversation_insights(memory.get_turns())

    print("Conversation insights:")
    for i, insight in enumerate(insights['insights']):
        print(f"{i+1}. {insight}")

    print("\nConversation metrics:")
    for metric, value in insights['metrics'].items():
        print(f"- {metric}: {value}")

    # Test 5: Topic evolution
    print_separator("TEST 5: TOPIC EVOLUTION")
    evolution = analyzer.get_topic_evolution(memory.get_turns())

    print("Topic evolution throughout the conversation:")
    for segment in evolution:
        print(f"Segment {segment['segment_index']} (turns {segment['start_turn']}-{segment['end_turn']}):")
        for topic in segment['topics']:
            print(f"- {topic['topic']} (frequency: {topic['frequency']})")

    # Test 6: Find similar conversations
    print_separator("TEST 6: FIND SIMILAR CONVERSATIONS")
    similar = analyzer.find_similar_conversations(
        memory.get_turns(),
        memory.get_all_conversations(),
        top_n=2
    )

    print("Similar conversations:")
    for i, conv in enumerate(similar):
        print(f"{i+1}. Conversation {conv['conversation_index']} (similarity: {conv['similarity']:.2f})")
        print(f"   Common topics: {', '.join(conv['common_topics'])}")

    # Test 7: Complete analysis
    print_separator("TEST 7: COMPLETE ANALYSIS")
    analysis = analyzer.analyze_conversation()

    print("Complete conversation analysis:")
    print(f"- Analysis timestamp: {time.ctime(analysis['timestamp'])}")
    print(f"- Top topic: {analysis['topics'][0]['topic'] if analysis['topics'] else 'None'}")
    print(f"- Summary: \"{analysis['summary']['summary']}\"")
    print(f"- Key insight: \"{analysis['insights']['insights'][0] if analysis['insights']['insights'] else 'None'}\"")

    # Clean up
    if os.path.exists(test_analysis_path):
        os.remove(test_analysis_path)

    print_separator("TEST COMPLETED")


if __name__ == "__main__":
    main()
