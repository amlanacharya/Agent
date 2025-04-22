"""
Knowledge Base Assistant Implementation
--------------------------------------
This file contains the implementation of a knowledge base assistant that can
store, retrieve, and reason with structured knowledge, answer questions,
learn from conversations, and provide citations.
"""

import os
import json
import time
import re
from datetime import datetime

# Import our knowledge base implementation
from module2.code.knowledge_base import KnowledgeBase, CitationManager, UncertaintyHandler
from module2.code.memory_types import ShortTermMemory


class KnowledgeBaseAssistant:
    """
    A knowledge base assistant that can answer questions, learn from conversations,
    and provide citations for its answers.
    """
    
    def __init__(self, storage_dir="kb_assistant", knowledge_base=None):
        """
        Initialize the knowledge base assistant
        
        Args:
            storage_dir (str): Directory for persistent storage
            knowledge_base (KnowledgeBase, optional): Existing knowledge base to use
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize or load the knowledge base
        self.knowledge_base = knowledge_base or KnowledgeBase(os.path.join(storage_dir, "knowledge_base"))
        
        # Initialize conversation memory
        self.conversation_memory = ShortTermMemory(capacity=20)
        
        # Initialize citation manager
        self.citation_manager = CitationManager(self.knowledge_base)
        
        # Initialize uncertainty handler
        self.uncertainty_handler = UncertaintyHandler(confidence_threshold=0.7)
        
        # Load or initialize settings
        self.settings_path = os.path.join(storage_dir, "settings.json")
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                self.settings = json.load(f)
        else:
            self.settings = {
                "citation_style": "standard",
                "confidence_threshold": 0.7,
                "learning_mode": "passive",  # passive, active, or off
                "max_results": 5
            }
            self._save_settings()
    
    def _save_settings(self):
        """Save settings to disk"""
        with open(self.settings_path, 'w') as f:
            json.dump(self.settings, f)
    
    def answer_question(self, question):
        """
        Answer a question using the knowledge base
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The assistant's response
        """
        # Add question to conversation memory
        self.conversation_memory.add({"role": "user", "content": question})
        
        # Retrieve relevant knowledge
        knowledge_results = self.knowledge_base.retrieve_knowledge(
            question, 
            top_k=self.settings["max_results"]
        )
        
        if not knowledge_results:
            # No relevant knowledge found
            response = self._handle_unknown(question)
        else:
            # Check if the top result is confident enough
            confidence = self.uncertainty_handler.evaluate_confidence(knowledge_results)
            
            if confidence >= self.settings["confidence_threshold"]:
                # Generate answer with citation
                response = self._generate_answer(question, knowledge_results)
            else:
                # Not confident enough
                response = self._handle_uncertain(question, knowledge_results)
        
        # Add response to conversation memory
        self.conversation_memory.add({"role": "assistant", "content": response})
        
        return response
    
    def _generate_answer(self, question, knowledge_results):
        """
        Generate an answer based on retrieved knowledge
        
        Args:
            question (str): The user's question
            knowledge_results (list): The retrieved knowledge
            
        Returns:
            str: The generated answer
        """
        # In a real implementation, this would use an LLM to generate a coherent answer
        # For this example, we'll use a simple template
        
        answer = f"Based on my knowledge, {knowledge_results[0]['text']}"
        
        # Add related information if available
        if len(knowledge_results) > 1:
            answer += f"\n\nAdditionally, you might want to know that {knowledge_results[1]['text']}"
        
        # Add citations
        answer = self.citation_manager.add_citations_to_response(
            answer, 
            knowledge_results, 
            self.settings["citation_style"]
        )
        
        return answer
    
    def _handle_unknown(self, question):
        """
        Handle case where no knowledge is available
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The response for unknown information
        """
        if self.settings["learning_mode"] == "active":
            return "I don't have information about that in my knowledge base. Would you like to teach me about this topic?"
        else:
            return "I don't have information about that in my knowledge base."
    
    def _handle_uncertain(self, question, knowledge_results):
        """
        Handle case where confidence is low
        
        Args:
            question (str): The user's question
            knowledge_results (list): The retrieved knowledge
            
        Returns:
            str: The response with uncertainty markers
        """
        response = self.uncertainty_handler.generate_response(question, knowledge_results)
        
        # Add citations
        response = self.citation_manager.add_citations_to_response(
            response, 
            knowledge_results, 
            self.settings["citation_style"]
        )
        
        return response
    
    def learn_from_statement(self, statement, source="user", confidence=0.8):
        """
        Learn new information from a statement
        
        Args:
            statement (str): The statement to learn from
            source (str): The source of the information
            confidence (float): The confidence in the information
            
        Returns:
            str: The response acknowledging the learning
        """
        # Add statement to conversation memory
        self.conversation_memory.add({"role": "user", "content": statement})
        
        # Check if learning mode is off
        if self.settings["learning_mode"] == "off":
            response = "I'm currently not in learning mode. Would you like me to enable it?"
            self.conversation_memory.add({"role": "assistant", "content": response})
            return response
        
        # Extract knowledge from the statement
        # In a real implementation, this would use an LLM or information extraction system
        # For this example, we'll just use the statement directly
        
        # Add to knowledge base
        knowledge_id = self.knowledge_base.add_knowledge(statement, {
            "source": source,
            "confidence": confidence,
            "tags": ["user-provided"]
        })
        
        response = "Thank you for sharing that information. I've added it to my knowledge base."
        
        # Add response to conversation memory
        self.conversation_memory.add({"role": "assistant", "content": response})
        
        return response
    
    def process_input(self, user_input):
        """
        Process user input and determine whether it's a question or statement
        
        Args:
            user_input (str): The user's input
            
        Returns:
            str: The assistant's response
        """
        # Simple heuristic to determine if input is a question or statement
        # In a real implementation, this would use more sophisticated NLP
        if user_input.strip().endswith("?") or user_input.lower().startswith(("what", "who", "where", "when", "why", "how", "can", "could", "is", "are", "do", "does")):
            return self.answer_question(user_input)
        else:
            # If in active learning mode, learn from all statements
            if self.settings["learning_mode"] == "active":
                return self.learn_from_statement(user_input)
            # If in passive learning mode, only learn from statements that look like facts
            elif self.settings["learning_mode"] == "passive":
                # Simple heuristic for fact-like statements
                # In a real implementation, this would use more sophisticated NLP
                if " is " in user_input or " are " in user_input or " was " in user_input or " were " in user_input:
                    return self.learn_from_statement(user_input)
                else:
                    # Treat as a question if not clearly a fact
                    return self.answer_question(user_input)
            else:
                # Learning mode is off, treat everything as a question
                return self.answer_question(user_input)
    
    def update_settings(self, new_settings):
        """
        Update assistant settings
        
        Args:
            new_settings (dict): New settings to apply
            
        Returns:
            str: Confirmation message
        """
        for key, value in new_settings.items():
            if key in self.settings:
                self.settings[key] = value
        
        self._save_settings()
        
        return "Settings updated successfully."
    
    def get_settings(self):
        """
        Get current assistant settings
        
        Returns:
            dict: Current settings
        """
        return self.settings
    
    def get_conversation_history(self, n=None):
        """
        Get recent conversation history
        
        Args:
            n (int, optional): Number of recent interactions to retrieve
            
        Returns:
            list: Recent conversation history
        """
        return self.conversation_memory.get_recent(n)
    
    def clear_conversation_history(self):
        """
        Clear conversation history
        
        Returns:
            str: Confirmation message
        """
        self.conversation_memory.clear()
        return "Conversation history cleared."
    
    def save(self):
        """
        Save the assistant state
        
        Returns:
            str: Confirmation message
        """
        self.knowledge_base.save()
        self._save_settings()
        return "Assistant state saved successfully."
    
    def __str__(self):
        """String representation of the assistant"""
        return f"KnowledgeBaseAssistant(kb_size={len(self.knowledge_base)}, learning_mode={self.settings['learning_mode']})"


if __name__ == "__main__":
    # Example usage
    assistant = KnowledgeBaseAssistant("example_assistant")
    
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
    
    # Answer a question
    response = assistant.answer_question("What is Python?")
    print("Question: What is Python?")
    print("Response:", response)
    
    # Learn from a statement
    response = assistant.process_input("JavaScript is commonly used for web development.")
    print("\nStatement: JavaScript is commonly used for web development.")
    print("Response:", response)
    
    # Answer a question with uncertain information
    response = assistant.answer_question("What is JavaScript used for?")
    print("\nQuestion: What is JavaScript used for?")
    print("Response:", response)
    
    # Handle unknown information
    response = assistant.answer_question("What is Rust?")
    print("\nQuestion: What is Rust?")
    print("Response:", response)
