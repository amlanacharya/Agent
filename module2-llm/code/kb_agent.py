"""
Knowledge Base Assistant Implementation with Groq API
-------------------------------------------------
This file contains the implementation of a knowledge base assistant that can
store, retrieve, and reason with structured knowledge, answer questions,
learn from conversations, and provide citations, all powered by the Groq API.
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Import our modules
try:
    # When running from the module2-llm/code directory
    from groq_client import GroqClient
    from memory_types import ShortTermMemory
    from knowledge_base import KnowledgeBase, CitationManager, UncertaintyHandler
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient
    from module2_llm.code.memory_types import ShortTermMemory
    from module2_llm.code.knowledge_base import KnowledgeBase, CitationManager, UncertaintyHandler


class KnowledgeBaseAssistant:
    """
    A knowledge base assistant that can answer questions, learn from conversations,
    and provide citations for its answers, powered by the Groq API.
    """
    
    def __init__(self, storage_dir: str = "kb_assistant", knowledge_base: Optional[KnowledgeBase] = None):
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
        
        # Initialize Groq client
        self.groq_client = GroqClient()
        
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
                "max_results": 5,
                "system_prompt": "You are a helpful knowledge base assistant that provides accurate information based on your knowledge base."
            }
            self._save_settings()
    
    def _save_settings(self) -> None:
        """Save settings to disk"""
        with open(self.settings_path, 'w') as f:
            json.dump(self.settings, f)
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update assistant settings
        
        Args:
            new_settings (dict): New settings to apply
        """
        self.settings.update(new_settings)
        
        # Update uncertainty handler threshold if it changed
        if "confidence_threshold" in new_settings:
            self.uncertainty_handler.confidence_threshold = new_settings["confidence_threshold"]
        
        self._save_settings()
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and update conversation memory
        
        Args:
            user_input (str): The user's input
            
        Returns:
            dict: Processed input with metadata
        """
        # Add to conversation memory
        self.conversation_memory.add({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Extract key information using LLM
        key_info = self._extract_key_information(user_input)
        
        return {
            'text': user_input,
            'key_information': key_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from text using LLM
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Extracted key information
        """
        prompt = f"""
        Extract key information from this text:
        
        "{text}"
        
        Return a JSON object with the following fields:
        - topics: List of main topics mentioned
        - entities: List of named entities (people, organizations, places, etc.)
        - intent: The apparent intent of the text (question, statement, request, etc.)
        - sentiment: The sentiment of the text (positive, negative, neutral)
        - is_question: Boolean indicating if this is a question
        - potential_facts: List of factual statements that could be added to a knowledge base
        
        Return only the JSON object, without additional text.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=300)
            result = self.groq_client.extract_text_from_response(response)
            
            # Parse the JSON response
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # If the response is not valid JSON, return a simplified structure
                return {
                    'topics': [],
                    'entities': [],
                    'intent': 'unknown',
                    'sentiment': 'neutral',
                    'is_question': '?' in text,
                    'potential_facts': []
                }
        except Exception as e:
            # Return a default structure if LLM extraction fails
            print(f"Key information extraction failed: {e}")
            return {
                'topics': [],
                'entities': [],
                'intent': 'unknown',
                'sentiment': 'neutral',
                'is_question': '?' in text,
                'potential_facts': []
            }
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the knowledge base
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The assistant's answer
        """
        # Process the question
        processed_input = self.process_input(question)
        
        # Retrieve relevant knowledge
        results = self.knowledge_base.retrieve(question, top_k=self.settings["max_results"])
        
        # Assess confidence
        confidence = self.uncertainty_handler.assess_confidence(question, results)
        
        # Generate answer with uncertainty handling
        answer = self.uncertainty_handler.generate_response_with_uncertainty(
            question, results, confidence
        )
        
        # Add citations
        if results:
            answer = self.citation_manager.add_citations_to_response(
                answer, results, self.settings["citation_style"]
            )
        
        # Store the answer in conversation memory
        self.conversation_memory.add({
            'role': 'assistant',
            'content': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Learn from the interaction if learning mode is active
        if self.settings["learning_mode"] == "active":
            self._learn_from_interaction(question, answer, results)
        
        return answer
    
    def _learn_from_interaction(self, question: str, answer: str, results: List[Dict[str, Any]]) -> None:
        """
        Learn from the interaction by extracting new knowledge
        
        Args:
            question (str): The user's question
            answer (str): The assistant's answer
            results (list): The retrieved results
        """
        # Get recent conversation context
        recent_turns = self.conversation_memory.get_recent(5)
        conversation_text = "\n".join([
            f"{turn['content']['role']}: {turn['content']['content']}"
            for turn in recent_turns
        ])
        
        prompt = f"""
        Analyze this conversation and identify new facts that should be added to the knowledge base:
        
        Conversation:
        {conversation_text}
        
        Current question: "{question}"
        Current answer: "{answer}"
        
        Extract 0-3 new facts that:
        1. Are not already in the retrieved results
        2. Are factual and objective (not opinions)
        3. Would be useful for future questions
        
        For each fact, provide:
        - fact: The factual statement
        - confidence: How confident you are that this is factual (high, medium, low)
        - source: Where this fact comes from (user, inference, or external)
        
        Return a JSON array of facts. If no new facts should be added, return an empty array.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            result = self.groq_client.extract_text_from_response(response)
            
            # Parse the JSON response
            try:
                facts = json.loads(result)
                if isinstance(facts, list):
                    # Add high-confidence facts to the knowledge base
                    for fact_item in facts:
                        if isinstance(fact_item, dict) and 'fact' in fact_item:
                            fact = fact_item['fact']
                            confidence = fact_item.get('confidence', 'low')
                            source = fact_item.get('source', 'conversation')
                            
                            if confidence.lower() in ['high', 'medium']:
                                self.knowledge_base.add_item(fact, {
                                    'source': source,
                                    'confidence': confidence,
                                    'extracted_from': 'conversation',
                                    'timestamp': datetime.now().isoformat()
                                })
            except json.JSONDecodeError:
                # If the response is not valid JSON, don't add any facts
                pass
        except Exception as e:
            # If LLM extraction fails, don't add any facts
            print(f"Learning from interaction failed: {e}")
    
    def learn_from_statement(self, statement: str) -> str:
        """
        Learn from a statement by extracting facts
        
        Args:
            statement (str): The user's statement
            
        Returns:
            str: Acknowledgment of learning
        """
        # Process the statement
        processed_input = self.process_input(statement)
        
        # Extract potential facts
        potential_facts = processed_input['key_information'].get('potential_facts', [])
        
        # If no facts were automatically extracted, use LLM to extract them
        if not potential_facts:
            prompt = f"""
            Extract factual statements from this text:
            
            "{statement}"
            
            Return a JSON array of factual statements. For each fact, provide:
            - fact: The factual statement
            - confidence: How confident you are that this is factual (high, medium, low)
            
            Only include objective, factual information. Exclude opinions, subjective statements, or uncertain claims.
            If no facts can be extracted, return an empty array.
            """
            
            try:
                response = self.groq_client.generate_text(prompt, max_tokens=300)
                result = self.groq_client.extract_text_from_response(response)
                
                # Parse the JSON response
                try:
                    extracted = json.loads(result)
                    if isinstance(extracted, list):
                        potential_facts = [item.get('fact') for item in extracted if isinstance(item, dict) and 'fact' in item]
                except json.JSONDecodeError:
                    # If the response is not valid JSON, don't extract any facts
                    potential_facts = []
            except Exception as e:
                # If LLM extraction fails, don't extract any facts
                print(f"Fact extraction failed: {e}")
                potential_facts = []
        
        # Add facts to the knowledge base
        added_facts = []
        for fact in potential_facts:
            if fact and isinstance(fact, str):
                self.knowledge_base.add_item(fact, {
                    'source': 'user',
                    'confidence': 'medium',
                    'extracted_from': 'statement',
                    'timestamp': datetime.now().isoformat()
                })
                added_facts.append(fact)
        
        # Generate acknowledgment
        if added_facts:
            acknowledgment = f"I've learned {len(added_facts)} new facts from your statement."
            if len(added_facts) <= 3:
                acknowledgment += " Here's what I learned:\n"
                for i, fact in enumerate(added_facts):
                    acknowledgment += f"{i+1}. {fact}\n"
        else:
            acknowledgment = "I couldn't extract any new factual information from your statement."
        
        # Store the acknowledgment in conversation memory
        self.conversation_memory.add({
            'role': 'assistant',
            'content': acknowledgment,
            'timestamp': datetime.now().isoformat()
        })
        
        return acknowledgment
    
    def handle_input(self, user_input: str) -> str:
        """
        Handle user input and generate an appropriate response
        
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
                if any(phrase in user_input.lower() for phrase in ["is a", "are a", "refers to", "defined as", "means", "consists of"]):
                    return self.learn_from_statement(user_input)
                else:
                    # Process as a regular input
                    return self.generate_response(user_input)
            else:
                # Learning mode is off, just generate a response
                return self.generate_response(user_input)
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input using LLM
        
        Args:
            user_input (str): The user's input
            
        Returns:
            str: The assistant's response
        """
        # Process the input
        processed_input = self.process_input(user_input)
        
        # Get recent conversation context
        recent_turns = self.conversation_memory.get_recent(5)
        conversation_text = "\n".join([
            f"{turn['content']['role']}: {turn['content']['content']}"
            for turn in recent_turns
        ])
        
        # Retrieve relevant knowledge
        results = self.knowledge_base.retrieve(user_input, top_k=self.settings["max_results"])
        
        # Format retrieved knowledge
        knowledge_text = ""
        if results:
            knowledge_text = "\n\n".join([
                f"Knowledge {i+1}:\n{result['text']}"
                for i, result in enumerate(results)
            ])
        
        # Generate response using chat completion
        messages = [
            {"role": "system", "content": self.settings["system_prompt"]},
            {"role": "user", "content": f"""
            Generate a response to this user input: "{user_input}"
            
            Recent conversation:
            {conversation_text}
            
            Relevant knowledge from the knowledge base:
            {knowledge_text}
            
            Respond in a helpful and informative way. If the knowledge base doesn't contain relevant information,
            acknowledge that and provide a general response based on your training.
            """}
        ]
        
        try:
            response = self.groq_client.chat_completion(messages)
            answer = self.groq_client.extract_text_from_response(response)
            
            # Store the response in conversation memory
            self.conversation_memory.add({
                'role': 'assistant',
                'content': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            return answer
        except Exception as e:
            # Return a simple response if generation fails
            print(f"Response generation failed: {e}")
            fallback_response = "I'm having trouble generating a response right now. Could you try again or rephrase your input?"
            
            # Store the fallback response in conversation memory
            self.conversation_memory.add({
                'role': 'assistant',
                'content': fallback_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return fallback_response
    
    def get_knowledge_base_summary(self) -> str:
        """
        Get a summary of the knowledge base
        
        Returns:
            str: Summary of the knowledge base
        """
        return self.knowledge_base.generate_summary()
    
    def explain_answer(self, question: str) -> Dict[str, Any]:
        """
        Explain how the assistant arrived at an answer
        
        Args:
            question (str): The question to explain
            
        Returns:
            dict: Explanation of the answer
        """
        # Retrieve relevant knowledge
        retrieval_results = self.knowledge_base.retrieve_with_explanation(
            question, top_k=self.settings["max_results"]
        )
        
        # Assess confidence
        confidence = self.uncertainty_handler.assess_confidence(
            question, retrieval_results['results']
        )
        
        # Generate answer with uncertainty handling
        answer = self.uncertainty_handler.generate_response_with_uncertainty(
            question, retrieval_results['results'], confidence
        )
        
        # Generate explanation of the reasoning process
        reasoning_explanation = self._explain_reasoning_process(
            question, retrieval_results['results'], answer, confidence
        )
        
        return {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'retrieval_explanation': retrieval_results['explanation'],
            'reasoning_explanation': reasoning_explanation,
            'results': retrieval_results['results']
        }
    
    def _explain_reasoning_process(
        self, 
        question: str, 
        results: List[Dict[str, Any]], 
        answer: str, 
        confidence: float
    ) -> str:
        """
        Explain the reasoning process used to generate an answer
        
        Args:
            question (str): The question
            results (list): The retrieved results
            answer (str): The generated answer
            confidence (float): The confidence score
            
        Returns:
            str: Explanation of the reasoning process
        """
        # Format results for the LLM
        results_text = "\n\n".join([
            f"Result {i+1} (Similarity: {result['similarity']:.4f}):\n{result['text']}"
            for i, result in enumerate(results)
        ])
        
        prompt = f"""
        Explain the reasoning process used to generate this answer:
        
        Question: "{question}"
        
        Retrieved information:
        {results_text}
        
        Confidence: {confidence:.2f}
        
        Answer: "{answer}"
        
        Provide a step-by-step explanation of:
        1. How the relevant information was identified
        2. How the information was synthesized
        3. How confidence was assessed
        4. Any limitations or uncertainties in the answer
        
        Explanation:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Return a simple explanation if generation fails
            print(f"Reasoning explanation failed: {e}")
            return f"The system retrieved relevant information about '{question}', assessed the confidence based on the quality of matches, and generated an answer that synthesizes the available knowledge."
    
    def add_to_knowledge_base(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add information to the knowledge base
        
        Args:
            text (str): The text to add
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Confirmation message
        """
        # Add default metadata if not provided
        if metadata is None:
            metadata = {}
        
        if 'source' not in metadata:
            metadata['source'] = 'user'
        
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
        
        # Add to knowledge base
        item_id = self.knowledge_base.add_item(text, metadata)
        
        return f"Added to knowledge base with ID: {item_id}"
    
    def get_conversation_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history
        
        Args:
            n (int): Number of turns to retrieve
            
        Returns:
            list: Recent conversation turns
        """
        return self.conversation_memory.get_recent(n)


# Example usage
if __name__ == "__main__":
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant()
    
    # Add some knowledge
    assistant.add_to_knowledge_base("Python is a high-level programming language known for its readability and simplicity.")
    assistant.add_to_knowledge_base("Python was created by Guido van Rossum and first released in 1991.")
    assistant.add_to_knowledge_base("Python is widely used in data science, machine learning, web development, and automation.")
    assistant.add_to_knowledge_base("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    assistant.add_to_knowledge_base("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    
    # Simulate a conversation
    print("User: What is Python?")
    response = assistant.handle_input("What is Python?")
    print(f"Assistant: {response}")
    
    print("\nUser: Who created it?")
    response = assistant.handle_input("Who created it?")
    print(f"Assistant: {response}")
    
    print("\nUser: Python is also the name of a snake.")
    response = assistant.handle_input("Python is also the name of a snake.")
    print(f"Assistant: {response}")
    
    print("\nUser: What can you tell me about machine learning?")
    response = assistant.handle_input("What can you tell me about machine learning?")
    print(f"Assistant: {response}")
    
    # Get an explanation
    print("\nExplaining answer to: 'What is NLP?'")
    explanation = assistant.explain_answer("What is NLP?")
    print(f"Answer: {explanation['answer']}")
    print(f"Confidence: {explanation['confidence']:.2f}")
    print(f"Retrieval explanation: {explanation['retrieval_explanation']}")
    print(f"Reasoning explanation: {explanation['reasoning_explanation']}")
