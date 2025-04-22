"""
Retrieval Agent Implementation with LLM Enhancement
------------------------------------------------
This file contains an implementation of a retrieval agent that uses
LLM enhancement for better context understanding and retrieval.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime

# Import our modules
try:
    # When running from the module2-llm/code directory
    from groq_client import GroqClient
    from memory_types import ShortTermMemory
    from vector_store import SimpleVectorDB, EnhancedVectorDB
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient
    from module2_llm.code.memory_types import ShortTermMemory
    from module2_llm.code.vector_store import SimpleVectorDB, EnhancedVectorDB


class RetrievalAgent:
    """
    A retrieval agent that uses LLM enhancement for better context understanding and retrieval.
    """
    
    def __init__(self, vector_db: Optional[EnhancedVectorDB] = None):
        """
        Initialize the retrieval agent
        
        Args:
            vector_db (EnhancedVectorDB, optional): Vector database to use
        """
        self.vector_db = vector_db or EnhancedVectorDB()
        self.conversation_memory = ShortTermMemory(capacity=20)
        self.groq_client = GroqClient()
    
    def add_knowledge(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add knowledge to the agent's vector database
        
        Args:
            text (str): The text to add
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: ID of the added item
        """
        return self.vector_db.add(text, metadata)
    
    def add_knowledge_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple knowledge items to the agent's vector database
        
        Args:
            texts (list): List of texts to add
            metadatas (list, optional): List of metadata dictionaries
            
        Returns:
            list: IDs of the added items
        """
        return self.vector_db.add_batch(texts, metadatas)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to the query
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            
        Returns:
            list: Relevant knowledge items
        """
        return self.vector_db.search(query, top_k=top_k)
    
    def retrieve_with_expansion(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge with query expansion for better recall
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            
        Returns:
            list: Relevant knowledge items
        """
        return self.vector_db.search_with_expansion(query, top_k=top_k)
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
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
        
        Return only the JSON object, without additional text.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=200)
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
                    'sentiment': 'neutral'
                }
        except Exception as e:
            # Return a default structure if LLM extraction fails
            print(f"Key information extraction failed: {e}")
            return {
                'topics': [],
                'entities': [],
                'intent': 'unknown',
                'sentiment': 'neutral'
            }
    
    def store_response(self, response: str) -> None:
        """
        Store an agent response in conversation memory
        
        Args:
            response (str): The agent's response
        """
        self.conversation_memory.add({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
    
    def retrieve_with_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge with conversation context for better relevance
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            
        Returns:
            list: Relevant knowledge items with context
        """
        # Get recent conversation turns
        recent_turns = self.conversation_memory.get_recent(5)
        
        # Extract conversation context
        conversation_context = self._extract_conversation_context(recent_turns)
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, conversation_context)
        
        # Retrieve with enhanced query
        results = self.vector_db.search_with_expansion(enhanced_query, top_k=top_k)
        
        # Add context information to results
        for result in results:
            result['context_info'] = {
                'original_query': query,
                'enhanced_query': enhanced_query,
                'conversation_context': conversation_context
            }
        
        return results
    
    def _extract_conversation_context(self, conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract context from conversation history using LLM
        
        Args:
            conversation_turns (list): Recent conversation turns
            
        Returns:
            dict: Extracted context information
        """
        if not conversation_turns:
            return {
                'topics': [],
                'entities': [],
                'user_interests': [],
                'open_questions': []
            }
        
        # Format conversation for the LLM
        conversation_text = "\n".join([
            f"{turn['content']['role']}: {turn['content']['content']}"
            for turn in conversation_turns
        ])
        
        prompt = f"""
        Analyze this conversation history:
        
        {conversation_text}
        
        Extract the following information:
        - topics: Main topics discussed in the conversation
        - entities: Named entities mentioned (people, organizations, places, etc.)
        - user_interests: What the user seems interested in
        - open_questions: Any questions that haven't been fully answered
        
        Return a JSON object with these fields. Return only the JSON object, without additional text.
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
                    'user_interests': [],
                    'open_questions': []
                }
        except Exception as e:
            # Return a default structure if LLM extraction fails
            print(f"Conversation context extraction failed: {e}")
            return {
                'topics': [],
                'entities': [],
                'user_interests': [],
                'open_questions': []
            }
    
    def _enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """
        Enhance a query with conversation context using LLM
        
        Args:
            query (str): The original query
            context (dict): Conversation context
            
        Returns:
            str: Enhanced query
        """
        # Format context for the LLM
        context_text = json.dumps(context, indent=2)
        
        prompt = f"""
        Original query: "{query}"
        
        Conversation context:
        {context_text}
        
        Enhance the original query to make it more specific and relevant based on the conversation context.
        The enhanced query should help retrieve more relevant information.
        
        Return only the enhanced query, without quotes or additional text.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=150)
            enhanced_query = self.groq_client.extract_text_from_response(response).strip()
            
            # If the enhancement failed or returned nothing, use the original query
            if not enhanced_query:
                return query
            
            return enhanced_query
        except Exception as e:
            # Return the original query if enhancement fails
            print(f"Query enhancement failed: {e}")
            return query
    
    def generate_response(self, query: str, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on retrieved items using LLM
        
        Args:
            query (str): The user's query
            retrieved_items (list): Retrieved knowledge items
            
        Returns:
            str: Generated response
        """
        if not retrieved_items:
            return "I don't have any information about that."
        
        # Format retrieved items for the LLM
        retrieved_text = "\n\n".join([
            f"Item {i+1}:\n{item['text']}"
            for i, item in enumerate(retrieved_items)
        ])
        
        # Get recent conversation for context
        recent_turns = self.conversation_memory.get_recent(3)
        conversation_text = "\n".join([
            f"{turn['content']['role']}: {turn['content']['content']}"
            for turn in recent_turns
        ])
        
        prompt = f"""
        User query: "{query}"
        
        Recent conversation:
        {conversation_text}
        
        Retrieved information:
        {retrieved_text}
        
        Based on the user's query and the retrieved information, generate a helpful and informative response.
        If the retrieved information doesn't fully answer the query, acknowledge that.
        If there are multiple relevant pieces of information, synthesize them into a coherent response.
        
        Response:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Return a simple response if generation fails
            print(f"Response generation failed: {e}")
            return f"Based on my knowledge, {retrieved_items[0]['text']}"
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the retrieval agent
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The agent's answer
        """
        # Process the question
        processed_input = self.process_user_input(question)
        
        # Retrieve relevant knowledge with context
        retrieved_items = self.retrieve_with_context(question, top_k=5)
        
        # Generate a response
        response = self.generate_response(question, retrieved_items)
        
        # Store the response
        self.store_response(response)
        
        return response
    
    def explain_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Explain the retrieval process for a query
        
        Args:
            query (str): The search query
            
        Returns:
            dict: Explanation of the retrieval process
        """
        # Process the query
        processed_input = self.process_user_input(query)
        
        # Get conversation context
        recent_turns = self.conversation_memory.get_recent(5)
        conversation_context = self._extract_conversation_context(recent_turns)
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, conversation_context)
        
        # Expand the enhanced query
        expanded_queries = self.vector_db.expand_query(enhanced_query)
        
        # Retrieve with each expanded query
        all_results = []
        for expanded_query in expanded_queries:
            results = self.vector_db.search(expanded_query, top_k=3)
            for result in results:
                result['expanded_query'] = expanded_query
            all_results.extend(results)
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = unique_results[:5]
        
        # Generate explanation
        explanation = self._generate_retrieval_explanation(
            query, 
            enhanced_query, 
            expanded_queries, 
            top_results, 
            conversation_context
        )
        
        return {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'expanded_queries': expanded_queries,
            'conversation_context': conversation_context,
            'top_results': top_results,
            'explanation': explanation
        }
    
    def _generate_retrieval_explanation(
        self, 
        original_query: str, 
        enhanced_query: str, 
        expanded_queries: List[str], 
        results: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate an explanation of the retrieval process using LLM
        
        Args:
            original_query (str): The original query
            enhanced_query (str): The enhanced query
            expanded_queries (list): The expanded queries
            results (list): The retrieved results
            context (dict): The conversation context
            
        Returns:
            str: Explanation of the retrieval process
        """
        # Format inputs for the LLM
        expanded_text = "\n".join([f"- {q}" for q in expanded_queries])
        
        results_text = "\n\n".join([
            f"Result {i+1} (Similarity: {result['similarity']:.4f}):\n{result['text'][:100]}..."
            for i, result in enumerate(results)
        ])
        
        context_text = json.dumps(context, indent=2)
        
        prompt = f"""
        Explain the retrieval process for this query:
        
        Original query: "{original_query}"
        
        Step 1: The query was enhanced with conversation context:
        Enhanced query: "{enhanced_query}"
        
        Step 2: The enhanced query was expanded to improve recall:
        {expanded_text}
        
        Step 3: The system searched for relevant information using these queries and found:
        {results_text}
        
        Conversation context used:
        {context_text}
        
        Provide a clear explanation of how the retrieval process worked, including:
        - How the conversation context influenced the query enhancement
        - How query expansion helped find relevant information
        - Why the top results were selected
        - Any challenges or limitations in the retrieval process
        
        Explanation:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Return a simple explanation if generation fails
            print(f"Explanation generation failed: {e}")
            return f"The system enhanced the query '{original_query}' to '{enhanced_query}' based on conversation context, then expanded it to find the most relevant information."


# Example usage
if __name__ == "__main__":
    # Create a retrieval agent
    agent = RetrievalAgent()
    
    # Add some knowledge
    agent.add_knowledge("Python is a high-level programming language known for its readability and simplicity.")
    agent.add_knowledge("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    agent.add_knowledge("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    agent.add_knowledge("Vector databases store data as high-dimensional vectors and enable semantic search.")
    agent.add_knowledge("Embeddings are numerical representations of text that capture semantic meaning.")
    agent.add_knowledge("Transformers are a type of neural network architecture used in NLP tasks.")
    agent.add_knowledge("BERT is a transformer-based language model developed by Google.")
    agent.add_knowledge("GPT (Generative Pre-trained Transformer) is a type of language model developed by OpenAI.")
    agent.add_knowledge("Groq is a company that provides high-performance AI inference services.")
    agent.add_knowledge("LangChain is a framework for developing applications powered by language models.")
    
    # Simulate a conversation
    print("User: What is Python?")
    response = agent.answer_question("What is Python?")
    print(f"Agent: {response}")
    
    print("\nUser: How is it used in AI?")
    response = agent.answer_question("How is it used in AI?")
    print(f"Agent: {response}")
    
    print("\nUser: Tell me about language models")
    response = agent.answer_question("Tell me about language models")
    print(f"Agent: {response}")
    
    # Explain the retrieval process
    print("\nExplaining retrieval for: 'What are transformers in AI?'")
    explanation = agent.explain_retrieval("What are transformers in AI?")
    print(explanation['explanation'])
