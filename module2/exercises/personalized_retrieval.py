"""
Module 2: Memory Systems - Personalized Retrieval System Exercise
----------------------------------------------------------------
This file contains the implementation of Exercise 2 from Lesson 4:
A Personalized Retrieval System that adapts to user profiles and preferences.
"""

import time
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import Counter


class PersonalizedRetrievalSystem:
    """
    Exercise 2: Personalized Retrieval System
    
    This system enhances retrieval by incorporating user profiles, interests,
    expertise levels, and feedback to provide personalized results.
    """
    
    def __init__(self, memory_system, user_profiles_path="user_profiles.json"):
        """
        Initialize the personalized retrieval system
        
        Args:
            memory_system: The memory system to search (must have a retrieve method)
            user_profiles_path (str): Path to store user profiles
        """
        self.memory_system = memory_system
        self.user_profiles_path = user_profiles_path
        self.user_profiles = self._load_profiles()
        
        # Default expertise levels and their weights
        self.expertise_levels = {
            "beginner": 0.2,
            "intermediate": 0.5,
            "advanced": 0.8,
            "expert": 1.0
        }
        
        # Feedback weights for result scoring
        self.feedback_weights = {
            "helpful": 0.8,
            "somewhat_helpful": 0.4,
            "not_helpful": -0.2
        }
    
    def _load_profiles(self) -> Dict:
        """Load user profiles from storage"""
        if os.path.exists(self.user_profiles_path):
            try:
                with open(self.user_profiles_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_profiles(self) -> None:
        """Save user profiles to storage"""
        with open(self.user_profiles_path, 'w') as f:
            json.dump(self.user_profiles, f)
    
    def create_user_profile(self, user_id: str, initial_data: Optional[Dict] = None) -> bool:
        """
        Create a new user profile
        
        Args:
            user_id (str): Unique identifier for the user
            initial_data (dict, optional): Initial profile data
            
        Returns:
            bool: True if created, False if user already exists
        """
        if user_id in self.user_profiles:
            return False  # Profile already exists
        
        self.user_profiles[user_id] = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "interests": [],
            "expertise_level": "beginner",
            "interaction_history": [],
            "feedback_history": {},
            "result_preferences": {},
            **(initial_data or {})
        }
        
        self._save_profiles()
        return True
    
    def update_user_profile(self, user_id: str, **kwargs) -> bool:
        """
        Update a user profile with provided key-value pairs
        
        Args:
            user_id (str): User identifier
            **kwargs: Key-value pairs to update in the profile
            
        Returns:
            bool: True if successful, False if user doesn't exist
        """
        if user_id not in self.user_profiles:
            return False
        
        profile = self.user_profiles[user_id]
        
        for key, value in kwargs.items():
            if key == "interests" and isinstance(value, list):
                profile["interests"] = value
            elif key == "expertise_level" and value in self.expertise_levels:
                profile["expertise_level"] = value
            elif key in ["result_preferences", "feedback_history"] and isinstance(value, dict):
                profile[key].update(value)
            else:
                profile[key] = value
        
        profile["last_updated"] = time.time()
        self._save_profiles()
        return True
    
    def add_user_interest(self, user_id: str, interest: str) -> bool:
        """
        Add an interest to a user's profile
        
        Args:
            user_id (str): User identifier
            interest (str): Interest to add
            
        Returns:
            bool: True if successful, False if user doesn't exist
        """
        if user_id not in self.user_profiles:
            return False
        
        if interest not in self.user_profiles[user_id]["interests"]:
            self.user_profiles[user_id]["interests"].append(interest)
            self.user_profiles[user_id]["last_updated"] = time.time()
            self._save_profiles()
        
        return True
    
    def set_expertise_level(self, user_id: str, level: str) -> bool:
        """
        Set the expertise level for a user
        
        Args:
            user_id (str): User identifier
            level (str): Expertise level (beginner, intermediate, advanced, expert)
            
        Returns:
            bool: True if successful, False if user doesn't exist or level is invalid
        """
        if user_id not in self.user_profiles or level not in self.expertise_levels:
            return False
        
        self.user_profiles[user_id]["expertise_level"] = level
        self.user_profiles[user_id]["last_updated"] = time.time()
        self._save_profiles()
        return True
    
    def record_user_feedback(self, user_id: str, result_id: str, feedback: str) -> bool:
        """
        Record user feedback on a retrieval result
        
        Args:
            user_id (str): User identifier
            result_id (str): ID of the result that received feedback
            feedback (str): Feedback type (helpful, somewhat_helpful, not_helpful)
            
        Returns:
            bool: True if successful, False if user doesn't exist or feedback is invalid
        """
        if user_id not in self.user_profiles or feedback not in self.feedback_weights:
            return False
        
        # Store the feedback
        if "feedback_history" not in self.user_profiles[user_id]:
            self.user_profiles[user_id]["feedback_history"] = {}
        
        self.user_profiles[user_id]["feedback_history"][result_id] = {
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        self.user_profiles[user_id]["last_updated"] = time.time()
        self._save_profiles()
        return True
    
    def _enhance_query_with_interests(self, query: str, interests: List[str]) -> str:
        """
        Enhance a query with user interests
        
        Args:
            query (str): Original query
            interests (list): User interests
            
        Returns:
            str: Enhanced query
        """
        # Add top 3 interests to the query with lower weight
        if interests:
            top_interests = interests[:3]
            interest_terms = " ".join(top_interests)
            enhanced_query = f"{query} {interest_terms}"
            return enhanced_query
        
        return query
    
    def _score_results_for_user(self, results: List[Dict], user_id: str) -> List[Dict]:
        """
        Score and rerank results based on user profile
        
        Args:
            results (list): Original retrieval results
            user_id (str): User identifier
            
        Returns:
            list: Reranked results with personalized scores
        """
        if user_id not in self.user_profiles:
            return results
        
        profile = self.user_profiles[user_id]
        scored_results = []
        
        for result in results:
            # Start with the original similarity score
            original_score = result.get("score", 0.5)
            personalized_score = original_score
            
            # Factor 1: Adjust based on user interests
            interest_bonus = 0
            result_text = result.get("text", "").lower()
            for interest in profile.get("interests", []):
                if interest.lower() in result_text:
                    interest_bonus += 0.1  # Add a small bonus for each matching interest
            
            # Cap the interest bonus
            interest_bonus = min(interest_bonus, 0.3)
            personalized_score += interest_bonus
            
            # Factor 2: Consider previous feedback on this item
            feedback_history = profile.get("feedback_history", {})
            if result.get("id") in feedback_history:
                feedback = feedback_history[result["id"]]["feedback"]
                feedback_weight = self.feedback_weights.get(feedback, 0)
                personalized_score += feedback_weight
            
            # Factor 3: Adjust for expertise level
            expertise_level = profile.get("expertise_level", "beginner")
            expertise_weight = self.expertise_levels.get(expertise_level, 0.5)
            
            # For beginners, prioritize simpler content
            # For experts, prioritize more complex content
            # This is a simplified approach - in a real system, you'd have content complexity metadata
            word_count = len(result.get("text", "").split())
            if expertise_level == "beginner" and word_count > 100:
                personalized_score -= 0.1  # Penalize long, potentially complex content for beginners
            elif expertise_level == "expert" and word_count < 50:
                personalized_score -= 0.1  # Penalize very short content for experts
            
            # Create a copy of the result with the personalized score
            scored_result = result.copy()
            scored_result["original_score"] = original_score
            scored_result["personalized_score"] = personalized_score
            scored_result["personalization_factors"] = {
                "interest_bonus": interest_bonus,
                "expertise_adjustment": expertise_weight,
                "feedback_adjustment": feedback_weight if result.get("id") in feedback_history else 0
            }
            
            scored_results.append(scored_result)
        
        # Sort by personalized score
        scored_results.sort(key=lambda x: x["personalized_score"], reverse=True)
        return scored_results
    
    def retrieve(self, query: str, user_id: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieve information personalized for a specific user
        
        Args:
            query (str): The search query
            user_id (str, optional): User identifier for personalization
            top_k (int): Number of results to return
            
        Returns:
            list: Personalized retrieval results
        """
        # If no user_id provided, perform standard retrieval
        if not user_id or user_id not in self.user_profiles:
            return self.memory_system.retrieve(query, top_k=top_k)
        
        # Get user profile
        profile = self.user_profiles[user_id]
        
        # Enhance query with user interests
        enhanced_query = self._enhance_query_with_interests(query, profile.get("interests", []))
        
        # Retrieve results with enhanced query
        results = self.memory_system.retrieve(enhanced_query, top_k=top_k*2)  # Get more results for reranking
        
        # Score and rerank results based on user profile
        personalized_results = self._score_results_for_user(results, user_id)
        
        # Record this interaction
        if "interaction_history" not in profile:
            profile["interaction_history"] = []
        
        profile["interaction_history"].append({
            "query": query,
            "enhanced_query": enhanced_query,
            "timestamp": time.time(),
            "num_results": len(personalized_results[:top_k])
        })
        
        self.user_profiles[user_id]["last_updated"] = time.time()
        self._save_profiles()
        
        return personalized_results[:top_k]
    
    def retrieve_with_explanation(self, query: str, user_id: str, top_k: int = 5) -> Dict:
        """
        Retrieve information with explanation of personalization
        
        Args:
            query (str): The search query
            user_id (str): User identifier
            top_k (int): Number of results to return
            
        Returns:
            dict: Results and personalization explanation
        """
        # Perform personalized retrieval
        results = self.retrieve(query, user_id, top_k)
        
        # If no user profile, return basic results
        if user_id not in self.user_profiles:
            return {
                "results": results,
                "explanation": "No personalization applied (user profile not found)"
            }
        
        # Get user profile
        profile = self.user_profiles[user_id]
        
        # Create explanation
        explanation = {
            "original_query": query,
            "user_interests_applied": profile.get("interests", [])[:3],
            "expertise_level": profile.get("expertise_level", "beginner"),
            "num_past_interactions": len(profile.get("interaction_history", [])),
            "num_feedback_items": len(profile.get("feedback_history", {})),
            "personalization_applied": True
        }
        
        return {
            "results": results,
            "explanation": explanation
        }
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """
        Get a user's profile
        
        Args:
            user_id (str): User identifier
            
        Returns:
            dict: User profile or None if not found
        """
        return self.user_profiles.get(user_id)
    
    def adapt_to_expertise(self, content: str, user_id: str) -> str:
        """
        Adapt content based on user expertise level
        
        Args:
            content (str): Original content
            user_id (str): User identifier
            
        Returns:
            str: Adapted content
        """
        if user_id not in self.user_profiles:
            return content
        
        expertise_level = self.user_profiles[user_id].get("expertise_level", "beginner")
        
        # This is a simplified approach - in a real system, you'd use more sophisticated
        # content adaptation techniques, possibly with an LLM
        
        if expertise_level == "beginner":
            # Add explanatory notes for beginners
            adapted_content = content + "\n\n(Note: This is a simplified explanation suitable for beginners.)"
        elif expertise_level == "intermediate":
            # Keep as is for intermediate users
            adapted_content = content
        elif expertise_level in ["advanced", "expert"]:
            # Add technical details for advanced users
            adapted_content = content + "\n\n(For advanced users: Consider exploring related technical concepts.)"
        else:
            adapted_content = content
        
        return adapted_content


# Example usage
if __name__ == "__main__":
    # Create a simple mock memory system for testing
    class MockMemorySystem:
        def __init__(self):
            self.items = {
                "item1": {
                    "id": "item1",
                    "text": "Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data.",
                    "metadata": {"category": "AI", "complexity": "beginner"}
                },
                "item2": {
                    "id": "item2",
                    "text": "Natural language processing (NLP) helps computers understand, interpret, and generate human language in a valuable way.",
                    "metadata": {"category": "NLP", "complexity": "intermediate"}
                },
                "item3": {
                    "id": "item3",
                    "text": "Vector databases are specialized database systems optimized for similarity search operations on high-dimensional vectors.",
                    "metadata": {"category": "Databases", "complexity": "advanced"}
                },
                "item4": {
                    "id": "item4",
                    "text": "Large language models like GPT-4 use transformer architectures with attention mechanisms to process and generate text.",
                    "metadata": {"category": "NLP", "complexity": "advanced"}
                },
                "item5": {
                    "id": "item5",
                    "text": "Python is a popular programming language for AI development due to its simplicity and extensive libraries.",
                    "metadata": {"category": "Programming", "complexity": "beginner"}
                }
            }

        def retrieve(self, query, top_k=5):
            # Simple keyword matching for demonstration
            results = []
            query_terms = query.lower().split()

            for item_id, item in self.items.items():
                text = item["text"].lower()
                # Count how many query terms appear in the text
                matches = sum(term in text for term in query_terms)
                if matches > 0:
                    # Create a copy of the item with a score
                    result = item.copy()
                    result["score"] = matches / len(query_terms)
                    results.append(result)

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    # Test the personalized retrieval system
    print("Testing Personalized Retrieval System...")
    memory = MockMemorySystem()
    retrieval_system = PersonalizedRetrievalSystem(memory, "example_user_profiles.json")

    # Create user profiles
    retrieval_system.create_user_profile("user1", {
        "interests": ["AI", "machine learning", "Python"],
        "expertise_level": "beginner"
    })

    retrieval_system.create_user_profile("user2", {
        "interests": ["NLP", "language models", "transformers"],
        "expertise_level": "expert"
    })

    # Test basic retrieval
    query = "AI technology"
    print(f"\nOriginal query: {query}")
    
    # Get results for user1 (beginner interested in AI)
    print("\nResults for beginner user interested in AI:")
    results_user1 = retrieval_system.retrieve(query, "user1", top_k=3)
    for result in results_user1:
        print(f"\n- {result['text']}")
        print(f"  Original score: {result.get('original_score', 0):.2f}")
        print(f"  Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"  Personalization factors: {result['personalization_factors']}")

    # Get results for user2 (expert interested in NLP)
    print("\nResults for expert user interested in NLP:")
    results_user2 = retrieval_system.retrieve(query, "user2", top_k=3)
    for result in results_user2:
        print(f"\n- {result['text']}")
        print(f"  Original score: {result.get('original_score', 0):.2f}")
        print(f"  Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"  Personalization factors: {result['personalization_factors']}")

    # Test retrieval with explanation
    print("\nRetrieval with explanation for user1:")
    response = retrieval_system.retrieve_with_explanation(query, "user1", top_k=2)
    
    print("\nExplanation:")
    explanation = response['explanation']
    print(f"Original query: {explanation['original_query']}")
    print(f"User interests applied: {explanation['user_interests_applied']}")
    print(f"Expertise level: {explanation['expertise_level']}")
    
    # Test content adaptation
    content = "Vector embeddings represent semantic meaning in high-dimensional space."
    print("\nContent adaptation examples:")
    
    adapted_beginner = retrieval_system.adapt_to_expertise(content, "user1")
    print(f"\nFor beginner:\n{adapted_beginner}")
    
    adapted_expert = retrieval_system.adapt_to_expertise(content, "user2")
    print(f"\nFor expert:\n{adapted_expert}")
    
    # Test feedback recording
    print("\nRecording user feedback...")
    retrieval_system.record_user_feedback("user1", "item1", "helpful")
    retrieval_system.record_user_feedback("user1", "item3", "not_helpful")
    
    # Retrieve again to see the effect of feedback
    print("\nResults after feedback for user1:")
    results_after_feedback = retrieval_system.retrieve(query, "user1", top_k=3)
    for result in results_after_feedback:
        print(f"\n- {result['text']}")
        print(f"  Original score: {result.get('original_score', 0):.2f}")
        print(f"  Personalized score: {result.get('personalized_score', 0):.2f}")
        if 'personalization_factors' in result:
            print(f"  Personalization factors: {result['personalization_factors']}")
