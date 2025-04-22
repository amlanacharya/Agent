"""
Memory Types Exercises
---------------------
This file contains solutions for the exercises from Lesson 1: Memory Types for AI Agents.
"""

import time
import json
import os
from collections import deque


class ConversationHistorySystem:
    """
    Exercise 1: Implement a Conversation History System
    
    This system tracks conversation history, summarizes long conversations,
    and retrieves relevant past interactions.
    """
    
    def __init__(self, max_history=50):
        """Initialize the conversation history system"""
        self.history = []
        self.max_history = max_history
        self.summaries = []
    
    def add_interaction(self, user_input, agent_response):
        """Add a new interaction to the history"""
        interaction = {
            "timestamp": time.time(),
            "user_input": user_input,
            "agent_response": agent_response
        }
        
        self.history.append(interaction)
        
        # Keep history within max size
        if len(self.history) > self.max_history:
            # Create a summary of oldest interactions before removing them
            self._create_summary(5)
    
    def _create_summary(self, num_interactions):
        """Summarize the oldest n interactions"""
        if len(self.history) <= num_interactions:
            return
        
        # Get the oldest interactions
        to_summarize = self.history[:num_interactions]
        
        # Create a simple summary (in a real system, this would use an LLM)
        summary = {
            "start_time": to_summarize[0]["timestamp"],
            "end_time": to_summarize[-1]["timestamp"],
            "num_interactions": len(to_summarize),
            "topics": self._extract_topics(to_summarize),
            "summary_text": f"Conversation with {len(to_summarize)} interactions"
        }
        
        # Add to summaries
        self.summaries.append(summary)
        
        # Remove summarized interactions from history
        self.history = self.history[num_interactions:]
    
    def _extract_topics(self, interactions):
        """Extract topics from interactions (simplified version)"""
        # In a real system, this would use NLP techniques
        all_text = " ".join([
            interaction["user_input"] + " " + interaction["agent_response"]
            for interaction in interactions
        ])
        
        # Simple keyword extraction (very basic)
        common_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]
        words = [word.lower() for word in all_text.split() if word.lower() not in common_words]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top 5 words as topics
        topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic[0] for topic in topics]
    
    def get_recent_history(self, n=10):
        """Get the n most recent interactions"""
        return self.history[-n:] if n < len(self.history) else self.history
    
    def search_history(self, query):
        """Search for relevant past interactions"""
        # In a real system, this would use semantic search
        # For this exercise, we'll use simple keyword matching
        results = []
        
        for interaction in self.history:
            user_input = interaction["user_input"].lower()
            agent_response = interaction["agent_response"].lower()
            query_lower = query.lower()
            
            if query_lower in user_input or query_lower in agent_response:
                results.append(interaction)
        
        # Also search in summaries
        for summary in self.summaries:
            if query.lower() in " ".join(summary["topics"]).lower():
                # Add a placeholder for summarized content
                results.append({
                    "timestamp": summary["end_time"],
                    "user_input": f"[Summary containing '{query}']",
                    "agent_response": f"[Summary of {summary['num_interactions']} interactions about {', '.join(summary['topics'])}]",
                    "is_summary": True
                })
        
        return sorted(results, key=lambda x: x["timestamp"])
    
    def get_full_history(self):
        """Get the full conversation history including summaries"""
        # Combine history and summaries, sorted by timestamp
        full_history = self.history.copy()
        
        for summary in self.summaries:
            full_history.append({
                "timestamp": summary["end_time"],
                "user_input": "[Summarized interactions]",
                "agent_response": summary["summary_text"],
                "is_summary": True,
                "topics": summary["topics"]
            })
        
        return sorted(full_history, key=lambda x: x["timestamp"])


class UserProfileManager:
    """
    Exercise 2: Build a User Profile Manager
    
    This system stores and retrieves user preferences, updates user information
    over time, and generates personalized responses.
    """
    
    def __init__(self, storage_path="user_profiles.json"):
        """Initialize the user profile manager"""
        self.storage_path = storage_path
        self.profiles = self._load_profiles()
    
    def _load_profiles(self):
        """Load profiles from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_profiles(self):
        """Save profiles to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.profiles, f)
    
    def create_profile(self, user_id, initial_data=None):
        """Create a new user profile"""
        if user_id in self.profiles:
            return False  # Profile already exists
        
        self.profiles[user_id] = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "preferences": {},
            "interaction_history": [],
            "topics_of_interest": [],
            **(initial_data or {})
        }
        
        self._save_profiles()
        return True
    
    def get_profile(self, user_id):
        """Get a user profile"""
        return self.profiles.get(user_id)
    
    def update_preference(self, user_id, preference_name, preference_value):
        """Update a user preference"""
        if user_id not in self.profiles:
            return False
        
        if "preferences" not in self.profiles[user_id]:
            self.profiles[user_id]["preferences"] = {}
        
        self.profiles[user_id]["preferences"][preference_name] = preference_value
        self.profiles[user_id]["last_updated"] = time.time()
        
        self._save_profiles()
        return True
    
    def add_interaction(self, user_id, interaction_type, content):
        """Add an interaction to the user's history"""
        if user_id not in self.profiles:
            return False
        
        if "interaction_history" not in self.profiles[user_id]:
            self.profiles[user_id]["interaction_history"] = []
        
        # Keep only the last 20 interactions
        if len(self.profiles[user_id]["interaction_history"]) >= 20:
            self.profiles[user_id]["interaction_history"].pop(0)
        
        self.profiles[user_id]["interaction_history"].append({
            "timestamp": time.time(),
            "type": interaction_type,
            "content": content
        })
        
        self.profiles[user_id]["last_updated"] = time.time()
        
        self._save_profiles()
        return True
    
    def add_topic_of_interest(self, user_id, topic):
        """Add a topic of interest to the user's profile"""
        if user_id not in self.profiles:
            return False
        
        if "topics_of_interest" not in self.profiles[user_id]:
            self.profiles[user_id]["topics_of_interest"] = []
        
        if topic not in self.profiles[user_id]["topics_of_interest"]:
            self.profiles[user_id]["topics_of_interest"].append(topic)
            self.profiles[user_id]["last_updated"] = time.time()
            self._save_profiles()
        
        return True
    
    def generate_personalized_response(self, user_id, message, response_templates):
        """Generate a personalized response based on user profile"""
        if user_id not in self.profiles:
            return "I don't have a profile for you yet. Let's get to know each other!"
        
        profile = self.profiles[user_id]
        
        # Check for name preference
        name = profile.get("preferences", {}).get("name", "there")
        greeting = f"Hi {name}"
        
        # Check for time preference
        time_preference = profile.get("preferences", {}).get("time_format", "12h")
        current_time = time.localtime()
        if time_preference == "12h":
            time_str = time.strftime("%I:%M %p", current_time)
        else:
            time_str = time.strftime("%H:%M", current_time)
        
        # Check for topics of interest
        topics = profile.get("topics_of_interest", [])
        topic_str = ""
        if topics:
            topic_str = f"I remember you're interested in {', '.join(topics)}. "
        
        # Select a response template
        template = response_templates.get("default", "I don't have a specific response for that.")
        
        # Fill in the template
        response = template.format(
            greeting=greeting,
            time=time_str,
            topics=topic_str,
            message=message
        )
        
        return response


class KnowledgeTrackingSystem:
    """
    Exercise 3: Create a Knowledge Tracking System
    
    This system tracks what the agent has learned, identifies knowledge gaps,
    and updates knowledge based on new information.
    """
    
    def __init__(self, storage_path="knowledge_tracking.json"):
        """Initialize the knowledge tracking system"""
        self.storage_path = storage_path
        self.knowledge = self._load_knowledge()
        
        # Confidence thresholds
        self.high_confidence = 0.8
        self.medium_confidence = 0.5
        self.low_confidence = 0.2
    
    def _load_knowledge(self):
        """Load knowledge from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"facts": {}, "relationships": {}, "gaps": []}
        return {"facts": {}, "relationships": {}, "gaps": []}
    
    def _save_knowledge(self):
        """Save knowledge to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.knowledge, f)
    
    def add_fact(self, fact_id, fact_text, confidence=0.5, source="user"):
        """Add a new fact to the knowledge base"""
        self.knowledge["facts"][fact_id] = {
            "text": fact_text,
            "confidence": confidence,
            "source": source,
            "added_at": time.time(),
            "last_updated": time.time(),
            "verification_status": "unverified" if confidence < self.high_confidence else "verified"
        }
        
        # Check if this fact resolves any knowledge gaps
        self._check_gap_resolution(fact_text)
        
        self._save_knowledge()
        return True
    
    def add_relationship(self, source_id, target_id, relationship_type, confidence=0.5):
        """Add a relationship between facts"""
        if source_id not in self.knowledge["facts"] or target_id not in self.knowledge["facts"]:
            return False
        
        if source_id not in self.knowledge["relationships"]:
            self.knowledge["relationships"][source_id] = []
        
        # Check if relationship already exists
        for rel in self.knowledge["relationships"][source_id]:
            if rel["target"] == target_id and rel["type"] == relationship_type:
                # Update confidence if new confidence is higher
                if confidence > rel["confidence"]:
                    rel["confidence"] = confidence
                    rel["last_updated"] = time.time()
                    self._save_knowledge()
                return True
        
        # Add new relationship
        self.knowledge["relationships"][source_id].append({
            "target": target_id,
            "type": relationship_type,
            "confidence": confidence,
            "added_at": time.time(),
            "last_updated": time.time()
        })
        
        self._save_knowledge()
        return True
    
    def update_fact(self, fact_id, fact_text=None, confidence=None, source=None):
        """Update an existing fact"""
        if fact_id not in self.knowledge["facts"]:
            return False
        
        fact = self.knowledge["facts"][fact_id]
        
        if fact_text is not None:
            fact["text"] = fact_text
        
        if confidence is not None:
            fact["confidence"] = confidence
            # Update verification status based on new confidence
            fact["verification_status"] = "unverified" if confidence < self.high_confidence else "verified"
        
        if source is not None:
            fact["source"] = source
        
        fact["last_updated"] = time.time()
        
        self._save_knowledge()
        return True
    
    def get_fact(self, fact_id):
        """Get a specific fact"""
        return self.knowledge["facts"].get(fact_id)
    
    def get_related_facts(self, fact_id):
        """Get facts related to a specific fact"""
        if fact_id not in self.knowledge["relationships"]:
            return []
        
        related = []
        for rel in self.knowledge["relationships"][fact_id]:
            target_id = rel["target"]
            if target_id in self.knowledge["facts"]:
                related.append({
                    "id": target_id,
                    "text": self.knowledge["facts"][target_id]["text"],
                    "relationship": rel["type"],
                    "confidence": rel["confidence"]
                })
        
        return related
    
    def add_knowledge_gap(self, topic, question=None, priority="medium"):
        """Add a knowledge gap"""
        gap = {
            "topic": topic,
            "question": question,
            "priority": priority,
            "identified_at": time.time(),
            "status": "open"
        }
        
        self.knowledge["gaps"].append(gap)
        self._save_knowledge()
        return len(self.knowledge["gaps"]) - 1  # Return gap index
    
    def _check_gap_resolution(self, fact_text):
        """Check if a new fact resolves any knowledge gaps"""
        for gap in self.knowledge["gaps"]:
            if gap["status"] == "open":
                # Simple keyword matching (in a real system, use semantic matching)
                if gap["topic"].lower() in fact_text.lower():
                    gap["status"] = "resolved"
                    gap["resolved_at"] = time.time()
    
    def get_open_gaps(self):
        """Get all open knowledge gaps"""
        return [gap for gap in self.knowledge["gaps"] if gap["status"] == "open"]
    
    def get_facts_by_confidence(self, min_confidence=0.0, max_confidence=1.0):
        """Get facts within a confidence range"""
        return {
            fact_id: fact for fact_id, fact in self.knowledge["facts"].items()
            if min_confidence <= fact["confidence"] <= max_confidence
        }
    
    def get_unverified_facts(self):
        """Get all unverified facts"""
        return {
            fact_id: fact for fact_id, fact in self.knowledge["facts"].items()
            if fact["verification_status"] == "unverified"
        }
    
    def verify_fact(self, fact_id, verified=True, new_confidence=None):
        """Mark a fact as verified or unverified"""
        if fact_id not in self.knowledge["facts"]:
            return False
        
        fact = self.knowledge["facts"][fact_id]
        
        if verified:
            fact["verification_status"] = "verified"
            if new_confidence is not None:
                fact["confidence"] = new_confidence
            elif fact["confidence"] < self.high_confidence:
                fact["confidence"] = self.high_confidence
        else:
            fact["verification_status"] = "unverified"
            if new_confidence is not None:
                fact["confidence"] = new_confidence
        
        fact["last_updated"] = time.time()
        
        self._save_knowledge()
        return True


# Example usage
if __name__ == "__main__":
    print("Memory Types Exercises - Example Usage")
    print("=" * 50)
    
    # Exercise 1: Conversation History System
    print("\nExercise 1: Conversation History System")
    print("-" * 50)
    
    convo = ConversationHistorySystem(max_history=10)
    
    # Add some sample interactions
    convo.add_interaction("Hello, how are you?", "I'm doing well, thank you for asking!")
    convo.add_interaction("What's the weather like today?", "I don't have access to real-time weather data, but I can help you find that information.")
    convo.add_interaction("Tell me about Python programming.", "Python is a high-level, interpreted programming language known for its readability and versatility.")
    convo.add_interaction("How do I create a list in Python?", "You can create a list in Python using square brackets. For example: my_list = [1, 2, 3]")
    
    # Get recent history
    print("Recent conversation history:")
    for interaction in convo.get_recent_history(2):
        print(f"User: {interaction['user_input']}")
        print(f"Agent: {interaction['agent_response']}")
        print()
    
    # Search history
    print("Search results for 'Python':")
    results = convo.search_history("Python")
    for interaction in results:
        print(f"User: {interaction['user_input']}")
        print(f"Agent: {interaction['agent_response']}")
        print()
    
    # Exercise 2: User Profile Manager
    print("\nExercise 2: User Profile Manager")
    print("-" * 50)
    
    profile_manager = UserProfileManager("example_profiles.json")
    
    # Create a profile
    profile_manager.create_profile("user123", {
        "preferences": {
            "name": "Alice",
            "time_format": "24h",
            "language": "English"
        }
    })
    
    # Update preferences
    profile_manager.update_preference("user123", "theme", "dark")
    
    # Add topics of interest
    profile_manager.add_topic_of_interest("user123", "Python")
    profile_manager.add_topic_of_interest("user123", "AI")
    
    # Generate personalized response
    templates = {
        "default": "{greeting}! It's currently {time}. {topics}How can I help you with your message: '{message}'?"
    }
    
    response = profile_manager.generate_personalized_response(
        "user123", 
        "I want to learn more about machine learning", 
        templates
    )
    
    print("Personalized response:")
    print(response)
    print()
    
    # Exercise 3: Knowledge Tracking System
    print("\nExercise 3: Knowledge Tracking System")
    print("-" * 50)
    
    knowledge_tracker = KnowledgeTrackingSystem("example_knowledge.json")
    
    # Add some facts
    knowledge_tracker.add_fact("fact1", "Python is a programming language", 0.9, "documentation")
    knowledge_tracker.add_fact("fact2", "Python was created by Guido van Rossum", 0.8, "interview")
    knowledge_tracker.add_fact("fact3", "Python 3.9 was released in 2020", 0.7, "release notes")
    
    # Add relationships
    knowledge_tracker.add_relationship("fact1", "fact2", "created_by")
    knowledge_tracker.add_relationship("fact1", "fact3", "has_version")
    
    # Add a knowledge gap
    knowledge_tracker.add_knowledge_gap("Python performance optimization", "How can I make Python code run faster?")
    
    # Get facts by confidence
    high_confidence_facts = knowledge_tracker.get_facts_by_confidence(0.8, 1.0)
    print("High confidence facts:")
    for fact_id, fact in high_confidence_facts.items():
        print(f"- {fact['text']} (Confidence: {fact['confidence']})")
    
    # Get related facts
    related = knowledge_tracker.get_related_facts("fact1")
    print("\nFacts related to 'Python is a programming language':")
    for fact in related:
        print(f"- {fact['text']} (Relationship: {fact['relationship']})")
    
    # Get open knowledge gaps
    gaps = knowledge_tracker.get_open_gaps()
    print("\nOpen knowledge gaps:")
    for gap in gaps:
        print(f"- Topic: {gap['topic']}")
        if gap['question']:
            print(f"  Question: {gap['question']}")
    
    # Clean up example files
    if os.path.exists("example_profiles.json"):
        os.remove("example_profiles.json")
    
    if os.path.exists("example_knowledge.json"):
        os.remove("example_knowledge.json")
