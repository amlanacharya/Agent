# 🧠 Module 2: Memory Systems - Lesson 1 💾

![Memory Types](https://media.giphy.com/media/l0HlQXlQ3nHyLMvte/giphy.gif)

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧩 Understand the **different memory types** for AI agents
- 🔄 Implement **working memory** for immediate context
- 📚 Build **short-term memory** for recent interactions
- 💾 Create **long-term memory** for persistent knowledge
- 🧠 Design a **memory architecture** that combines multiple memory types

---

## 📚 Introduction to Memory Systems for AI Agents

![Brain Memory](https://media.giphy.com/media/3o7TKsQ8Xb3gcGEgZW/giphy.gif)

Memory is a fundamental component of intelligent agents. Without memory, agents would be stateless, responding to each input as if it were the first interaction. Memory systems allow agents to:

1. **Remember past interactions** with users
2. **Maintain context** throughout a conversation
3. **Learn from experience** over time
4. **Store and retrieve knowledge** when needed
5. **Develop personalized responses** based on user history

Just as human memory has different systems (working memory, short-term memory, long-term memory), AI agents benefit from a similar architecture.

---

## 🧩 Types of Memory for AI Agents

### 1. Working Memory (Buffer Memory)

![Working Memory](https://media.giphy.com/media/3o7TKT6gL5B7Lzq3re/giphy.gif)

**Working memory** holds the immediate context of the current interaction. It's like the "RAM" of an AI agent:

- **Purpose**: Maintain immediate context for the current task
- **Capacity**: Limited (typically just the current conversation turn)
- **Persistence**: Very short (cleared after task completion)
- **Access Pattern**: Direct and immediate

**Example Implementation:**

```python
class WorkingMemory:
    def __init__(self):
        self.current_context = None
    
    def set_context(self, context):
        """Set the current working context"""
        self.current_context = context
    
    def get_context(self):
        """Retrieve the current working context"""
        return self.current_context
    
    def clear(self):
        """Clear the working memory"""
        self.current_context = None
```

### 2. Short-Term Memory (Conversation Memory)

![Short-Term Memory](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

**Short-term memory** stores recent interactions, allowing the agent to maintain conversation flow:

- **Purpose**: Track recent conversation history
- **Capacity**: Limited (last N interactions)
- **Persistence**: Short (typically cleared between sessions)
- **Access Pattern**: Sequential, recent-first

**Example Implementation:**

```python
class ShortTermMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.memory = []
    
    def add(self, item):
        """Add an item to short-term memory"""
        self.memory.append(item)
        # Keep only the most recent items within capacity
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
    
    def get_recent(self, n=None):
        """Get the n most recent items (or all if n is None)"""
        if n is None or n > len(self.memory):
            return self.memory
        return self.memory[-n:]
    
    def clear(self):
        """Clear the short-term memory"""
        self.memory = []
```

### 3. Long-Term Memory (Persistent Storage)

![Long-Term Memory](https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif)

**Long-term memory** provides persistent storage for knowledge and experiences:

- **Purpose**: Store knowledge and experiences over time
- **Capacity**: Large (potentially unlimited)
- **Persistence**: Long (persists across sessions)
- **Access Pattern**: Associative, query-based

**Example Implementation:**

```python
import json
import os

class LongTermMemory:
    def __init__(self, file_path="long_term_memory.json"):
        self.file_path = file_path
        self.memory = self._load_memory()
    
    def _load_memory(self):
        """Load memory from persistent storage"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_memory(self):
        """Save memory to persistent storage"""
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f)
    
    def store(self, key, value):
        """Store information in long-term memory"""
        self.memory[key] = value
        self._save_memory()
    
    def retrieve(self, key):
        """Retrieve information from long-term memory"""
        return self.memory.get(key)
    
    def forget(self, key):
        """Remove information from long-term memory"""
        if key in self.memory:
            del self.memory[key]
            self._save_memory()
    
    def clear(self):
        """Clear all long-term memory"""
        self.memory = {}
        self._save_memory()
```

### 4. Episodic Memory (Experience Memory)

![Episodic Memory](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

**Episodic memory** stores specific experiences or interactions:

- **Purpose**: Remember specific events or interactions
- **Capacity**: Medium to large
- **Persistence**: Medium to long
- **Access Pattern**: Temporal, event-based

**Example Implementation:**

```python
import time

class EpisodicMemory:
    def __init__(self):
        self.episodes = []
    
    def record_episode(self, episode_type, content, metadata=None):
        """Record a new episode"""
        episode = {
            "timestamp": time.time(),
            "type": episode_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.episodes.append(episode)
        return len(self.episodes) - 1  # Return episode ID
    
    def get_episode(self, episode_id):
        """Retrieve a specific episode by ID"""
        if 0 <= episode_id < len(self.episodes):
            return self.episodes[episode_id]
        return None
    
    def get_episodes_by_type(self, episode_type):
        """Retrieve all episodes of a specific type"""
        return [ep for ep in self.episodes if ep["type"] == episode_type]
    
    def get_episodes_in_timeframe(self, start_time, end_time=None):
        """Retrieve episodes within a specific timeframe"""
        if end_time is None:
            end_time = time.time()
        
        return [
            ep for ep in self.episodes 
            if start_time <= ep["timestamp"] <= end_time
        ]
    
    def clear(self):
        """Clear all episodes"""
        self.episodes = []
```

---

## 🔄 Integrating Memory Systems

![Integration](https://media.giphy.com/media/3o7btNDyBs5dKdhTqM/giphy.gif)

A comprehensive agent memory architecture integrates these different memory types:

```python
class AgentMemorySystem:
    def __init__(self, storage_dir="agent_memory"):
        """Initialize the memory system"""
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize memory components
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(capacity=20)
        self.long_term = LongTermMemory(os.path.join(storage_dir, "long_term_memory.json"))
        self.episodic = EpisodicMemory()
    
    def process_interaction(self, user_input, agent_response):
        """Process a new interaction"""
        # Update working memory with current context
        self.working.set_context({
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        # Add to short-term memory
        self.short_term.add({
            "timestamp": time.time(),
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        # Record as an episode
        self.episodic.record_episode(
            episode_type="conversation",
            content={
                "user_input": user_input,
                "agent_response": agent_response
            }
        )
    
    def get_conversation_context(self, n_turns=5):
        """Get recent conversation context"""
        return self.short_term.get_recent(n_turns)
    
    def store_fact(self, key, value):
        """Store a fact in long-term memory"""
        self.long_term.store(key, value)
    
    def retrieve_fact(self, key):
        """Retrieve a fact from long-term memory"""
        return self.long_term.retrieve(key)
    
    def save_user_preference(self, preference_name, preference_value):
        """Save a user preference"""
        preferences = self.long_term.retrieve("user_preferences") or {}
        preferences[preference_name] = preference_value
        self.long_term.store("user_preferences", preferences)
    
    def get_user_preferences(self):
        """Get all user preferences"""
        return self.long_term.retrieve("user_preferences") or {}
```

---

## 💪 Practice Exercises

![Practice](https://media.giphy.com/media/3oKIPrc2ngFZ6BTyww/giphy.gif)

1. **Implement a Conversation History System**:
   - Create a memory system that tracks conversation history
   - Add functionality to summarize long conversations
   - Implement a method to retrieve relevant past interactions

2. **Build a User Profile Manager**:
   - Design a system to store and retrieve user preferences
   - Implement methods to update user information over time
   - Create a function that generates personalized responses based on user profile

3. **Create a Knowledge Tracking System**:
   - Implement a system that tracks what the agent has learned
   - Add functionality to identify knowledge gaps
   - Create methods to update knowledge based on new information

---

## 🔍 Key Concepts to Remember

![Key Concepts](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

1. **Memory Hierarchy**: Different memory types serve different purposes
2. **Persistence vs. Speed**: Trade-offs between persistence and access speed
3. **Context Management**: Effective memory systems maintain appropriate context
4. **Forgetting Mechanism**: Sometimes forgetting is as important as remembering
5. **Integration**: A complete memory system integrates multiple memory types

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Vector database fundamentals
- Embedding text for semantic search
- Implementing similarity search
- Building a more sophisticated retrieval system

---

## 📚 Resources

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [Pinecone Memory Systems](https://www.pinecone.io/learn/memory-systems/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## 🎯 Mini-Project Preview: Knowledge Base Assistant

![Knowledge Base](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Throughout this module, we'll be building a Knowledge Base Assistant that can:
- Store and retrieve information from a knowledge base
- Answer questions based on stored knowledge
- Learn new information from conversations
- Identify when it doesn't know something
- Provide citations for its answers

Start thinking about how you would implement these features using the memory systems we've discussed!

---

Happy coding! 🚀
