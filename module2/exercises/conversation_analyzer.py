"""
Module 2: Memory Systems - Conversation Memory Analyzer Exercise
---------------------------------------------------------------
This file contains the implementation of Exercise 3 from Lesson 4:
A Conversation Memory Analyzer that extracts insights from conversation history.
"""

import time
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union, Set


class ConversationMemoryAnalyzer:
    """
    Exercise 3: Conversation Memory Analyzer
    
    This system analyzes conversation history to extract key topics,
    identify patterns, generate summaries, and provide insights about
    conversation flow.
    """
    
    def __init__(self, conversation_memory=None, storage_path="conversation_analysis.json"):
        """
        Initialize the conversation memory analyzer
        
        Args:
            conversation_memory: The conversation memory system to analyze
                                (must have a method to retrieve conversation history)
            storage_path (str): Path to store analysis results
        """
        self.conversation_memory = conversation_memory
        self.storage_path = storage_path
        self.analysis_cache = {}
        self.stopwords = self._load_stopwords()
        
        # Load previous analysis if available
        self._load_analysis()
    
    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords to filter out from topic extraction"""
        # Common English stopwords
        return {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
            "be", "been", "being", "in", "on", "at", "to", "for", "with", "by",
            "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "from", "up", "down", "of", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
            "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
            "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
            "won", "wouldn", "i", "me", "my", "myself", "we", "our", "ours", 
            "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", 
            "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", 
            "itself", "they", "them", "their", "theirs", "themselves", "what", 
            "which", "who", "whom", "this", "that", "these", "those", "am", "have", 
            "has", "had", "do", "does", "did", "doing", "would", "could", "should",
            "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
            "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd",
            "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
            "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
            "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't",
            "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's",
            "what's", "here's", "there's", "when's", "where's", "why's", "how's"
        }
    
    def _load_analysis(self) -> None:
        """Load previous analysis from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.analysis_cache = json.load(f)
            except json.JSONDecodeError:
                self.analysis_cache = {}
    
    def _save_analysis(self) -> None:
        """Save analysis to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.analysis_cache, f)
    
    def extract_key_topics(self, conversation: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Extract key topics from a conversation
        
        Args:
            conversation (list): List of conversation turns
            top_n (int): Number of top topics to extract
            
        Returns:
            list: Top topics with their frequency and context
        """
        # Extract all text from the conversation
        all_text = ""
        for turn in conversation:
            if "user_input" in turn:
                all_text += " " + turn["user_input"]
            if "agent_response" in turn:
                all_text += " " + turn["agent_response"]
        
        # Clean and tokenize the text
        words = self._tokenize_text(all_text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get the most common words as topics
        top_words = word_counts.most_common(top_n + 10)  # Get extra to filter stopwords
        
        # Filter out stopwords and single characters
        topics = []
        for word, count in top_words:
            if word not in self.stopwords and len(word) > 1:
                # Find context for this topic
                context = self._find_topic_context(word, conversation)
                
                topics.append({
                    "topic": word,
                    "frequency": count,
                    "context": context
                })
                
                if len(topics) >= top_n:
                    break
        
        return topics
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]{2,}\b', text)
        
        return words
    
    def _find_topic_context(self, topic: str, conversation: List[Dict]) -> List[str]:
        """
        Find context snippets for a topic
        
        Args:
            topic (str): The topic to find context for
            conversation (list): List of conversation turns
            
        Returns:
            list: Context snippets containing the topic
        """
        context_snippets = []
        
        for turn in conversation:
            # Check user input
            if "user_input" in turn and topic.lower() in turn["user_input"].lower():
                context_snippets.append(f"User: {turn['user_input']}")
            
            # Check agent response
            if "agent_response" in turn and topic.lower() in turn["agent_response"].lower():
                context_snippets.append(f"Agent: {turn['agent_response']}")
            
            # Limit to 3 context snippets
            if len(context_snippets) >= 3:
                break
        
        return context_snippets
    
    def identify_patterns(self, conversation: List[Dict]) -> Dict:
        """
        Identify patterns in user queries and agent responses
        
        Args:
            conversation (list): List of conversation turns
            
        Returns:
            dict: Identified patterns
        """
        patterns = {
            "question_frequency": 0,
            "common_question_types": [],
            "topic_shifts": [],
            "repeated_queries": [],
            "response_length_trend": "consistent",
            "user_sentiment_trend": "neutral"
        }
        
        # Analyze question frequency
        question_count = 0
        question_types = Counter()
        previous_topics = set()
        current_topics = set()
        query_counter = Counter()
        response_lengths = []
        
        for i, turn in enumerate(conversation):
            if "user_input" in turn:
                user_input = turn["user_input"]
                
                # Count questions
                if "?" in user_input:
                    question_count += 1
                    
                    # Identify question type
                    if user_input.lower().startswith("what"):
                        question_types["what"] += 1
                    elif user_input.lower().startswith("how"):
                        question_types["how"] += 1
                    elif user_input.lower().startswith("why"):
                        question_types["why"] += 1
                    elif user_input.lower().startswith("when"):
                        question_types["when"] += 1
                    elif user_input.lower().startswith("where"):
                        question_types["where"] += 1
                    elif user_input.lower().startswith("who"):
                        question_types["who"] += 1
                    elif user_input.lower().startswith("can") or user_input.lower().startswith("could"):
                        question_types["yes/no"] += 1
                    elif user_input.lower().startswith("is") or user_input.lower().startswith("are"):
                        question_types["yes/no"] += 1
                    else:
                        question_types["other"] += 1
                
                # Track repeated queries
                query_counter[user_input.lower()] += 1
                
                # Extract topics for topic shift detection
                if i > 0:
                    previous_topics = current_topics
                    current_topics = set(self._extract_simple_topics(user_input))
                    
                    # Check for topic shift
                    if previous_topics and current_topics and not previous_topics.intersection(current_topics):
                        patterns["topic_shifts"].append({
                            "turn_index": i,
                            "from_topics": list(previous_topics),
                            "to_topics": list(current_topics)
                        })
            
            # Track response lengths
            if "agent_response" in turn:
                response_lengths.append(len(turn["agent_response"].split()))
        
        # Set question frequency
        patterns["question_frequency"] = question_count / len(conversation) if conversation else 0
        
        # Set common question types
        patterns["common_question_types"] = [
            {"type": q_type, "count": count}
            for q_type, count in question_types.most_common(3)
        ]
        
        # Set repeated queries
        patterns["repeated_queries"] = [
            {"query": query, "count": count}
            for query, count in query_counter.items()
            if count > 1
        ]
        
        # Analyze response length trend
        if len(response_lengths) >= 3:
            first_half = response_lengths[:len(response_lengths)//2]
            second_half = response_lengths[len(response_lengths)//2:]
            
            first_half_avg = sum(first_half) / len(first_half)
            second_half_avg = sum(second_half) / len(second_half)
            
            if second_half_avg > first_half_avg * 1.5:
                patterns["response_length_trend"] = "increasing"
            elif second_half_avg < first_half_avg * 0.75:
                patterns["response_length_trend"] = "decreasing"
            else:
                patterns["response_length_trend"] = "consistent"
        
        return patterns
    
    def _extract_simple_topics(self, text: str) -> List[str]:
        """
        Extract simple topics from text
        
        Args:
            text (str): Text to extract topics from
            
        Returns:
            list: Simple topics
        """
        words = self._tokenize_text(text)
        return [word for word in words if word not in self.stopwords and len(word) > 1]
    
    def generate_conversation_summary(self, conversation: List[Dict]) -> Dict:
        """
        Generate a summary of the conversation
        
        Args:
            conversation (list): List of conversation turns
            
        Returns:
            dict: Conversation summary
        """
        if not conversation:
            return {
                "summary": "No conversation to summarize",
                "duration": 0,
                "turns": 0,
                "topics": []
            }
        
        # Calculate conversation duration
        start_time = conversation[0].get("timestamp", 0)
        end_time = conversation[-1].get("timestamp", 0)
        duration_seconds = end_time - start_time
        
        # Extract key topics
        topics = self.extract_key_topics(conversation, top_n=3)
        topic_names = [topic["topic"] for topic in topics]
        
        # Count turns
        turns = len(conversation)
        
        # Generate summary text
        if turns == 1:
            summary = f"Brief conversation about {', '.join(topic_names) if topic_names else 'various topics'}."
        else:
            # In a real system, this would use an LLM to generate a more coherent summary
            summary = f"Conversation with {turns} turns over {self._format_duration(duration_seconds)}. "
            summary += f"Main topics discussed: {', '.join(topic_names) if topic_names else 'various topics'}. "
            
            # Add info about question frequency
            patterns = self.identify_patterns(conversation)
            if patterns["question_frequency"] > 0.5:
                summary += "The conversation was question-intensive. "
            
            # Add info about topic shifts
            if len(patterns["topic_shifts"]) > 0:
                summary += f"The conversation shifted topics {len(patterns['topic_shifts'])} times. "
            
            # Add info about repeated queries
            if len(patterns["repeated_queries"]) > 0:
                summary += "Some queries were repeated during the conversation. "
        
        return {
            "summary": summary,
            "duration": duration_seconds,
            "turns": turns,
            "topics": topic_names,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to a human-readable string
        
        Args:
            seconds (float): Duration in seconds
            
        Returns:
            str: Formatted duration
        """
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def provide_conversation_insights(self, conversation: List[Dict]) -> Dict:
        """
        Provide insights about conversation flow and dynamics
        
        Args:
            conversation (list): List of conversation turns
            
        Returns:
            dict: Conversation insights
        """
        if not conversation:
            return {"insights": "No conversation to analyze"}
        
        # Get patterns and summary
        patterns = self.identify_patterns(conversation)
        summary = self.generate_conversation_summary(conversation)
        
        # Calculate additional metrics
        avg_user_input_length = 0
        avg_agent_response_length = 0
        user_input_count = 0
        agent_response_count = 0
        
        for turn in conversation:
            if "user_input" in turn:
                avg_user_input_length += len(turn["user_input"].split())
                user_input_count += 1
            if "agent_response" in turn:
                avg_agent_response_length += len(turn["agent_response"].split())
                agent_response_count += 1
        
        if user_input_count > 0:
            avg_user_input_length /= user_input_count
        
        if agent_response_count > 0:
            avg_agent_response_length /= agent_response_count
        
        # Generate insights
        insights = []
        
        # Insight about conversation length
        if summary["turns"] < 3:
            insights.append("This was a very brief conversation.")
        elif summary["turns"] < 10:
            insights.append("This was a short conversation.")
        else:
            insights.append("This was an extended conversation.")
        
        # Insight about question frequency
        if patterns["question_frequency"] > 0.7:
            insights.append("The user asked many questions, suggesting an information-seeking conversation.")
        elif patterns["question_frequency"] < 0.3:
            insights.append("The conversation had few questions, suggesting it was more statement-oriented.")
        
        # Insight about topic shifts
        if len(patterns["topic_shifts"]) == 0:
            insights.append("The conversation stayed focused on the same topics throughout.")
        elif len(patterns["topic_shifts"]) > 3:
            insights.append("The conversation shifted between multiple topics, suggesting a wide-ranging discussion.")
        
        # Insight about message length
        if avg_user_input_length < 10 and avg_agent_response_length > 50:
            insights.append("The user provided brief inputs while receiving detailed responses.")
        elif avg_user_input_length > 50 and avg_agent_response_length < 20:
            insights.append("The user provided detailed information with the agent giving brief responses.")
        
        # Insight about response length trend
        if patterns["response_length_trend"] == "increasing":
            insights.append("Agent responses became longer over time, possibly indicating increasing complexity or detail.")
        elif patterns["response_length_trend"] == "decreasing":
            insights.append("Agent responses became shorter over time, possibly indicating resolution or narrowing focus.")
        
        # Insight about repeated queries
        if len(patterns["repeated_queries"]) > 0:
            insights.append("Some queries were repeated, which might indicate unclear responses or user persistence.")
        
        return {
            "insights": insights,
            "metrics": {
                "avg_user_input_length": avg_user_input_length,
                "avg_agent_response_length": avg_agent_response_length,
                "question_frequency": patterns["question_frequency"],
                "topic_shifts": len(patterns["topic_shifts"]),
                "repeated_queries": len(patterns["repeated_queries"])
            },
            "patterns": patterns,
            "summary": summary
        }
    
    def analyze_conversation(self, conversation_id: Optional[str] = None) -> Dict:
        """
        Perform a complete analysis of a conversation
        
        Args:
            conversation_id (str, optional): ID of the conversation to analyze
                                           If None, analyzes the current conversation
            
        Returns:
            dict: Complete conversation analysis
        """
        # Get the conversation to analyze
        if self.conversation_memory is None:
            return {"error": "No conversation memory system provided"}
        
        # Check if we have a cached analysis
        cache_key = conversation_id or "current"
        if cache_key in self.analysis_cache:
            # Check if the analysis is recent (less than 1 hour old)
            if time.time() - self.analysis_cache[cache_key].get("timestamp", 0) < 3600:
                return self.analysis_cache[cache_key]
        
        # Get conversation history
        if hasattr(self.conversation_memory, "get_conversation"):
            conversation = self.conversation_memory.get_conversation(conversation_id)
        elif hasattr(self.conversation_memory, "get_history"):
            conversation = self.conversation_memory.get_history(conversation_id)
        elif hasattr(self.conversation_memory, "get_turns"):
            conversation = self.conversation_memory.get_turns()
        elif hasattr(self.conversation_memory, "turns"):
            conversation = self.conversation_memory.turns
        else:
            return {"error": "Cannot retrieve conversation history from the provided memory system"}
        
        # Perform analysis
        topics = self.extract_key_topics(conversation)
        patterns = self.identify_patterns(conversation)
        summary = self.generate_conversation_summary(conversation)
        insights = self.provide_conversation_insights(conversation)
        
        # Create complete analysis
        analysis = {
            "conversation_id": conversation_id or "current",
            "timestamp": time.time(),
            "topics": topics,
            "patterns": patterns,
            "summary": summary,
            "insights": insights
        }
        
        # Cache the analysis
        self.analysis_cache[cache_key] = analysis
        self._save_analysis()
        
        return analysis
    
    def get_topic_evolution(self, conversation: List[Dict]) -> List[Dict]:
        """
        Track how topics evolve throughout a conversation
        
        Args:
            conversation (list): List of conversation turns
            
        Returns:
            list: Topic evolution over time
        """
        if not conversation:
            return []
        
        # Divide conversation into segments
        segment_size = max(1, len(conversation) // 5)  # 5 segments or fewer
        segments = []
        
        for i in range(0, len(conversation), segment_size):
            segment = conversation[i:i+segment_size]
            if segment:
                segment_topics = self.extract_key_topics(segment, top_n=3)
                segments.append({
                    "segment_index": i // segment_size,
                    "start_turn": i,
                    "end_turn": min(i + segment_size - 1, len(conversation) - 1),
                    "topics": segment_topics
                })
        
        return segments
    
    def find_similar_conversations(self, current_conversation: List[Dict], 
                                  past_conversations: List[List[Dict]], 
                                  top_n: int = 3) -> List[Dict]:
        """
        Find past conversations similar to the current one
        
        Args:
            current_conversation (list): Current conversation turns
            past_conversations (list): List of past conversations
            top_n (int): Number of similar conversations to return
            
        Returns:
            list: Similar conversations with similarity scores
        """
        if not current_conversation or not past_conversations:
            return []
        
        # Extract topics from current conversation
        current_topics = set()
        for topic in self.extract_key_topics(current_conversation):
            current_topics.add(topic["topic"])
        
        # Calculate similarity with each past conversation
        similarities = []
        
        for i, past_conv in enumerate(past_conversations):
            past_topics = set()
            for topic in self.extract_key_topics(past_conv):
                past_topics.add(topic["topic"])
            
            # Calculate Jaccard similarity
            if not current_topics or not past_topics:
                similarity = 0
            else:
                intersection = len(current_topics.intersection(past_topics))
                union = len(current_topics.union(past_topics))
                similarity = intersection / union if union > 0 else 0
            
            similarities.append({
                "conversation_index": i,
                "similarity": similarity,
                "common_topics": list(current_topics.intersection(past_topics))
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_n]


# Mock conversation memory for testing
class MockConversationMemory:
    """A simple mock conversation memory for testing"""
    
    def __init__(self):
        """Initialize with some test conversations"""
        self.turns = [
            {
                "turn_id": 0,
                "timestamp": time.time() - 3600,
                "user_input": "Hello, I'm interested in learning about artificial intelligence.",
                "agent_response": "Hi there! I'd be happy to help you learn about AI. Is there a specific aspect of artificial intelligence you're interested in?"
            },
            {
                "turn_id": 1,
                "timestamp": time.time() - 3500,
                "user_input": "I want to know more about machine learning algorithms.",
                "agent_response": "Machine learning algorithms are methods that allow computers to learn patterns from data and make predictions. The main types include supervised learning, unsupervised learning, and reinforcement learning. Would you like me to explain any of these in more detail?"
            },
            {
                "turn_id": 2,
                "timestamp": time.time() - 3400,
                "user_input": "Yes, can you tell me about supervised learning?",
                "agent_response": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. It's called 'supervised' because the process of learning is guided by the labeled examples. The algorithm makes predictions based on the input data and is corrected when its predictions are wrong. Over time, it improves its accuracy. Common supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks."
            },
            {
                "turn_id": 3,
                "timestamp": time.time() - 3300,
                "user_input": "What's the difference between supervised and unsupervised learning?",
                "agent_response": "The main difference is that supervised learning uses labeled data, while unsupervised learning works with unlabeled data. In supervised learning, the algorithm learns to map inputs to known outputs. In unsupervised learning, the algorithm tries to find patterns or structure in the data without specific guidance. Unsupervised learning is often used for clustering, dimensionality reduction, and anomaly detection."
            },
            {
                "turn_id": 4,
                "timestamp": time.time() - 3200,
                "user_input": "Can you give me an example of a real-world application of machine learning?",
                "agent_response": "There are many real-world applications of machine learning! Some examples include: 1) Recommendation systems on platforms like Netflix or Amazon, 2) Email spam filters, 3) Voice assistants like Siri or Alexa, 4) Medical diagnosis from images, 5) Fraud detection in banking, and 6) Self-driving cars. These applications use different types of machine learning algorithms depending on their specific needs."
            }
        ]
    
    def get_turns(self):
        """Get all conversation turns"""
        return self.turns


# Example usage
if __name__ == "__main__":
    print("Conversation Memory Analyzer - Example Usage")
    print("=" * 50)
    
    # Create a mock conversation memory
    memory = MockConversationMemory()
    
    # Create the analyzer
    analyzer = ConversationMemoryAnalyzer(memory, "example_analysis.json")
    
    # Get the conversation
    conversation = memory.get_turns()
    
    # Extract key topics
    print("\nKey Topics:")
    topics = analyzer.extract_key_topics(conversation)
    for topic in topics:
        print(f"- {topic['topic']} (frequency: {topic['frequency']})")
        if topic['context']:
            print(f"  Context: \"{topic['context'][0]}\"")
    
    # Identify patterns
    print("\nConversation Patterns:")
    patterns = analyzer.identify_patterns(conversation)
    print(f"- Question frequency: {patterns['question_frequency']:.2f}")
    print("- Common question types:")
    for q_type in patterns['common_question_types']:
        print(f"  * {q_type['type']}: {q_type['count']}")
    print(f"- Topic shifts: {len(patterns['topic_shifts'])}")
    print(f"- Response length trend: {patterns['response_length_trend']}")
    
    # Generate summary
    print("\nConversation Summary:")
    summary = analyzer.generate_conversation_summary(conversation)
    print(summary['summary'])
    print(f"Duration: {analyzer._format_duration(summary['duration'])}")
    print(f"Turns: {summary['turns']}")
    print(f"Topics: {', '.join(summary['topics'])}")
    
    # Provide insights
    print("\nConversation Insights:")
    insights = analyzer.provide_conversation_insights(conversation)
    for insight in insights['insights']:
        print(f"- {insight}")
    
    # Get topic evolution
    print("\nTopic Evolution:")
    evolution = analyzer.get_topic_evolution(conversation)
    for segment in evolution:
        print(f"Segment {segment['segment_index']} (turns {segment['start_turn']}-{segment['end_turn']}):")
        for topic in segment['topics']:
            print(f"- {topic['topic']}")
    
    # Complete analysis
    print("\nComplete Analysis:")
    analysis = analyzer.analyze_conversation()
    print(f"Analysis timestamp: {time.ctime(analysis['timestamp'])}")
    print(f"Summary: {analysis['summary']['summary']}")
    print(f"Top topic: {analysis['topics'][0]['topic'] if analysis['topics'] else 'None'}")
    print(f"Key insight: {analysis['insights']['insights'][0] if analysis['insights']['insights'] else 'None'}")
    
    print("\nAnalysis complete!")
