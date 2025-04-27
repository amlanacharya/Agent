"""
Tests for Exercise 4.7.1: Chatbot Agent Validation
------------------------------------------------
This module contains tests for the chatbot agent validation patterns.
"""

import unittest
from datetime import datetime, timedelta
from exercise4_7_1_chatbot_validator import (
    ConversationState, EmotionLevel, ContentCategory, PersonalityTrait,
    PersonalityProfile, MessageMetadata, ChatbotResponse, UserQuery,
    ConversationContext, CustomerServiceChatbot
)


class TestPersonalityProfile(unittest.TestCase):
    """Tests for the PersonalityProfile class."""
    
    def test_add_trait(self):
        """Test adding traits to a personality profile."""
        profile = PersonalityProfile()
        
        # Add valid traits
        profile.add_trait("formal", 0.8)
        profile.add_trait("empathetic", 0.9)
        
        self.assertEqual(profile.get_trait("formal"), 0.8)
        self.assertEqual(profile.get_trait("empathetic"), 0.9)
        
        # Add trait with value > 1.0 (should be capped at 1.0)
        profile.add_trait("technical", 1.5)
        self.assertEqual(profile.get_trait("technical"), 1.0)
        
        # Add trait with value < 0.0 (should be capped at 0.0)
        profile.add_trait("casual", -0.5)
        self.assertEqual(profile.get_trait("casual"), 0.0)
    
    def test_is_trait_dominant(self):
        """Test checking if a trait is dominant."""
        profile = PersonalityProfile()
        profile.add_trait("formal", 0.8)
        profile.add_trait("casual", 0.3)
        
        self.assertTrue(profile.is_trait_dominant("formal"))
        self.assertFalse(profile.is_trait_dominant("casual"))
        self.assertFalse(profile.is_trait_dominant("nonexistent"))
        
        # Test with custom threshold
        self.assertTrue(profile.is_trait_dominant("formal", threshold=0.7))
        self.assertFalse(profile.is_trait_dominant("formal", threshold=0.9))


class TestChatbotResponse(unittest.TestCase):
    """Tests for the ChatbotResponse class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formal_profile = PersonalityProfile()
        self.formal_profile.add_trait("formal", 0.8)
        self.formal_profile.add_trait("empathetic", 0.5)
        
        self.empathetic_profile = PersonalityProfile()
        self.empathetic_profile.add_trait("formal", 0.3)
        self.empathetic_profile.add_trait("empathetic", 0.9)
        
        self.technical_profile = PersonalityProfile()
        self.technical_profile.add_trait("technical", 0.8)
    
    def test_validate_personality_consistency_formal(self):
        """Test validation of formal personality consistency."""
        # Valid formal response
        valid_response = ChatbotResponse(
            message="Hello, I understand your concern. How may I assist you today?",
            conversation_state=ConversationState.GREETING,
            personality_profile=self.formal_profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should not raise an exception
        valid_response.validate_personality_consistency()
        
        # Invalid formal response (contains informal language)
        invalid_response = ChatbotResponse(
            message="Hey there! What's up? Gonna help you out today!",
            conversation_state=ConversationState.GREETING,
            personality_profile=self.formal_profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_response.validate_personality_consistency()
    
    def test_validate_personality_consistency_technical(self):
        """Test validation of technical personality consistency."""
        # Valid technical response
        valid_response = ChatbotResponse(
            message="The system requires a restart of the network services to resolve the connectivity issue.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=self.technical_profile,
            content_categories=[ContentCategory.TECHNICAL]
        )
        
        # This should not raise an exception
        valid_response.validate_personality_consistency()
        
        # Invalid technical response (missing technical content category)
        invalid_response = ChatbotResponse(
            message="The system needs to be restarted.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=self.technical_profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_response.validate_personality_consistency()
    
    def test_validate_personality_consistency_empathetic(self):
        """Test validation of empathetic personality consistency."""
        # Valid empathetic response for complaint handling
        valid_response = ChatbotResponse(
            message="I understand your frustration with the delayed shipment. I'm sorry for the inconvenience.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.EMOTIONAL],
            emotion_levels={"empathy": 0.8}
        )
        
        # This should not raise an exception
        valid_response.validate_personality_consistency()
        
        # Invalid empathetic response for complaint handling (lacks empathy)
        invalid_response = ChatbotResponse(
            message="Your shipment is delayed. It will arrive next week.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.INFORMATIONAL],
            emotion_levels={"empathy": 0.2}
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_response.validate_personality_consistency()
    
    def test_validate_response_appropriateness(self):
        """Test validation of response appropriateness for conversation state."""
        # Valid greeting
        valid_greeting = ChatbotResponse(
            message="Hello! How can I assist you today?",
            conversation_state=ConversationState.GREETING,
            personality_profile=self.formal_profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should not raise an exception
        valid_greeting.validate_response_appropriateness()
        
        # Invalid greeting (missing greeting phrase)
        invalid_greeting = ChatbotResponse(
            message="Let me help you with your issue.",
            conversation_state=ConversationState.GREETING,
            personality_profile=self.formal_profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_greeting.validate_response_appropriateness()
        
        # Valid complaint handling
        valid_complaint = ChatbotResponse(
            message="I'm sorry to hear about your issue. Let me help resolve it.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.EMOTIONAL],
            emotion_levels={"empathy": 0.8}
        )
        
        # This should not raise an exception
        valid_complaint.validate_response_appropriateness()
        
        # Invalid complaint handling (missing apology)
        invalid_complaint = ChatbotResponse(
            message="Let me help resolve your issue.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.GENERAL],
            emotion_levels={"empathy": 0.8}
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_complaint.validate_response_appropriateness()
    
    def test_validate_emotion_levels(self):
        """Test validation of emotion levels for conversation state."""
        # Valid technical support (appropriate emotion levels)
        valid_technical = ChatbotResponse(
            message="Let me help you troubleshoot this issue.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=self.technical_profile,
            content_categories=[ContentCategory.TECHNICAL],
            emotion_levels={"neutral": 0.8, "excitement": 0.3}
        )
        
        # This should not raise an exception
        valid_technical.validate_emotion_levels()
        
        # Invalid technical support (excessive excitement)
        invalid_technical = ChatbotResponse(
            message="I'm SUPER EXCITED to help you troubleshoot this issue!!!",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=self.technical_profile,
            content_categories=[ContentCategory.TECHNICAL],
            emotion_levels={"excitement": 0.9}
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_technical.validate_emotion_levels()
        
        # Valid complaint handling (appropriate empathy)
        valid_complaint = ChatbotResponse(
            message="I'm sorry to hear about your issue. I understand your frustration.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.EMOTIONAL],
            emotion_levels={"empathy": 0.8}
        )
        
        # This should not raise an exception
        valid_complaint.validate_emotion_levels()
        
        # Invalid complaint handling (insufficient empathy)
        invalid_complaint = ChatbotResponse(
            message="I'll look into your issue.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=self.empathetic_profile,
            content_categories=[ContentCategory.GENERAL],
            emotion_levels={"empathy": 0.2}
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_complaint.validate_emotion_levels()


class TestConversationContext(unittest.TestCase):
    """Tests for the ConversationContext class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile = PersonalityProfile()
        self.profile.add_trait("formal", 0.7)
        self.profile.add_trait("empathetic", 0.8)
        
        self.context = ConversationContext(
            conversation_id="test-conversation",
            current_state=ConversationState.GREETING
        )
    
    def test_validate_conversation_flow(self):
        """Test validation of conversation flow."""
        # Add a valid transition from GREETING to UNDERSTANDING_INTENT
        response1 = ChatbotResponse(
            message="Hello! How can I assist you today?",
            conversation_state=ConversationState.UNDERSTANDING_INTENT,
            personality_profile=self.profile,
            content_categories=[ContentCategory.GENERAL]
        )
        self.context.add_chatbot_response(response1)
        
        # Add a valid transition from UNDERSTANDING_INTENT to HANDLING_QUERY
        response2 = ChatbotResponse(
            message="Let me help you with that query.",
            conversation_state=ConversationState.HANDLING_QUERY,
            personality_profile=self.profile,
            content_categories=[ContentCategory.GENERAL]
        )
        self.context.add_chatbot_response(response2)
        
        # Add a valid transition from HANDLING_QUERY to PROVIDING_INFORMATION
        response3 = ChatbotResponse(
            message="Here's the information you requested.",
            conversation_state=ConversationState.PROVIDING_INFORMATION,
            personality_profile=self.profile,
            content_categories=[ContentCategory.INFORMATIONAL]
        )
        self.context.add_chatbot_response(response3)
        
        # This should not raise an exception
        self.context.validate_conversation_flow()
        
        # Try an invalid transition from PROVIDING_INFORMATION to UNDERSTANDING_INTENT
        response4 = ChatbotResponse(
            message="What else would you like to know?",
            conversation_state=ConversationState.UNDERSTANDING_INTENT,
            personality_profile=self.profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.context.add_chatbot_response(response4)
            self.context.validate_conversation_flow()
    
    def test_add_user_query(self):
        """Test adding a user query to the conversation context."""
        query = UserQuery(
            text="I need help with my order.",
            detected_intent="order_help",
            detected_entities={"order": True}
        )
        
        # Add the query
        self.context.add_user_query(query)
        
        # Check that the query was added
        self.assertEqual(len(self.context.user_queries), 1)
        self.assertEqual(self.context.user_queries[0].text, "I need help with my order.")
        
        # Check that last_updated was updated
        self.assertGreaterEqual(self.context.last_updated, self.context.session_start)
    
    def test_add_chatbot_response(self):
        """Test adding a chatbot response to the conversation context."""
        response = ChatbotResponse(
            message="Hello! How can I assist you today?",
            conversation_state=ConversationState.UNDERSTANDING_INTENT,
            personality_profile=self.profile,
            content_categories=[ContentCategory.GENERAL]
        )
        
        # Add the response
        self.context.add_chatbot_response(response)
        
        # Check that the response was added
        self.assertEqual(len(self.context.chatbot_responses), 1)
        self.assertEqual(self.context.chatbot_responses[0].message, "Hello! How can I assist you today?")
        
        # Check that the state was updated
        self.assertEqual(self.context.current_state, ConversationState.UNDERSTANDING_INTENT)
        self.assertEqual(self.context.previous_states[0], ConversationState.GREETING)
        
        # Check that last_updated was updated
        self.assertGreaterEqual(self.context.last_updated, self.context.session_start)


class TestCustomerServiceChatbot(unittest.TestCase):
    """Tests for the CustomerServiceChatbot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile = PersonalityProfile()
        self.profile.add_trait("formal", 0.7)
        self.profile.add_trait("empathetic", 0.8)
    
    def test_validate_supported_capabilities(self):
        """Test validation of supported capabilities."""
        # Valid chatbot with all required intents
        valid_chatbot = CustomerServiceChatbot(
            name="ValidBot",
            personality_profile=self.profile,
            supported_intents={"greeting", "complaint", "information", "technical_support", "farewell"},
            supported_languages={"en"}
        )
        
        # This should not raise an exception
        valid_chatbot.validate_supported_capabilities()
        
        # Invalid chatbot missing required intents
        invalid_chatbot = CustomerServiceChatbot(
            name="InvalidBot",
            personality_profile=self.profile,
            supported_intents={"greeting", "information"},
            supported_languages={"en"}
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_chatbot.validate_supported_capabilities()
        
        # Invalid chatbot with no supported languages
        invalid_chatbot2 = CustomerServiceChatbot(
            name="InvalidBot2",
            personality_profile=self.profile,
            supported_intents={"greeting", "complaint", "information", "technical_support", "farewell"},
            supported_languages=set()
        )
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            invalid_chatbot2.validate_supported_capabilities()
    
    def test_generate_response(self):
        """Test generating a response to a user query."""
        chatbot = CustomerServiceChatbot(
            name="TestBot",
            personality_profile=self.profile,
            supported_intents={"greeting", "complaint", "information", "technical_support", "farewell"},
            supported_languages={"en"}
        )
        
        # Test greeting query
        greeting_query = UserQuery(
            text="Hello, I need some help.",
            detected_intent="greeting"
        )
        
        greeting_response = chatbot.generate_response(greeting_query)
        self.assertEqual(greeting_response.conversation_state, ConversationState.GREETING)
        self.assertIn("Hello", greeting_response.message)
        
        # Test complaint query
        complaint_query = UserQuery(
            text="I have a problem with my order.",
            detected_intent="complaint",
            detected_entities={"order": True}
        )
        
        complaint_response = chatbot.generate_response(complaint_query)
        self.assertEqual(complaint_response.conversation_state, ConversationState.HANDLING_COMPLAINT)
        self.assertIn("sorry", complaint_response.message.lower())
        
        # Test technical support query
        tech_query = UserQuery(
            text="Can you help me fix this issue?",
            detected_intent="technical_support"
        )
        
        tech_response = chatbot.generate_response(tech_query)
        self.assertEqual(tech_response.conversation_state, ConversationState.TECHNICAL_SUPPORT)
        self.assertIn("troubleshoot", tech_response.message.lower())
        
        # Test farewell query
        farewell_query = UserQuery(
            text="Thank you, goodbye!",
            detected_intent="farewell"
        )
        
        farewell_response = chatbot.generate_response(farewell_query)
        self.assertEqual(farewell_response.conversation_state, ConversationState.ENDING)
        self.assertIn("thank you", farewell_response.message.lower())


if __name__ == "__main__":
    unittest.main()
