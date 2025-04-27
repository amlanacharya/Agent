"""
Exercise 4.7.1: Chatbot Agent Validation
---------------------------------------
This module implements validation patterns specific to chatbot agents, focusing on:
1. Conversation flow validation
2. Response appropriateness validation
3. Personality consistency validation
4. Content safety validation
5. User interaction validation

These validation patterns help ensure that chatbot agents maintain consistent
personality traits while providing appropriate responses in different conversation contexts.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Literal, Set, Any
from enum import Enum
from datetime import datetime
import re


class ConversationState(str, Enum):
    """Enum for conversation states in a chatbot interaction."""
    GREETING = "greeting"
    UNDERSTANDING_INTENT = "understanding_intent"
    HANDLING_QUERY = "handling_query"
    HANDLING_COMPLAINT = "handling_complaint"
    PROVIDING_INFORMATION = "providing_information"
    TECHNICAL_SUPPORT = "technical_support"
    CLARIFYING = "clarifying"
    CONFIRMING = "confirming"
    FOLLOW_UP = "follow_up"
    ENDING = "ending"


class EmotionLevel(float, Enum):
    """Enum for emotion levels in chatbot responses."""
    NONE = 0.0
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.9


class ContentCategory(str, Enum):
    """Enum for content categories in chatbot responses."""
    GENERAL = "general"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    INFORMATIONAL = "informational"
    PROMOTIONAL = "promotional"
    SUPPORTIVE = "supportive"
    DIRECTIVE = "directive"


class PersonalityTrait(BaseModel):
    """Model for a personality trait with a value between 0 and 1."""
    name: str
    value: float = Field(ge=0.0, le=1.0)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate that the trait name is valid."""
        if not v.strip():
            raise ValueError("Trait name cannot be empty")
        return v.lower()


class PersonalityProfile(BaseModel):
    """Model for a chatbot's personality profile."""
    traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of personality traits with values between 0 and 1"
    )
    
    def add_trait(self, name: str, value: float):
        """Add a trait to the personality profile."""
        self.traits[name.lower()] = max(0.0, min(1.0, value))
    
    def get_trait(self, name: str) -> float:
        """Get a trait value from the personality profile."""
        return self.traits.get(name.lower(), 0.0)
    
    def is_trait_dominant(self, name: str, threshold: float = 0.7) -> bool:
        """Check if a trait is dominant in the personality profile."""
        return self.get_trait(name) >= threshold


class MessageMetadata(BaseModel):
    """Model for metadata associated with a chatbot message."""
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    channel: Optional[str] = None
    language: str = "en"
    sentiment_score: Optional[float] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)


class ChatbotResponse(BaseModel):
    """Model for a chatbot response with validation for personality consistency."""
    message: str
    conversation_state: ConversationState
    personality_profile: PersonalityProfile
    content_categories: List[ContentCategory] = Field(default_factory=list)
    emotion_levels: Dict[str, float] = Field(default_factory=dict)
    metadata: Optional[MessageMetadata] = None
    
    @model_validator(mode='after')
    def validate_personality_consistency(self):
        """Validate that the message reflects the personality traits."""
        # Check for formal personality trait
        if self.personality_profile.is_trait_dominant("formal"):
            informal_phrases = ["hey", "yeah", "cool", "awesome", "btw", "gonna", "wanna"]
            if any(re.search(r'\b' + phrase + r'\b', self.message.lower()) for phrase in informal_phrases):
                raise ValueError("Message tone doesn't match formal personality trait")
        
        # Check for technical personality trait
        if self.personality_profile.is_trait_dominant("technical"):
            if ContentCategory.TECHNICAL not in self.content_categories:
                raise ValueError("Message content doesn't match technical personality trait")
        
        # Check for empathetic personality trait
        if self.personality_profile.is_trait_dominant("empathetic"):
            empathy_phrases = ["understand", "appreciate", "feel", "concern", "sorry"]
            if not any(phrase in self.message.lower() for phrase in empathy_phrases):
                if self.conversation_state in [ConversationState.HANDLING_COMPLAINT, ConversationState.TECHNICAL_SUPPORT]:
                    raise ValueError("Message lacks empathy for complaint or support context")
        
        return self
    
    @model_validator(mode='after')
    def validate_response_appropriateness(self):
        """Validate that the response is appropriate for the conversation state."""
        # Check for greeting state
        if self.conversation_state == ConversationState.GREETING:
            greeting_phrases = ["hello", "hi", "welcome", "greetings", "good morning", "good afternoon", "good evening"]
            if not any(phrase in self.message.lower() for phrase in greeting_phrases):
                raise ValueError("Message doesn't contain appropriate greeting")
        
        # Check for ending state
        if self.conversation_state == ConversationState.ENDING:
            ending_phrases = ["goodbye", "bye", "farewell", "thank you", "thanks", "have a good day"]
            if not any(phrase in self.message.lower() for phrase in ending_phrases):
                raise ValueError("Message doesn't contain appropriate ending")
        
        # Check for handling complaint state
        if self.conversation_state == ConversationState.HANDLING_COMPLAINT:
            if "sorry" not in self.message.lower() and "apologize" not in self.message.lower():
                raise ValueError("Complaint handling should include an apology")
        
        return self
    
    @model_validator(mode='after')
    def validate_emotion_levels(self):
        """Validate that emotion levels are appropriate for the conversation state."""
        # Check for excessive emotion in technical support
        if self.conversation_state == ConversationState.TECHNICAL_SUPPORT:
            for emotion, level in self.emotion_levels.items():
                if emotion in ["excitement", "enthusiasm"] and level > 0.7:
                    raise ValueError(f"Excessive {emotion} in technical support context")
        
        # Check for insufficient empathy in complaint handling
        if self.conversation_state == ConversationState.HANDLING_COMPLAINT:
            empathy_level = self.emotion_levels.get("empathy", 0.0)
            if empathy_level < 0.5:
                raise ValueError("Insufficient empathy in complaint handling context")
        
        return self
    
    @model_validator(mode='after')
    def validate_message_length(self):
        """Validate that the message length is appropriate for the conversation state."""
        message_words = len(self.message.split())
        
        # Check for overly verbose greetings
        if self.conversation_state == ConversationState.GREETING and message_words > 30:
            raise ValueError("Greeting message is too verbose")
        
        # Check for overly terse technical support
        if self.conversation_state == ConversationState.TECHNICAL_SUPPORT and message_words < 15:
            raise ValueError("Technical support message is too terse")
        
        return self


class UserQuery(BaseModel):
    """Model for a user query to a chatbot."""
    text: str
    detected_intent: Optional[str] = None
    detected_entities: Dict[str, Any] = Field(default_factory=dict)
    sentiment: Optional[float] = None
    metadata: Optional[MessageMetadata] = None


class ConversationContext(BaseModel):
    """Model for the context of a conversation with a chatbot."""
    conversation_id: str
    current_state: ConversationState = ConversationState.GREETING
    previous_states: List[ConversationState] = Field(default_factory=list)
    user_queries: List[UserQuery] = Field(default_factory=list)
    chatbot_responses: List[ChatbotResponse] = Field(default_factory=list)
    session_start: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_conversation_flow(self):
        """Validate that the conversation flow follows logical transitions."""
        # Skip validation if no previous states
        if not self.previous_states:
            return self
        
        # Define valid state transitions
        valid_transitions = {
            ConversationState.GREETING: {
                ConversationState.UNDERSTANDING_INTENT,
                ConversationState.HANDLING_QUERY,
                ConversationState.ENDING
            },
            ConversationState.UNDERSTANDING_INTENT: {
                ConversationState.HANDLING_QUERY,
                ConversationState.HANDLING_COMPLAINT,
                ConversationState.PROVIDING_INFORMATION,
                ConversationState.TECHNICAL_SUPPORT,
                ConversationState.CLARIFYING
            },
            ConversationState.HANDLING_QUERY: {
                ConversationState.PROVIDING_INFORMATION,
                ConversationState.CLARIFYING,
                ConversationState.CONFIRMING,
                ConversationState.FOLLOW_UP,
                ConversationState.ENDING
            },
            ConversationState.HANDLING_COMPLAINT: {
                ConversationState.PROVIDING_INFORMATION,
                ConversationState.TECHNICAL_SUPPORT,
                ConversationState.CONFIRMING,
                ConversationState.FOLLOW_UP,
                ConversationState.ENDING
            },
            ConversationState.PROVIDING_INFORMATION: {
                ConversationState.CONFIRMING,
                ConversationState.FOLLOW_UP,
                ConversationState.ENDING
            },
            ConversationState.TECHNICAL_SUPPORT: {
                ConversationState.CLARIFYING,
                ConversationState.CONFIRMING,
                ConversationState.FOLLOW_UP,
                ConversationState.ENDING
            },
            ConversationState.CLARIFYING: {
                ConversationState.HANDLING_QUERY,
                ConversationState.HANDLING_COMPLAINT,
                ConversationState.PROVIDING_INFORMATION,
                ConversationState.TECHNICAL_SUPPORT
            },
            ConversationState.CONFIRMING: {
                ConversationState.FOLLOW_UP,
                ConversationState.ENDING
            },
            ConversationState.FOLLOW_UP: {
                ConversationState.HANDLING_QUERY,
                ConversationState.PROVIDING_INFORMATION,
                ConversationState.ENDING
            },
            ConversationState.ENDING: set()  # No transitions from ending
        }
        
        # Check if the current state is a valid transition from the previous state
        previous_state = self.previous_states[-1]
        if self.current_state not in valid_transitions[previous_state]:
            raise ValueError(f"Invalid state transition from {previous_state} to {self.current_state}")
        
        return self
    
    def add_user_query(self, query: UserQuery):
        """Add a user query to the conversation context."""
        self.user_queries.append(query)
        self.last_updated = datetime.now()
    
    def add_chatbot_response(self, response: ChatbotResponse):
        """Add a chatbot response to the conversation context."""
        # Update previous states before changing current state
        self.previous_states.append(self.current_state)
        self.current_state = response.conversation_state
        self.chatbot_responses.append(response)
        self.last_updated = datetime.now()


class CustomerServiceChatbot(BaseModel):
    """Model for a customer service chatbot with specialized validation."""
    name: str
    personality_profile: PersonalityProfile
    supported_intents: Set[str] = Field(default_factory=set)
    supported_languages: Set[str] = Field(default_factory=set)
    conversation_context: Optional[ConversationContext] = None
    
    @model_validator(mode='after')
    def validate_supported_capabilities(self):
        """Validate that the chatbot has the necessary capabilities."""
        # Check for minimum supported intents for customer service
        required_intents = {"greeting", "complaint", "information", "technical_support", "farewell"}
        if not required_intents.issubset(self.supported_intents):
            missing = required_intents - self.supported_intents
            raise ValueError(f"Customer service chatbot missing required intents: {missing}")
        
        # Check for minimum supported languages
        if not self.supported_languages:
            raise ValueError("Chatbot must support at least one language")
        
        return self
    
    def generate_response(self, user_query: UserQuery) -> ChatbotResponse:
        """
        Generate a response to a user query.
        
        This is a simplified implementation that would be replaced with actual
        response generation logic in a real system.
        """
        # Create a new conversation context if none exists
        if not self.conversation_context:
            self.conversation_context = ConversationContext(
                conversation_id="conv-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                current_state=ConversationState.GREETING
            )
        
        # Add the user query to the conversation context
        self.conversation_context.add_user_query(user_query)
        
        # Determine the next conversation state based on the query
        # This is a simplified implementation
        next_state = ConversationState.HANDLING_QUERY
        if "hello" in user_query.text.lower() or "hi" in user_query.text.lower():
            next_state = ConversationState.GREETING
        elif "problem" in user_query.text.lower() or "issue" in user_query.text.lower():
            next_state = ConversationState.HANDLING_COMPLAINT
        elif "how" in user_query.text.lower() or "what" in user_query.text.lower():
            next_state = ConversationState.PROVIDING_INFORMATION
        elif "help" in user_query.text.lower() or "fix" in user_query.text.lower():
            next_state = ConversationState.TECHNICAL_SUPPORT
        elif "bye" in user_query.text.lower() or "goodbye" in user_query.text.lower():
            next_state = ConversationState.ENDING
        
        # Generate a simple response based on the next state
        # This would be replaced with more sophisticated logic in a real system
        response_text = "I'm here to help."
        if next_state == ConversationState.GREETING:
            response_text = "Hello! How can I assist you today?"
        elif next_state == ConversationState.HANDLING_COMPLAINT:
            response_text = "I'm sorry to hear that you're experiencing an issue. Could you please provide more details so I can help resolve it?"
        elif next_state == ConversationState.PROVIDING_INFORMATION:
            response_text = "Here's the information you requested. Is there anything else you'd like to know?"
        elif next_state == ConversationState.TECHNICAL_SUPPORT:
            response_text = "Let me help you troubleshoot this technical issue. Could you please tell me what steps you've already tried?"
        elif next_state == ConversationState.ENDING:
            response_text = "Thank you for chatting with us today. Have a great day!"
        
        # Create the response
        response = ChatbotResponse(
            message=response_text,
            conversation_state=next_state,
            personality_profile=self.personality_profile,
            content_categories=[ContentCategory.GENERAL],
            emotion_levels={"neutral": 0.7}
        )
        
        # Add the response to the conversation context
        self.conversation_context.add_chatbot_response(response)
        
        return response


# Example usage
if __name__ == "__main__":
    # Create a personality profile
    profile = PersonalityProfile()
    profile.add_trait("formal", 0.8)
    profile.add_trait("empathetic", 0.9)
    profile.add_trait("technical", 0.6)
    
    # Create a customer service chatbot
    chatbot = CustomerServiceChatbot(
        name="ServiceBot",
        personality_profile=profile,
        supported_intents={"greeting", "complaint", "information", "technical_support", "farewell"},
        supported_languages={"en"}
    )
    
    # Create a user query
    query = UserQuery(
        text="I'm having a problem with my order. It hasn't arrived yet.",
        detected_intent="complaint",
        detected_entities={"order": True},
        sentiment=-0.5
    )
    
    # Generate a response
    try:
        response = chatbot.generate_response(query)
        print(f"User: {query.text}")
        print(f"Bot: {response.message}")
        print(f"State: {response.conversation_state}")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Try another query
    query = UserQuery(
        text="Can you help me fix this technical issue?",
        detected_intent="technical_support",
        detected_entities={"technical_issue": True},
        sentiment=-0.3
    )
    
    # Generate a response
    try:
        response = chatbot.generate_response(query)
        print(f"\nUser: {query.text}")
        print(f"Bot: {response.message}")
        print(f"State: {response.conversation_state}")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Try a farewell
    query = UserQuery(
        text="Thank you for your help. Goodbye!",
        detected_intent="farewell",
        sentiment=0.5
    )
    
    # Generate a response
    try:
        response = chatbot.generate_response(query)
        print(f"\nUser: {query.text}")
        print(f"Bot: {response.message}")
        print(f"State: {response.conversation_state}")
    except ValueError as e:
        print(f"Validation error: {e}")
