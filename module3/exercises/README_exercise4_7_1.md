# Exercise 4.7.1: Chatbot Agent Validation

## Overview

This exercise implements validation patterns specific to chatbot agents, focusing on ensuring consistent personality, appropriate responses, and logical conversation flow. The validation system helps maintain high-quality chatbot interactions by enforcing domain-specific rules for conversational agents.

## Key Features

1. **Personality Consistency Validation**: Ensures chatbot responses match the defined personality traits (formal, technical, empathetic, etc.)
2. **Response Appropriateness Validation**: Validates that responses are appropriate for the current conversation state (greetings, complaints, technical support, etc.)
3. **Emotion Levels Validation**: Ensures emotion levels in responses are appropriate for the context
4. **Conversation Flow Validation**: Enforces logical state transitions in conversations
5. **Content Category Validation**: Validates that response content matches the agent's personality and conversation state

## Components

### Personality Models

- `PersonalityTrait`: Defines a single personality trait with a value between 0 and 1
- `PersonalityProfile`: Collection of personality traits that define a chatbot's character

### Conversation State Models

- `ConversationState`: Enum of possible conversation states (greeting, handling_query, technical_support, etc.)
- `ContentCategory`: Enum of content categories (general, technical, emotional, etc.)
- `EmotionLevel`: Enum of emotion levels (none, low, medium, high)

### Message Models

- `MessageMetadata`: Metadata associated with a chatbot message
- `ChatbotResponse`: Model for a chatbot response with validation for personality consistency
- `UserQuery`: Model for a user query to a chatbot

### Conversation Context

- `ConversationContext`: Tracks the state of a conversation, including history and transitions

### Chatbot Implementation

- `CustomerServiceChatbot`: Implementation of a customer service chatbot with specialized validation

## Usage Example

```python
from exercise4_7_1_chatbot_validator import (
    PersonalityProfile, CustomerServiceChatbot, UserQuery
)

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
```

## Personality Consistency Example

```python
from exercise4_7_1_chatbot_validator import (
    ConversationState, ContentCategory, PersonalityProfile, ChatbotResponse
)

# Create a formal personality profile
profile = PersonalityProfile()
profile.add_trait("formal", 0.9)
profile.add_trait("empathetic", 0.7)

# Valid formal response
try:
    response = ChatbotResponse(
        message="Good morning. I understand your concern regarding the delayed shipment. I would be happy to assist you with tracking your package.",
        conversation_state=ConversationState.HANDLING_COMPLAINT,
        personality_profile=profile,
        content_categories=[ContentCategory.EMOTIONAL, ContentCategory.INFORMATIONAL],
        emotion_levels={"empathy": 0.7, "neutral": 0.3}
    )
    print("Valid response:", response.message)
except ValueError as e:
    print("Validation error:", e)

# Invalid formal response (contains informal language)
try:
    response = ChatbotResponse(
        message="Hey there! Yeah, I totally get your frustration with the late delivery. We're gonna fix this ASAP!",
        conversation_state=ConversationState.HANDLING_COMPLAINT,
        personality_profile=profile,
        content_categories=[ContentCategory.EMOTIONAL],
        emotion_levels={"empathy": 0.7, "enthusiasm": 0.8}
    )
    print("Valid response:", response.message)
except ValueError as e:
    print("Validation error:", e)
    # Output: Validation error: Message tone doesn't match formal personality trait
```

## Conversation Flow Example

```python
from exercise4_7_1_chatbot_validator import (
    ConversationState, ConversationContext, PersonalityProfile, ChatbotResponse,
    ContentCategory
)

# Create a conversation context
context = ConversationContext(
    conversation_id="test-conversation",
    current_state=ConversationState.GREETING
)

# Create a personality profile
profile = PersonalityProfile()
profile.add_trait("formal", 0.7)

# Valid state transition: GREETING -> UNDERSTANDING_INTENT
response1 = ChatbotResponse(
    message="Hello! How can I assist you today?",
    conversation_state=ConversationState.UNDERSTANDING_INTENT,
    personality_profile=profile,
    content_categories=[ContentCategory.GENERAL]
)
context.add_chatbot_response(response1)

# Valid state transition: UNDERSTANDING_INTENT -> HANDLING_QUERY
response2 = ChatbotResponse(
    message="I'll help you with that query.",
    conversation_state=ConversationState.HANDLING_QUERY,
    personality_profile=profile,
    content_categories=[ContentCategory.GENERAL]
)
context.add_chatbot_response(response2)

# Invalid state transition: HANDLING_QUERY -> GREETING
try:
    response3 = ChatbotResponse(
        message="Hello again! How can I help you?",
        conversation_state=ConversationState.GREETING,
        personality_profile=profile,
        content_categories=[ContentCategory.GENERAL]
    )
    context.add_chatbot_response(response3)
except ValueError as e:
    print("Validation error:", e)
    # Output: Validation error: Invalid state transition from handling_query to greeting
```

## Running the Demo

To run the interactive demo:

```bash
python demo_exercise4_7_1_chatbot_validator.py
```

The demo showcases:
1. Personality Consistency Validation
2. Response Appropriateness Validation
3. Emotion Levels Validation
4. Conversation Flow Validation
5. Complete Customer Service Chatbot Interaction

## Running the Tests

To run the tests:

```bash
python -m unittest test_exercise4_7_1_chatbot_validator.py
```

## Integration with Agent Systems

This validator can be integrated with agent systems to:

1. **Ensure Consistent Personality**: Maintain a consistent chatbot personality across interactions
2. **Guide Conversation Flow**: Enforce logical conversation state transitions
3. **Validate Response Appropriateness**: Ensure responses match the conversation context
4. **Monitor Emotion Levels**: Keep emotion levels appropriate for the conversation context
5. **Improve User Experience**: Create more natural and appropriate chatbot interactions

## Real-World Applications

- **Customer Service Chatbots**: Ensure consistent and appropriate responses to customer inquiries
- **Technical Support Agents**: Validate technical content and appropriate emotion levels
- **Sales Assistants**: Maintain appropriate tone and personality for sales conversations
- **Healthcare Chatbots**: Ensure empathetic responses to health concerns
- **Educational Assistants**: Maintain appropriate teaching tone and personality
