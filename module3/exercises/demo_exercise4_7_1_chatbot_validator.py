"""
Demo script for Exercise 4.7.1: Chatbot Agent Validation
-----------------------------------------------------
This script demonstrates the usage of the chatbot agent validation patterns
through a simulated conversation with a customer service chatbot.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama
init()

# Import the chatbot validator module
from exercise4_7_1_chatbot_validator import (
    ConversationState, EmotionLevel, ContentCategory, PersonalityTrait,
    PersonalityProfile, MessageMetadata, ChatbotResponse, UserQuery,
    ConversationContext, CustomerServiceChatbot
)


def print_header(text: str) -> None:
    """Print a header with formatting."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text.center(80)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")


def print_subheader(text: str) -> None:
    """Print a subheader with formatting."""
    print(f"\n{Fore.YELLOW}{'-' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{text.center(80)}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * 80}{Style.RESET_ALL}")


def print_user_message(text: str) -> None:
    """Print a user message with formatting."""
    print(f"\n{Fore.GREEN}User: {text}{Style.RESET_ALL}")


def print_bot_message(text: str, state: ConversationState) -> None:
    """Print a bot message with formatting."""
    print(f"{Fore.BLUE}Bot ({state.value}): {text}{Style.RESET_ALL}")


def print_validation_error(error: str) -> None:
    """Print a validation error with formatting."""
    print(f"{Fore.RED}Validation Error: {error}{Style.RESET_ALL}")


def print_personality_profile(profile: PersonalityProfile) -> None:
    """Print a personality profile with formatting."""
    print(f"\n{Fore.MAGENTA}Personality Profile:{Style.RESET_ALL}")
    for trait, value in profile.traits.items():
        # Color code the trait value
        if value >= 0.7:
            color = Fore.GREEN
        elif value >= 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        print(f"  - {trait}: {color}{value:.1f}{Style.RESET_ALL}")


def print_conversation_context(context: ConversationContext) -> None:
    """Print a conversation context with formatting."""
    print(f"\n{Fore.MAGENTA}Conversation Context:{Style.RESET_ALL}")
    print(f"  - ID: {context.conversation_id}")
    print(f"  - Current State: {Fore.CYAN}{context.current_state.value}{Style.RESET_ALL}")
    print(f"  - Previous States: {', '.join([s.value for s in context.previous_states])}")
    print(f"  - Session Start: {context.session_start}")
    print(f"  - Last Updated: {context.last_updated}")
    print(f"  - User Queries: {len(context.user_queries)}")
    print(f"  - Bot Responses: {len(context.chatbot_responses)}")


def demo_personality_consistency() -> None:
    """Demonstrate personality consistency validation."""
    print_header("Personality Consistency Validation Demo")
    
    # Create a formal personality profile
    formal_profile = PersonalityProfile()
    formal_profile.add_trait("formal", 0.9)
    formal_profile.add_trait("empathetic", 0.7)
    formal_profile.add_trait("technical", 0.5)
    
    print_personality_profile(formal_profile)
    
    print_subheader("Valid Formal Response")
    
    # Create a valid formal response
    try:
        valid_response = ChatbotResponse(
            message="Good morning. I understand your concern regarding the delayed shipment. I would be happy to assist you with tracking your package.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=formal_profile,
            content_categories=[ContentCategory.EMOTIONAL, ContentCategory.INFORMATIONAL],
            emotion_levels={"empathy": 0.7, "neutral": 0.3}
        )
        
        print(f"Message: {valid_response.message}")
        print(f"State: {valid_response.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Formal Response (Informal Language)")
    
    # Create an invalid formal response with informal language
    try:
        invalid_response = ChatbotResponse(
            message="Hey there! Yeah, I totally get your frustration with the late delivery. We're gonna fix this ASAP!",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=formal_profile,
            content_categories=[ContentCategory.EMOTIONAL],
            emotion_levels={"empathy": 0.7, "enthusiasm": 0.8}
        )
        
        print(f"Message: {invalid_response.message}")
        print(f"State: {invalid_response.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    # Create a technical personality profile
    technical_profile = PersonalityProfile()
    technical_profile.add_trait("technical", 0.9)
    technical_profile.add_trait("formal", 0.7)
    technical_profile.add_trait("empathetic", 0.3)
    
    print_subheader("Technical Personality Profile")
    print_personality_profile(technical_profile)
    
    print_subheader("Valid Technical Response")
    
    # Create a valid technical response
    try:
        valid_technical = ChatbotResponse(
            message="The system error you're experiencing is likely due to a network configuration issue. Please verify your router settings and ensure that ports 80 and 443 are open for HTTP and HTTPS traffic.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=technical_profile,
            content_categories=[ContentCategory.TECHNICAL, ContentCategory.DIRECTIVE],
            emotion_levels={"neutral": 0.8}
        )
        
        print(f"Message: {valid_technical.message}")
        print(f"State: {valid_technical.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Technical Response (Missing Technical Content)")
    
    # Create an invalid technical response without technical content
    try:
        invalid_technical = ChatbotResponse(
            message="I'll help you fix that problem. Let me know if you need anything else.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=technical_profile,
            content_categories=[ContentCategory.GENERAL, ContentCategory.SUPPORTIVE],
            emotion_levels={"neutral": 0.8}
        )
        
        print(f"Message: {invalid_technical.message}")
        print(f"State: {invalid_technical.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))


def demo_response_appropriateness() -> None:
    """Demonstrate response appropriateness validation."""
    print_header("Response Appropriateness Validation Demo")
    
    # Create a personality profile
    profile = PersonalityProfile()
    profile.add_trait("formal", 0.7)
    profile.add_trait("empathetic", 0.8)
    
    print_subheader("Valid Greeting Response")
    
    # Create a valid greeting response
    try:
        valid_greeting = ChatbotResponse(
            message="Hello! Welcome to our customer service. How may I assist you today?",
            conversation_state=ConversationState.GREETING,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL],
            emotion_levels={"neutral": 0.7, "enthusiasm": 0.3}
        )
        
        print(f"Message: {valid_greeting.message}")
        print(f"State: {valid_greeting.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Greeting Response (Missing Greeting)")
    
    # Create an invalid greeting response without a greeting
    try:
        invalid_greeting = ChatbotResponse(
            message="I can help you with your issue. What seems to be the problem?",
            conversation_state=ConversationState.GREETING,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL],
            emotion_levels={"neutral": 0.8}
        )
        
        print(f"Message: {invalid_greeting.message}")
        print(f"State: {invalid_greeting.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Valid Complaint Handling Response")
    
    # Create a valid complaint handling response
    try:
        valid_complaint = ChatbotResponse(
            message="I'm very sorry to hear about the issues you've experienced with our service. I understand how frustrating this must be for you. Let me help resolve this situation.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=profile,
            content_categories=[ContentCategory.EMOTIONAL, ContentCategory.SUPPORTIVE],
            emotion_levels={"empathy": 0.9, "neutral": 0.1}
        )
        
        print(f"Message: {valid_complaint.message}")
        print(f"State: {valid_complaint.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Complaint Handling Response (Missing Apology)")
    
    # Create an invalid complaint handling response without an apology
    try:
        invalid_complaint = ChatbotResponse(
            message="I'll help you resolve this issue. Please provide more details about what happened.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL, ContentCategory.DIRECTIVE],
            emotion_levels={"neutral": 0.7, "empathy": 0.3}
        )
        
        print(f"Message: {invalid_complaint.message}")
        print(f"State: {invalid_complaint.conversation_state.value}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))


def demo_emotion_levels() -> None:
    """Demonstrate emotion levels validation."""
    print_header("Emotion Levels Validation Demo")
    
    # Create a personality profile
    profile = PersonalityProfile()
    profile.add_trait("formal", 0.6)
    profile.add_trait("empathetic", 0.7)
    
    print_subheader("Valid Technical Support Response (Appropriate Emotion)")
    
    # Create a valid technical support response with appropriate emotion
    try:
        valid_technical = ChatbotResponse(
            message="I understand this technical issue can be frustrating. Let's troubleshoot it step by step to identify the root cause.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=profile,
            content_categories=[ContentCategory.TECHNICAL, ContentCategory.SUPPORTIVE],
            emotion_levels={"neutral": 0.6, "empathy": 0.4}
        )
        
        print(f"Message: {valid_technical.message}")
        print(f"State: {valid_technical.conversation_state.value}")
        print(f"Emotion Levels: neutral={valid_technical.emotion_levels.get('neutral', 0)}, empathy={valid_technical.emotion_levels.get('empathy', 0)}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Technical Support Response (Excessive Enthusiasm)")
    
    # Create an invalid technical support response with excessive enthusiasm
    try:
        invalid_technical = ChatbotResponse(
            message="I'm SUPER EXCITED to help you fix this technical problem!!! Let's get started RIGHT AWAY and make your system AWESOME again!!!",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=profile,
            content_categories=[ContentCategory.TECHNICAL, ContentCategory.SUPPORTIVE],
            emotion_levels={"enthusiasm": 0.9, "excitement": 0.8}
        )
        
        print(f"Message: {invalid_technical.message}")
        print(f"State: {invalid_technical.conversation_state.value}")
        print(f"Emotion Levels: enthusiasm={invalid_technical.emotion_levels.get('enthusiasm', 0)}, excitement={invalid_technical.emotion_levels.get('excitement', 0)}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Valid Complaint Handling Response (Appropriate Empathy)")
    
    # Create a valid complaint handling response with appropriate empathy
    try:
        valid_complaint = ChatbotResponse(
            message="I'm truly sorry to hear about your experience. I understand how frustrating this situation must be for you. I want to assure you that I'll do everything possible to resolve this issue.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=profile,
            content_categories=[ContentCategory.EMOTIONAL, ContentCategory.SUPPORTIVE],
            emotion_levels={"empathy": 0.9, "concern": 0.7}
        )
        
        print(f"Message: {valid_complaint.message}")
        print(f"State: {valid_complaint.conversation_state.value}")
        print(f"Emotion Levels: empathy={valid_complaint.emotion_levels.get('empathy', 0)}, concern={valid_complaint.emotion_levels.get('concern', 0)}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Complaint Handling Response (Insufficient Empathy)")
    
    # Create an invalid complaint handling response with insufficient empathy
    try:
        invalid_complaint = ChatbotResponse(
            message="I'll look into this issue for you. Please provide your order number.",
            conversation_state=ConversationState.HANDLING_COMPLAINT,
            personality_profile=profile,
            content_categories=[ContentCategory.DIRECTIVE],
            emotion_levels={"neutral": 0.9, "empathy": 0.2}
        )
        
        print(f"Message: {invalid_complaint.message}")
        print(f"State: {invalid_complaint.conversation_state.value}")
        print(f"Emotion Levels: neutral={invalid_complaint.emotion_levels.get('neutral', 0)}, empathy={invalid_complaint.emotion_levels.get('empathy', 0)}")
        print(f"Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))


def demo_conversation_flow() -> None:
    """Demonstrate conversation flow validation."""
    print_header("Conversation Flow Validation Demo")
    
    # Create a personality profile
    profile = PersonalityProfile()
    profile.add_trait("formal", 0.7)
    profile.add_trait("empathetic", 0.8)
    
    # Create a conversation context
    context = ConversationContext(
        conversation_id="demo-conversation",
        current_state=ConversationState.GREETING
    )
    
    print_conversation_context(context)
    
    print_subheader("Valid Conversation Flow")
    
    # Simulate a conversation with valid state transitions
    try:
        # Greeting -> Understanding Intent
        response1 = ChatbotResponse(
            message="Hello! Welcome to our customer service. How may I assist you today?",
            conversation_state=ConversationState.UNDERSTANDING_INTENT,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL]
        )
        context.add_chatbot_response(response1)
        print_bot_message(response1.message, response1.conversation_state)
        print_conversation_context(context)
        
        # Understanding Intent -> Handling Query
        response2 = ChatbotResponse(
            message="I'll help you with information about your order status.",
            conversation_state=ConversationState.HANDLING_QUERY,
            personality_profile=profile,
            content_categories=[ContentCategory.INFORMATIONAL]
        )
        context.add_chatbot_response(response2)
        print_bot_message(response2.message, response2.conversation_state)
        print_conversation_context(context)
        
        # Handling Query -> Providing Information
        response3 = ChatbotResponse(
            message="Your order #12345 is currently in transit and is expected to be delivered tomorrow by 5 PM.",
            conversation_state=ConversationState.PROVIDING_INFORMATION,
            personality_profile=profile,
            content_categories=[ContentCategory.INFORMATIONAL]
        )
        context.add_chatbot_response(response3)
        print_bot_message(response3.message, response3.conversation_state)
        print_conversation_context(context)
        
        # Providing Information -> Follow Up
        response4 = ChatbotResponse(
            message="Is there anything else you would like to know about your order?",
            conversation_state=ConversationState.FOLLOW_UP,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL]
        )
        context.add_chatbot_response(response4)
        print_bot_message(response4.message, response4.conversation_state)
        print_conversation_context(context)
        
        # Follow Up -> Ending
        response5 = ChatbotResponse(
            message="Thank you for contacting our customer service. Have a great day!",
            conversation_state=ConversationState.ENDING,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL]
        )
        context.add_chatbot_response(response5)
        print_bot_message(response5.message, response5.conversation_state)
        print_conversation_context(context)
        
        print(f"\nConversation Flow Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))
    
    print_subheader("Invalid Conversation Flow")
    
    # Create a new conversation context
    context2 = ConversationContext(
        conversation_id="demo-conversation-2",
        current_state=ConversationState.GREETING
    )
    
    print_conversation_context(context2)
    
    # Simulate a conversation with an invalid state transition
    try:
        # Greeting -> Understanding Intent (valid)
        response1 = ChatbotResponse(
            message="Hello! Welcome to our customer service. How may I assist you today?",
            conversation_state=ConversationState.UNDERSTANDING_INTENT,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL]
        )
        context2.add_chatbot_response(response1)
        print_bot_message(response1.message, response1.conversation_state)
        print_conversation_context(context2)
        
        # Understanding Intent -> Technical Support (valid)
        response2 = ChatbotResponse(
            message="I'll help you troubleshoot that technical issue.",
            conversation_state=ConversationState.TECHNICAL_SUPPORT,
            personality_profile=profile,
            content_categories=[ContentCategory.TECHNICAL]
        )
        context2.add_chatbot_response(response2)
        print_bot_message(response2.message, response2.conversation_state)
        print_conversation_context(context2)
        
        # Technical Support -> Greeting (invalid)
        response3 = ChatbotResponse(
            message="Hello again! How else can I help you?",
            conversation_state=ConversationState.GREETING,
            personality_profile=profile,
            content_categories=[ContentCategory.GENERAL]
        )
        context2.add_chatbot_response(response3)
        print_bot_message(response3.message, response3.conversation_state)
        print_conversation_context(context2)
        
        print(f"\nConversation Flow Validation: {Fore.GREEN}Passed{Style.RESET_ALL}")
    except ValueError as e:
        print_validation_error(str(e))


def demo_customer_service_chatbot() -> None:
    """Demonstrate a complete customer service chatbot interaction."""
    print_header("Customer Service Chatbot Demo")
    
    # Create a personality profile
    profile = PersonalityProfile()
    profile.add_trait("formal", 0.7)
    profile.add_trait("empathetic", 0.9)
    profile.add_trait("technical", 0.6)
    
    print_personality_profile(profile)
    
    # Create a customer service chatbot
    chatbot = CustomerServiceChatbot(
        name="ServiceBot",
        personality_profile=profile,
        supported_intents={"greeting", "complaint", "information", "technical_support", "farewell"},
        supported_languages={"en"}
    )
    
    print_subheader("Simulated Conversation")
    
    # Simulate a conversation
    try:
        # User greeting
        query1 = UserQuery(
            text="Hello, I need some help with my recent order.",
            detected_intent="greeting",
            detected_entities={"order": True}
        )
        print_user_message(query1.text)
        
        response1 = chatbot.generate_response(query1)
        print_bot_message(response1.message, response1.conversation_state)
        
        # User complaint
        query2 = UserQuery(
            text="I ordered a product last week, but it still hasn't arrived. The tracking information hasn't updated in 3 days.",
            detected_intent="complaint",
            detected_entities={"order": True, "product": True, "tracking": True},
            sentiment=-0.7
        )
        print_user_message(query2.text)
        
        response2 = chatbot.generate_response(query2)
        print_bot_message(response2.message, response2.conversation_state)
        
        # User provides order details
        query3 = UserQuery(
            text="My order number is ABC123456. I ordered it on May 15th.",
            detected_intent="information",
            detected_entities={"order_number": "ABC123456", "date": "May 15th"}
        )
        print_user_message(query3.text)
        
        response3 = chatbot.generate_response(query3)
        print_bot_message(response3.message, response3.conversation_state)
        
        # User asks for technical help
        query4 = UserQuery(
            text="I also can't seem to update my delivery address on your website. Can you help me with that?",
            detected_intent="technical_support",
            detected_entities={"delivery_address": True, "website": True}
        )
        print_user_message(query4.text)
        
        response4 = chatbot.generate_response(query4)
        print_bot_message(response4.message, response4.conversation_state)
        
        # User thanks and says goodbye
        query5 = UserQuery(
            text="Thank you for your help. I'll try those steps. Goodbye!",
            detected_intent="farewell",
            sentiment=0.8
        )
        print_user_message(query5.text)
        
        response5 = chatbot.generate_response(query5)
        print_bot_message(response5.message, response5.conversation_state)
        
        # Print final conversation context
        if chatbot.conversation_context:
            print_conversation_context(chatbot.conversation_context)
    except ValueError as e:
        print_validation_error(str(e))


def main() -> None:
    """Main function to run the demo."""
    print_header("Chatbot Agent Validation Demo")
    
    print(f"""
This demo showcases validation patterns specific to chatbot agents, including:

1. Personality consistency validation
2. Response appropriateness validation
3. Emotion levels validation
4. Conversation flow validation
5. Complete customer service chatbot interaction

These validation patterns help ensure that chatbot agents maintain consistent
personality traits while providing appropriate responses in different conversation contexts.
""")
    
    while True:
        print(f"\n{Fore.YELLOW}Choose a demo to run (1-5, or q to quit):{Style.RESET_ALL}")
        print("1. Personality Consistency Validation")
        print("2. Response Appropriateness Validation")
        print("3. Emotion Levels Validation")
        print("4. Conversation Flow Validation")
        print("5. Customer Service Chatbot Interaction")
        print("q. Quit")
        
        choice = input("> ").strip().lower()
        
        if choice == "q":
            break
        elif choice == "1":
            demo_personality_consistency()
        elif choice == "2":
            demo_response_appropriateness()
        elif choice == "3":
            demo_emotion_levels()
        elif choice == "4":
            demo_conversation_flow()
        elif choice == "5":
            demo_customer_service_chatbot()
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-5 or q.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
