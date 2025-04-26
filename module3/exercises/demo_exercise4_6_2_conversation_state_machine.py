"""
Demo for Exercise 4.6.2: Conversation State Machine
-------------------------------------------------
This script demonstrates the usage of the Conversation State Machine implementation.
It simulates a conversation flow for a weather information agent.
"""

import time
from datetime import datetime
from colorama import init, Fore, Style
from exercise4_6_2_conversation_state_machine import (
    ConversationState,
    StateTransitionValidator,
    ConversationStateMachine
)

# Initialize colorama for colored terminal output
init()


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + f"{Style.RESET_ALL}\n")


def print_state_change(from_state, to_state):
    """Print a state transition."""
    print(f"{Fore.YELLOW}State transition: {Fore.WHITE}{from_state} {Fore.GREEN}→ {Fore.WHITE}{to_state}{Style.RESET_ALL}")


def print_user_input(text):
    """Print user input."""
    print(f"\n{Fore.BLUE}User: {Fore.WHITE}{text}{Style.RESET_ALL}")


def print_agent_response(text):
    """Print agent response."""
    print(f"{Fore.GREEN}Agent: {Fore.WHITE}{text}{Style.RESET_ALL}")


def print_system_message(text):
    """Print a system message."""
    print(f"{Fore.MAGENTA}System: {Fore.WHITE}{text}{Style.RESET_ALL}")


def print_state_info(state_machine):
    """Print information about the current state."""
    current_state = state_machine.get_current_state()
    allowed_states = state_machine.get_allowed_next_states()
    context_data = state_machine.context.context_data
    
    print(f"\n{Fore.CYAN}Current State: {Fore.WHITE}{current_state}")
    print(f"{Fore.CYAN}Allowed Next States: {Fore.WHITE}{', '.join(allowed_states)}")
    print(f"{Fore.CYAN}Context Data: {Fore.WHITE}{context_data}")
    print(f"{Fore.CYAN}State Duration: {Fore.WHITE}{state_machine.context.get_state_duration():.2f} seconds")
    print(f"{Fore.CYAN}Conversation Duration: {Fore.WHITE}{state_machine.context.get_conversation_duration():.2f} seconds{Style.RESET_ALL}")


def simulate_weather_conversation():
    """Simulate a conversation about weather with state transitions."""
    print_header("Weather Information Agent Demo")
    print("This demo simulates a conversation with a weather information agent,")
    print("showing how the conversation state machine manages state transitions.")
    
    # Create a state machine for the conversation
    state_machine = ConversationStateMachine(
        conversation_id="weather-demo-1",
        user_id="demo-user"
    )
    
    # Start in GREETING state
    print_state_info(state_machine)
    print_agent_response("Hello! I'm your weather assistant. How can I help you today?")
    
    # User provides initial query
    time.sleep(1)
    print_user_input("What's the weather like?")
    
    # Transition to COLLECTING_INFO state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.COLLECTING_INFO.value,
        reason="User provided initial query"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent asks for location
    print_agent_response("I'd be happy to help you with that. Could you please tell me which location you're interested in?")
    
    # User provides location
    time.sleep(1.5)
    print_user_input("New York")
    
    # Add context data
    state_machine.add_context_data("location", "New York")
    print_system_message("Added 'location: New York' to context data")
    
    # Transition to PROCESSING state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.PROCESSING.value,
        reason="Collected required location information",
        metadata={"collected_info": ["location"]}
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent processes the request
    print_agent_response("Let me check the weather in New York for you...")
    
    # Simulate processing delay
    print_system_message("Processing weather data...")
    time.sleep(2)
    
    # Add weather data to context
    state_machine.add_context_data("weather_data", {
        "temperature": 72,
        "condition": "Partly Cloudy",
        "humidity": 65,
        "wind_speed": 8
    })
    print_system_message("Added weather data to context")
    
    # Transition to PROVIDING_RESULTS state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.PROVIDING_RESULTS.value,
        reason="Weather data retrieved successfully"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent provides weather information
    weather_data = state_machine.get_context_data("weather_data")
    location = state_machine.get_context_data("location")
    print_agent_response(
        f"The current weather in {location} is {weather_data['condition']} with a temperature of "
        f"{weather_data['temperature']}°F. The humidity is {weather_data['humidity']}% and "
        f"wind speed is {weather_data['wind_speed']} mph. Is there anything else you'd like to know?"
    )
    
    # User asks follow-up question
    time.sleep(1.5)
    print_user_input("Will it rain tomorrow?")
    
    # Transition to FOLLOW_UP state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.FOLLOW_UP.value,
        reason="User asked follow-up question"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Transition back to PROCESSING for the follow-up
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.PROCESSING.value,
        reason="Processing follow-up question",
        metadata={"follow_up_type": "forecast"}
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent processes the follow-up
    print_agent_response("Let me check the forecast for tomorrow in New York...")
    
    # Simulate processing delay
    print_system_message("Processing forecast data...")
    time.sleep(1.5)
    
    # Add forecast data to context
    state_machine.add_context_data("forecast_data", {
        "tomorrow": {
            "condition": "Rain",
            "precipitation_chance": 70,
            "temperature_high": 68,
            "temperature_low": 55
        }
    })
    print_system_message("Added forecast data to context")
    
    # Transition to PROVIDING_RESULTS state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.PROVIDING_RESULTS.value,
        reason="Forecast data retrieved successfully"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent provides forecast information
    forecast = state_machine.get_context_data("forecast_data")["tomorrow"]
    print_agent_response(
        f"Yes, there's a {forecast['precipitation_chance']}% chance of rain tomorrow in {location}. "
        f"The forecast shows {forecast['condition']} with temperatures between {forecast['temperature_low']}°F "
        f"and {forecast['temperature_high']}°F. Would you like any other information?"
    )
    
    # User ends the conversation
    time.sleep(1.5)
    print_user_input("No, that's all. Thank you!")
    
    # Transition to ENDING state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.ENDING.value,
        reason="User indicated end of conversation"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent ends the conversation
    print_agent_response("You're welcome! Have a great day. Feel free to ask if you need any more weather information.")
    
    # Print conversation summary
    print_header("Conversation Summary")
    print(f"{Fore.CYAN}Total Duration: {Fore.WHITE}{state_machine.context.get_conversation_duration():.2f} seconds")
    print(f"{Fore.CYAN}Total State Transitions: {Fore.WHITE}{len(state_machine.get_state_history()) - 1}")
    print(f"{Fore.CYAN}Final Context Data: {Fore.WHITE}{state_machine.context.context_data}")
    
    print("\n{Fore.CYAN}State Transition History:{Style.RESET_ALL}")
    for i, transition in enumerate(state_machine.get_state_history()):
        if i == 0:
            print(f"  {i+1}. Initial state: {transition['to_state']}")
        else:
            print(f"  {i+1}. {transition['from_state']} → {transition['to_state']} ({transition['reason']})")
    
    print_header("Demo Complete")


def demonstrate_error_handling():
    """Demonstrate error handling and recovery in the state machine."""
    print_header("Error Handling and Recovery Demo")
    
    # Create a state machine
    state_machine = ConversationStateMachine(conversation_id="error-demo")
    
    print_state_info(state_machine)
    print_agent_response("Hello! How can I assist you today?")
    
    # User provides query
    print_user_input("I want to book a flight")
    
    # Transition to COLLECTING_INFO
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
    print_state_change(previous_state, state_machine.get_current_state())
    
    # Agent asks for information
    print_agent_response("I can help you book a flight. Where would you like to fly from?")
    
    # User provides departure
    print_user_input("New York")
    state_machine.add_context_data("departure", "New York")
    
    # Agent asks for destination
    print_agent_response("And where would you like to fly to?")
    
    # User provides invalid input
    print_user_input("I'm not sure yet, maybe somewhere warm?")
    
    # Transition to CLARIFICATION state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.CLARIFICATION.value,
        reason="User provided ambiguous information"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent asks for clarification
    print_agent_response("I understand you're looking for somewhere warm. Could you provide a specific city or country you're interested in?")
    
    # User provides clarification
    print_user_input("How about Miami?")
    
    # Transition back to COLLECTING_INFO
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.COLLECTING_INFO.value,
        reason="User provided clarification"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    state_machine.add_context_data("destination", "Miami")
    
    # Simulate an error
    print_system_message("Simulating a system error...")
    
    # Transition to ERROR_HANDLING state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.ERROR_HANDLING.value,
        reason="System error occurred",
        metadata={"error_type": "api_failure", "error_message": "Flight booking service unavailable"}
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent handles the error
    print_agent_response("I apologize, but I'm currently having trouble connecting to our flight booking service. Would you like to try again later or is there something else I can help you with?")
    
    # User decides to end the conversation
    print_user_input("I'll try again later, thanks")
    
    # Transition to ENDING state
    previous_state = state_machine.get_current_state()
    state_machine.transition_to(
        ConversationState.ENDING.value,
        reason="User decided to end conversation after error"
    )
    print_state_change(previous_state, state_machine.get_current_state())
    print_state_info(state_machine)
    
    # Agent ends the conversation
    print_agent_response("I understand. I apologize for the inconvenience. Please try again later when our systems are back up. Have a great day!")
    
    print_header("Error Handling Demo Complete")


def demonstrate_invalid_transitions():
    """Demonstrate handling of invalid state transitions."""
    print_header("Invalid Transitions Demo")
    
    # Create a state machine
    state_machine = ConversationStateMachine()
    
    print_state_info(state_machine)
    
    # Try an invalid transition
    print_system_message("Attempting invalid transition: GREETING → PROCESSING")
    result = state_machine.transition_to(ConversationState.PROCESSING.value)
    print(f"{Fore.RED}Result: {Fore.WHITE}{'Success' if result else 'Failed'}{Style.RESET_ALL}")
    print_state_info(state_machine)
    
    # Try a valid transition
    print_system_message("Attempting valid transition: GREETING → COLLECTING_INFO")
    result = state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
    print(f"{Fore.GREEN}Result: {Fore.WHITE}{'Success' if result else 'Failed'}{Style.RESET_ALL}")
    print_state_info(state_machine)
    
    # Try another invalid transition
    print_system_message("Attempting invalid transition: COLLECTING_INFO → FOLLOW_UP")
    result = state_machine.transition_to(ConversationState.FOLLOW_UP.value)
    print(f"{Fore.RED}Result: {Fore.WHITE}{'Success' if result else 'Failed'}{Style.RESET_ALL}")
    print_state_info(state_machine)
    
    # Force an invalid transition
    print_system_message("Forcing invalid transition: COLLECTING_INFO → FOLLOW_UP")
    state_machine.force_transition(
        ConversationState.FOLLOW_UP.value,
        reason="Forced for demonstration"
    )
    print_state_info(state_machine)
    
    print_header("Invalid Transitions Demo Complete")


if __name__ == "__main__":
    try:
        # Run the weather conversation demo
        simulate_weather_conversation()
        
        # Wait a moment before the next demo
        time.sleep(1)
        
        # Run the error handling demo
        demonstrate_error_handling()
        
        # Wait a moment before the next demo
        time.sleep(1)
        
        # Run the invalid transitions demo
        demonstrate_invalid_transitions()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Demo interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during demo: {e}{Style.RESET_ALL}")
