"""
Exercise 4.6.2: Conversation State Machine
-----------------------------------------
This module implements a conversation state machine that enforces valid transitions
between different conversation phases.

The state machine tracks the current state of a conversation and ensures that
transitions between states follow predefined rules. It also maintains a history
of state transitions and associated context data.
"""

from enum import Enum
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class ConversationState(Enum):
    """Possible states in a conversation flow."""
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    PROVIDING_RESULTS = "providing_results"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    ERROR_HANDLING = "error_handling"
    ENDING = "ending"


class StateTransitionValidator(BaseModel):
    """Validates transitions between conversation states."""
    allowed_transitions: Dict[str, Set[str]] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """Initialize with default transition rules if not provided."""
        if 'allowed_transitions' not in data:
            data['allowed_transitions'] = {
                ConversationState.GREETING.value: {
                    ConversationState.COLLECTING_INFO.value,
                    ConversationState.ENDING.value
                },
                ConversationState.COLLECTING_INFO.value: {
                    ConversationState.COLLECTING_INFO.value,  # Can stay in this state
                    ConversationState.PROCESSING.value,
                    ConversationState.CLARIFICATION.value,
                    ConversationState.ERROR_HANDLING.value
                },
                ConversationState.PROCESSING.value: {
                    ConversationState.PROVIDING_RESULTS.value,
                    ConversationState.ERROR_HANDLING.value,
                    ConversationState.COLLECTING_INFO.value  # Can go back if more info needed
                },
                ConversationState.PROVIDING_RESULTS.value: {
                    ConversationState.FOLLOW_UP.value,
                    ConversationState.ENDING.value,
                    ConversationState.CLARIFICATION.value
                },
                ConversationState.FOLLOW_UP.value: {
                    ConversationState.COLLECTING_INFO.value,
                    ConversationState.PROCESSING.value,
                    ConversationState.ENDING.value
                },
                ConversationState.CLARIFICATION.value: {
                    ConversationState.COLLECTING_INFO.value,
                    ConversationState.PROCESSING.value,
                    ConversationState.PROVIDING_RESULTS.value
                },
                ConversationState.ERROR_HANDLING.value: {
                    ConversationState.COLLECTING_INFO.value,
                    ConversationState.ENDING.value,
                    ConversationState.GREETING.value  # Can restart
                },
                ConversationState.ENDING.value: {
                    ConversationState.GREETING.value  # Can start a new conversation
                }
            }
        super().__init__(**data)
    
    def validate_transition(self, current_state: str, new_state: str) -> bool:
        """
        Validate that a state transition is allowed.
        
        Args:
            current_state: The current state of the conversation
            new_state: The proposed new state
            
        Returns:
            bool: True if the transition is valid, False otherwise
            
        Raises:
            ValueError: If the current state is unknown
        """
        if current_state not in self.allowed_transitions:
            raise ValueError(f"Unknown current state: {current_state}")
        
        return new_state in self.allowed_transitions[current_state]
    
    def get_allowed_next_states(self, current_state: str) -> Set[str]:
        """
        Get all allowed next states from the current state.
        
        Args:
            current_state: The current state of the conversation
            
        Returns:
            Set[str]: Set of allowed next states
            
        Raises:
            ValueError: If the current state is unknown
        """
        if current_state not in self.allowed_transitions:
            raise ValueError(f"Unknown current state: {current_state}")
        
        return self.allowed_transitions[current_state]


class StateTransition(BaseModel):
    """Record of a state transition."""
    from_state: str
    to_state: str
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """
    Tracks the state and context of a conversation.
    
    This model maintains the current state, history of state transitions,
    and any context data associated with the conversation.
    """
    state: str = Field(default=ConversationState.GREETING.value)
    state_history: List[StateTransition] = Field(default_factory=list)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    last_state_change: datetime = Field(default_factory=datetime.now)
    conversation_id: str = Field(default="")
    user_id: Optional[str] = None
    
    def __init__(self, **data):
        """Initialize with a default state transition in history."""
        super().__init__(**data)
        # Add initial state to history if not already present
        if not self.state_history:
            self.state_history.append(
                StateTransition(
                    from_state="init",
                    to_state=self.state,
                    timestamp=self.last_state_change
                )
            )
    
    def transition_to(self, new_state: str, validator: StateTransitionValidator, 
                      reason: Optional[str] = None, metadata: Dict[str, Any] = None) -> "ConversationContext":
        """
        Transition to a new state if valid.
        
        Args:
            new_state: The new state to transition to
            validator: The validator to use for checking transition validity
            reason: Optional reason for the transition
            metadata: Optional metadata to associate with the transition
            
        Returns:
            ConversationContext: Self, for method chaining
            
        Raises:
            ValueError: If the transition is invalid
        """
        if not validator.validate_transition(self.state, new_state):
            raise ValueError(f"Invalid state transition from {self.state} to {new_state}")
        
        # Create transition record
        transition = StateTransition(
            from_state=self.state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason,
            metadata=metadata or {}
        )
        
        # Update state
        self.state = new_state
        self.state_history.append(transition)
        self.last_state_change = transition.timestamp
        
        return self
    
    def add_context_data(self, key: str, value: Any) -> "ConversationContext":
        """
        Add data to the conversation context.
        
        Args:
            key: The key for the context data
            value: The value to store
            
        Returns:
            ConversationContext: Self, for method chaining
        """
        self.context_data[key] = value
        return self
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from the conversation context.
        
        Args:
            key: The key for the context data
            default: Default value to return if key not found
            
        Returns:
            Any: The value associated with the key, or default if not found
        """
        return self.context_data.get(key, default)
    
    def get_state_duration(self) -> float:
        """
        Get the duration in seconds that the conversation has been in the current state.
        
        Returns:
            float: Duration in seconds
        """
        return (datetime.now() - self.last_state_change).total_seconds()
    
    def get_conversation_duration(self) -> float:
        """
        Get the total duration of the conversation in seconds.
        
        Returns:
            float: Duration in seconds
        """
        if not self.state_history:
            return 0
        
        start_time = self.state_history[0].timestamp
        return (datetime.now() - start_time).total_seconds()
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """
        Get a formatted history of state transitions.
        
        Returns:
            List[Dict[str, Any]]: List of state transition records
        """
        return [
            {
                "from_state": t.from_state,
                "to_state": t.to_state,
                "timestamp": t.timestamp.isoformat(),
                "reason": t.reason,
                "metadata": t.metadata
            }
            for t in self.state_history
        ]
    
    @model_validator(mode='after')
    def validate_state(self):
        """Validate that the state is a valid ConversationState value."""
        valid_states = [state.value for state in ConversationState]
        if self.state not in valid_states:
            raise ValueError(f"Invalid state: {self.state}. Must be one of {valid_states}")
        return self


class ConversationStateMachine:
    """
    Manages conversation state transitions and associated business logic.
    
    This class provides a higher-level interface for managing conversation
    state transitions, including validation and context management.
    """
    def __init__(self, initial_state: str = ConversationState.GREETING.value,
                 validator: Optional[StateTransitionValidator] = None,
                 conversation_id: str = "", user_id: Optional[str] = None):
        """
        Initialize the conversation state machine.
        
        Args:
            initial_state: The initial state of the conversation
            validator: Optional custom validator for state transitions
            conversation_id: Optional ID for the conversation
            user_id: Optional ID for the user
        """
        self.validator = validator or StateTransitionValidator()
        self.context = ConversationContext(
            state=initial_state,
            conversation_id=conversation_id,
            user_id=user_id
        )
    
    def get_current_state(self) -> str:
        """
        Get the current state of the conversation.
        
        Returns:
            str: The current state
        """
        return self.context.state
    
    def transition_to(self, new_state: str, reason: Optional[str] = None,
                      metadata: Dict[str, Any] = None) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: The new state to transition to
            reason: Optional reason for the transition
            metadata: Optional metadata to associate with the transition
            
        Returns:
            bool: True if the transition was successful, False otherwise
        """
        try:
            self.context.transition_to(new_state, self.validator, reason, metadata)
            return True
        except ValueError:
            return False
    
    def force_transition(self, new_state: str, reason: str = "Forced transition",
                         metadata: Dict[str, Any] = None) -> None:
        """
        Force a transition to a new state, bypassing validation.
        
        This should be used with caution, as it can lead to invalid state sequences.
        
        Args:
            new_state: The new state to transition to
            reason: Reason for the forced transition
            metadata: Optional metadata to associate with the transition
        """
        # Create transition record
        transition = StateTransition(
            from_state=self.context.state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason,
            metadata=metadata or {"forced": True}
        )
        
        # Update state
        self.context.state = new_state
        self.context.state_history.append(transition)
        self.context.last_state_change = transition.timestamp
    
    def can_transition_to(self, new_state: str) -> bool:
        """
        Check if a transition to a new state is valid.
        
        Args:
            new_state: The new state to check
            
        Returns:
            bool: True if the transition is valid, False otherwise
        """
        try:
            return self.validator.validate_transition(self.context.state, new_state)
        except ValueError:
            return False
    
    def get_allowed_next_states(self) -> List[str]:
        """
        Get all allowed next states from the current state.
        
        Returns:
            List[str]: List of allowed next states
        """
        try:
            return list(self.validator.get_allowed_next_states(self.context.state))
        except ValueError:
            return []
    
    def add_context_data(self, key: str, value: Any) -> None:
        """
        Add data to the conversation context.
        
        Args:
            key: The key for the context data
            value: The value to store
        """
        self.context.add_context_data(key, value)
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from the conversation context.
        
        Args:
            key: The key for the context data
            default: Default value to return if key not found
            
        Returns:
            Any: The value associated with the key, or default if not found
        """
        return self.context.get_context_data(key, default)
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """
        Get a formatted history of state transitions.
        
        Returns:
            List[Dict[str, Any]]: List of state transition records
        """
        return self.context.get_state_history()
    
    def reset(self, initial_state: str = ConversationState.GREETING.value) -> None:
        """
        Reset the state machine to its initial state.
        
        Args:
            initial_state: The initial state to reset to
        """
        conversation_id = self.context.conversation_id
        user_id = self.context.user_id
        
        self.context = ConversationContext(
            state=initial_state,
            conversation_id=conversation_id,
            user_id=user_id
        )


# Example usage
if __name__ == "__main__":
    # Create a state machine
    state_machine = ConversationStateMachine()
    
    print(f"Initial state: {state_machine.get_current_state()}")
    print(f"Allowed next states: {state_machine.get_allowed_next_states()}")
    
    # Perform valid transitions
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value, 
                               reason="User provided initial query")
    print(f"Transitioned to: {state_machine.get_current_state()}")
    
    # Add context data
    state_machine.add_context_data("query_type", "weather")
    state_machine.add_context_data("location", "New York")
    
    # Another valid transition
    state_machine.transition_to(ConversationState.PROCESSING.value,
                               reason="Collected all required information")
    print(f"Transitioned to: {state_machine.get_current_state()}")
    
    # Try an invalid transition
    result = state_machine.transition_to(ConversationState.FOLLOW_UP.value)
    print(f"Attempted invalid transition, result: {result}")
    
    # Complete the flow
    state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value)
    state_machine.transition_to(ConversationState.FOLLOW_UP.value)
    state_machine.transition_to(ConversationState.ENDING.value)
    
    # Print state history
    print("\nState history:")
    for transition in state_machine.get_state_history():
        print(f"  {transition['from_state']} -> {transition['to_state']} at {transition['timestamp']}")
