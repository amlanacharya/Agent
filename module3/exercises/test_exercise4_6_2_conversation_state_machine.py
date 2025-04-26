"""
Tests for Exercise 4.6.2: Conversation State Machine
--------------------------------------------------
This module contains tests for the conversation state machine implementation.
"""

import unittest
from datetime import datetime, timedelta
import time
from pydantic import ValidationError

from exercise4_6_2_conversation_state_machine import (
    ConversationState,
    StateTransitionValidator,
    StateTransition,
    ConversationContext,
    ConversationStateMachine
)


class TestConversationStateMachine(unittest.TestCase):
    """Test cases for the conversation state machine implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = StateTransitionValidator()
        self.context = ConversationContext(conversation_id="test-conversation")
        self.state_machine = ConversationStateMachine(
            conversation_id="test-conversation",
            user_id="test-user"
        )

    def test_initial_state(self):
        """Test the initial state of the conversation."""
        self.assertEqual(self.context.state, ConversationState.GREETING.value)
        self.assertEqual(len(self.context.state_history), 1)
        self.assertEqual(self.context.state_history[0].from_state, "init")
        self.assertEqual(self.context.state_history[0].to_state, ConversationState.GREETING.value)

    def test_valid_transition(self):
        """Test a valid state transition."""
        # GREETING -> COLLECTING_INFO is valid
        self.context.transition_to(
            ConversationState.COLLECTING_INFO.value,
            self.validator,
            reason="User provided query"
        )

        self.assertEqual(self.context.state, ConversationState.COLLECTING_INFO.value)
        self.assertEqual(len(self.context.state_history), 2)
        self.assertEqual(self.context.state_history[1].from_state, ConversationState.GREETING.value)
        self.assertEqual(self.context.state_history[1].to_state, ConversationState.COLLECTING_INFO.value)
        self.assertEqual(self.context.state_history[1].reason, "User provided query")

    def test_invalid_transition(self):
        """Test an invalid state transition."""
        # GREETING -> PROCESSING is invalid
        with self.assertRaises(ValueError):
            self.context.transition_to(
                ConversationState.PROCESSING.value,
                self.validator
            )

        # State should remain unchanged
        self.assertEqual(self.context.state, ConversationState.GREETING.value)
        self.assertEqual(len(self.context.state_history), 1)

    def test_state_machine_transitions(self):
        """Test transitions using the state machine interface."""
        # Valid transition
        result = self.state_machine.transition_to(
            ConversationState.COLLECTING_INFO.value,
            reason="User provided query"
        )
        self.assertTrue(result)
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.COLLECTING_INFO.value)

        # Another valid transition
        result = self.state_machine.transition_to(
            ConversationState.PROCESSING.value,
            reason="Collected all required information"
        )
        self.assertTrue(result)
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.PROCESSING.value)

        # Invalid transition
        result = self.state_machine.transition_to(
            ConversationState.FOLLOW_UP.value,
            reason="Attempt to skip providing results"
        )
        self.assertFalse(result)
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.PROCESSING.value)

    def test_force_transition(self):
        """Test forcing a transition that would normally be invalid."""
        # Force an invalid transition
        self.state_machine.force_transition(
            ConversationState.FOLLOW_UP.value,
            reason="Forced for testing"
        )

        # State should be updated despite being an invalid transition
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.FOLLOW_UP.value)

        # Check that the history records the forced transition
        history = self.state_machine.get_state_history()
        self.assertEqual(history[-1]["to_state"], ConversationState.FOLLOW_UP.value)
        self.assertEqual(history[-1]["reason"], "Forced for testing")
        self.assertTrue(history[-1]["metadata"].get("forced", False))

    def test_context_data(self):
        """Test adding and retrieving context data."""
        # Add context data
        self.state_machine.add_context_data("query_type", "weather")
        self.state_machine.add_context_data("location", "New York")

        # Retrieve context data
        self.assertEqual(self.state_machine.get_context_data("query_type"), "weather")
        self.assertEqual(self.state_machine.get_context_data("location"), "New York")
        self.assertIsNone(self.state_machine.get_context_data("non_existent_key"))
        self.assertEqual(
            self.state_machine.get_context_data("non_existent_key", "default"),
            "default"
        )

    def test_allowed_next_states(self):
        """Test getting allowed next states."""
        # From GREETING
        allowed_states = self.state_machine.get_allowed_next_states()
        self.assertIn(ConversationState.COLLECTING_INFO.value, allowed_states)
        self.assertIn(ConversationState.ENDING.value, allowed_states)
        self.assertEqual(len(allowed_states), 2)

        # Transition to COLLECTING_INFO
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # From COLLECTING_INFO
        allowed_states = self.state_machine.get_allowed_next_states()
        self.assertIn(ConversationState.COLLECTING_INFO.value, allowed_states)  # Can stay in this state
        self.assertIn(ConversationState.PROCESSING.value, allowed_states)
        self.assertIn(ConversationState.CLARIFICATION.value, allowed_states)
        self.assertIn(ConversationState.ERROR_HANDLING.value, allowed_states)
        self.assertEqual(len(allowed_states), 4)

    def test_can_transition_to(self):
        """Test checking if a transition is valid."""
        # From GREETING
        self.assertTrue(self.state_machine.can_transition_to(ConversationState.COLLECTING_INFO.value))
        self.assertTrue(self.state_machine.can_transition_to(ConversationState.ENDING.value))
        self.assertFalse(self.state_machine.can_transition_to(ConversationState.PROCESSING.value))

        # Transition to COLLECTING_INFO
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # From COLLECTING_INFO
        self.assertTrue(self.state_machine.can_transition_to(ConversationState.COLLECTING_INFO.value))
        self.assertTrue(self.state_machine.can_transition_to(ConversationState.PROCESSING.value))
        self.assertFalse(self.state_machine.can_transition_to(ConversationState.FOLLOW_UP.value))

    def test_state_history(self):
        """Test state history tracking."""
        # Perform a series of transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.transition_to(ConversationState.PROCESSING.value)
        self.state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value)
        self.state_machine.transition_to(ConversationState.FOLLOW_UP.value)

        # Check history
        history = self.state_machine.get_state_history()
        self.assertEqual(len(history), 5)  # Initial state + 4 transitions

        # Check sequence of states
        states = [h["to_state"] for h in history]
        self.assertEqual(states, [
            ConversationState.GREETING.value,
            ConversationState.COLLECTING_INFO.value,
            ConversationState.PROCESSING.value,
            ConversationState.PROVIDING_RESULTS.value,
            ConversationState.FOLLOW_UP.value
        ])

    def test_reset(self):
        """Test resetting the state machine."""
        # Perform some transitions and add context data
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.add_context_data("query_type", "weather")

        # Reset the state machine
        self.state_machine.reset()

        # Check that state is reset but conversation_id and user_id are preserved
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.GREETING.value)
        self.assertEqual(len(self.state_machine.get_state_history()), 1)
        self.assertEqual(self.state_machine.context.conversation_id, "test-conversation")
        self.assertEqual(self.state_machine.context.user_id, "test-user")

        # Context data should be cleared
        self.assertIsNone(self.state_machine.get_context_data("query_type"))

    def test_custom_validator(self):
        """Test using a custom validator with different rules."""
        # Create a custom validator with restricted transitions
        custom_validator = StateTransitionValidator(
            allowed_transitions={
                ConversationState.GREETING.value: {
                    ConversationState.COLLECTING_INFO.value
                },
                ConversationState.COLLECTING_INFO.value: {
                    ConversationState.ENDING.value
                },
                ConversationState.ENDING.value: set()  # No transitions from ENDING
            }
        )

        # Create a state machine with the custom validator
        custom_machine = ConversationStateMachine(validator=custom_validator)

        # Test transitions with custom rules
        self.assertTrue(custom_machine.can_transition_to(ConversationState.COLLECTING_INFO.value))
        self.assertFalse(custom_machine.can_transition_to(ConversationState.ENDING.value))

        # Perform valid transition
        custom_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Check new allowed transitions
        self.assertTrue(custom_machine.can_transition_to(ConversationState.ENDING.value))
        self.assertFalse(custom_machine.can_transition_to(ConversationState.PROCESSING.value))

    def test_conversation_duration(self):
        """Test getting conversation duration."""
        # Create a context with a state history entry from the past
        past_time = datetime.now() - timedelta(minutes=5)
        context = ConversationContext()
        context.state_history[0].timestamp = past_time

        # Duration should be approximately 5 minutes
        duration = context.get_conversation_duration()
        self.assertGreaterEqual(duration, 290)  # At least 4:50 minutes in seconds
        self.assertLessEqual(duration, 310)  # At most 5:10 minutes in seconds

    def test_state_duration(self):
        """Test getting state duration."""
        # Create a context and wait a short time
        context = ConversationContext()
        time.sleep(0.1)  # Wait 100ms

        # Duration should be at least 100ms
        duration = context.get_state_duration()
        self.assertGreaterEqual(duration, 0.1)

    def test_invalid_state(self):
        """Test validation of invalid state values."""
        # Try to create a context with an invalid state
        with self.assertRaises(ValidationError):
            ConversationContext(state="invalid_state")

    def test_complete_conversation_flow(self):
        """Test a complete conversation flow with all states."""
        # Start a new conversation
        machine = ConversationStateMachine()

        # Complete flow: GREETING -> COLLECTING_INFO -> PROCESSING ->
        # PROVIDING_RESULTS -> FOLLOW_UP -> ENDING -> GREETING (new conversation)
        self.assertEqual(machine.get_current_state(), ConversationState.GREETING.value)

        machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.assertEqual(machine.get_current_state(), ConversationState.COLLECTING_INFO.value)

        machine.transition_to(ConversationState.PROCESSING.value)
        self.assertEqual(machine.get_current_state(), ConversationState.PROCESSING.value)

        machine.transition_to(ConversationState.PROVIDING_RESULTS.value)
        self.assertEqual(machine.get_current_state(), ConversationState.PROVIDING_RESULTS.value)

        machine.transition_to(ConversationState.FOLLOW_UP.value)
        self.assertEqual(machine.get_current_state(), ConversationState.FOLLOW_UP.value)

        machine.transition_to(ConversationState.ENDING.value)
        self.assertEqual(machine.get_current_state(), ConversationState.ENDING.value)

        # Start a new conversation
        machine.transition_to(ConversationState.GREETING.value)
        self.assertEqual(machine.get_current_state(), ConversationState.GREETING.value)

        # Check that history contains all transitions
        history = machine.get_state_history()
        self.assertEqual(len(history), 7)  # Initial state + 6 transitions


if __name__ == "__main__":
    unittest.main()
