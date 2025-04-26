"""
Exercise 4.6.3: State Recovery System
-----------------------------------
This module implements a state recovery system that can detect and fix inconsistent states
in conversation contexts. It provides mechanisms for:

1. Detecting inconsistent or corrupted states
2. Recovering from errors using various strategies
3. Maintaining state integrity across interactions
4. Creating and restoring from backups
5. Handling state corruption gracefully
"""

import json
import os
import copy
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from pydantic import BaseModel, Field, model_validator

# Import the conversation state machine from the previous exercise
from exercise4_6_2_conversation_state_machine import (
    ConversationState,
    StateTransitionValidator,
    StateTransition,
    ConversationContext,
    ConversationStateMachine
)


class StateIntegrityError(Exception):
    """Exception raised when state integrity is compromised."""
    pass


class RecoveryStrategy(Enum):
    """Strategies for recovering from state errors."""
    RESET = "reset"  # Reset to initial state
    ROLLBACK = "rollback"  # Rollback to last valid state
    REPAIR = "repair"  # Attempt to repair the state
    USE_BACKUP = "use_backup"  # Use a backup state
    FORCE_TRANSITION = "force_transition"  # Force a transition to a valid state


class StateConsistencyLevel(Enum):
    """Levels of state consistency checks."""
    BASIC = "basic"  # Check basic state validity
    STANDARD = "standard"  # Check state transitions and context data
    STRICT = "strict"  # Comprehensive validation of all state aspects


class StateBackup(BaseModel):
    """Model for state backups."""
    backup_id: str = Field(default_factory=lambda: f"backup_{int(time.time())}_{int(time.time()*1000) % 1000}")
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: str
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateRecoveryResult(BaseModel):
    """Result of a state recovery operation."""
    success: bool
    strategy_used: RecoveryStrategy
    original_state: str
    recovered_state: str
    error_details: Optional[str] = None
    recovery_time: float
    changes_made: List[str] = Field(default_factory=list)


class StateValidator(BaseModel):
    """Validates conversation state integrity."""
    validator: StateTransitionValidator = Field(default_factory=StateTransitionValidator)

    def validate_state(self, context: ConversationContext,
                       level: StateConsistencyLevel = StateConsistencyLevel.STANDARD) -> List[str]:
        """
        Validate the integrity of a conversation state.

        Args:
            context: The conversation context to validate
            level: The level of consistency checking to perform

        Returns:
            List of validation errors, empty if state is valid
        """
        errors = []

        # Basic validation - check if state is a valid enum value
        valid_states = [state.value for state in ConversationState]
        if context.state not in valid_states:
            errors.append(f"Invalid state: {context.state}. Must be one of {valid_states}")

        if level == StateConsistencyLevel.BASIC:
            return errors

        # Standard validation - check state transitions
        if context.state_history:
            for i, transition in enumerate(context.state_history[1:], 1):
                prev_state = context.state_history[i-1].to_state
                curr_state = transition.from_state

                # Check if history is consistent
                if prev_state != curr_state:
                    errors.append(
                        f"Inconsistent state history: transition {i} has from_state '{curr_state}' "
                        f"but previous transition ended in '{prev_state}'"
                    )

                # Check if transitions were valid
                if not self._is_valid_transition(transition.from_state, transition.to_state):
                    errors.append(
                        f"Invalid state transition in history: {transition.from_state} -> {transition.to_state}"
                    )

        # Check if current state matches the last state in history
        if context.state_history and context.state != context.state_history[-1].to_state:
            errors.append(
                f"Current state '{context.state}' doesn't match last state in history "
                f"'{context.state_history[-1].to_state}'"
            )

        if level == StateConsistencyLevel.STANDARD:
            return errors

        # Strict validation - check timestamps and context data integrity
        if context.state_history:
            # Check if timestamps are in chronological order
            for i, transition in enumerate(context.state_history[1:], 1):
                prev_time = context.state_history[i-1].timestamp
                curr_time = transition.timestamp

                if curr_time < prev_time:
                    errors.append(
                        f"Non-chronological timestamps: transition {i} at {curr_time} "
                        f"is earlier than previous transition at {prev_time}"
                    )

        # Check if last_state_change matches the timestamp of the last transition
        if context.state_history and context.last_state_change != context.state_history[-1].timestamp:
            errors.append(
                f"last_state_change ({context.last_state_change}) doesn't match "
                f"timestamp of last transition ({context.state_history[-1].timestamp})"
            )

        return errors

    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a transition is valid according to the validator."""
        try:
            return self.validator.validate_transition(from_state, to_state)
        except ValueError:
            return False


class StateRecoverySystem(BaseModel):
    """
    System for detecting and recovering from inconsistent states.

    This class provides mechanisms to validate state integrity, create backups,
    and recover from various types of state corruption or inconsistency.
    """
    backup_dir: str = "state_backups"
    max_backups: int = 10
    validator: StateValidator = Field(default_factory=StateValidator)

    def __init__(self, **data):
        """Initialize the recovery system and create backup directory."""
        super().__init__(**data)
        os.makedirs(self.backup_dir, exist_ok=True)

    def check_state_integrity(self, state_machine: ConversationStateMachine,
                             level: StateConsistencyLevel = StateConsistencyLevel.STANDARD) -> Tuple[bool, List[str]]:
        """
        Check the integrity of a state machine's state.

        Args:
            state_machine: The conversation state machine to check
            level: The level of consistency checking to perform

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = self.validator.validate_state(state_machine.context, level)
        return len(errors) == 0, errors

    def create_backup(self, state_machine: ConversationStateMachine,
                     metadata: Dict[str, Any] = None) -> StateBackup:
        """
        Create a backup of the current state.

        Args:
            state_machine: The state machine to backup
            metadata: Optional metadata to associate with the backup

        Returns:
            StateBackup object representing the backup
        """
        # Create backup object
        backup = StateBackup(
            conversation_id=state_machine.context.conversation_id,
            state_data=self._serialize_context(state_machine.context),
            metadata=metadata or {}
        )

        # Save to file
        backup_path = os.path.join(self.backup_dir, f"{backup.backup_id}.json")
        with open(backup_path, 'w') as f:
            # Serialize the backup object manually to handle datetime
            backup_dict = backup.model_dump()
            backup_dict["timestamp"] = backup_dict["timestamp"].isoformat()
            json.dump(backup_dict, f, indent=2)

        # Manage backup retention
        self._cleanup_old_backups()

        return backup

    def recover_state(self, state_machine: ConversationStateMachine,
                     strategy: RecoveryStrategy = RecoveryStrategy.REPAIR,
                     error_details: Optional[str] = None) -> StateRecoveryResult:
        """
        Recover from an inconsistent state.

        Args:
            state_machine: The state machine to recover
            strategy: The recovery strategy to use
            error_details: Optional details about the error that triggered recovery

        Returns:
            StateRecoveryResult with details of the recovery operation
        """
        start_time = time.time()
        original_state = state_machine.get_current_state()
        changes = []

        try:
            if strategy == RecoveryStrategy.RESET:
                # Reset to initial state
                state_machine.reset()
                changes.append(f"Reset state machine to initial state: {state_machine.get_current_state()}")

            elif strategy == RecoveryStrategy.ROLLBACK:
                # Rollback to last valid state
                self._rollback_to_last_valid_state(state_machine)
                changes.append(f"Rolled back to last valid state: {state_machine.get_current_state()}")

            elif strategy == RecoveryStrategy.REPAIR:
                # Attempt to repair the state
                self._repair_state(state_machine)
                changes.append(f"Repaired state to: {state_machine.get_current_state()}")

            elif strategy == RecoveryStrategy.USE_BACKUP:
                # Use the latest backup
                self._restore_from_latest_backup(state_machine)
                changes.append(f"Restored from backup to state: {state_machine.get_current_state()}")

            elif strategy == RecoveryStrategy.FORCE_TRANSITION:
                # Force transition to a valid state
                self._force_transition_to_valid_state(state_machine)
                changes.append(f"Forced transition to valid state: {state_machine.get_current_state()}")

            # Verify the recovered state is valid
            is_valid, validation_errors = self.check_state_integrity(state_machine)
            if not is_valid:
                # If still invalid, fall back to reset
                state_machine.reset()
                changes.append(f"Recovery failed, reset to initial state: {state_machine.get_current_state()}")

            return StateRecoveryResult(
                success=True,
                strategy_used=strategy,
                original_state=original_state,
                recovered_state=state_machine.get_current_state(),
                error_details=error_details,
                recovery_time=time.time() - start_time,
                changes_made=changes
            )

        except Exception as e:
            # If recovery fails, reset to initial state as a last resort
            state_machine.reset()

            return StateRecoveryResult(
                success=False,
                strategy_used=strategy,
                original_state=original_state,
                recovered_state=state_machine.get_current_state(),
                error_details=f"Recovery failed: {str(e)}",
                recovery_time=time.time() - start_time,
                changes_made=changes + [f"Recovery exception: {str(e)}",
                                       f"Reset to initial state: {state_machine.get_current_state()}"]
            )

    def list_available_backups(self, conversation_id: Optional[str] = None) -> List[StateBackup]:
        """
        List available backups, optionally filtered by conversation ID.

        Args:
            conversation_id: Optional conversation ID to filter by

        Returns:
            List of StateBackup objects
        """
        backups = []

        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.backup_dir, filename), 'r') as f:
                        backup_data = json.load(f)

                        # Convert ISO format timestamp back to datetime
                        if isinstance(backup_data.get("timestamp"), str):
                            backup_data["timestamp"] = datetime.fromisoformat(backup_data["timestamp"])

                        backup = StateBackup(**backup_data)

                        if conversation_id is None or backup.conversation_id == conversation_id:
                            backups.append(backup)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip invalid backup files
                    continue

        # Sort by timestamp, newest first
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        return backups

    def restore_from_backup(self, state_machine: ConversationStateMachine,
                           backup_id: str) -> bool:
        """
        Restore state from a specific backup.

        Args:
            state_machine: The state machine to restore
            backup_id: ID of the backup to restore from

        Returns:
            True if successful, False otherwise
        """
        backup_path = os.path.join(self.backup_dir, f"{backup_id}.json")

        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
                backup = StateBackup(**backup_data)

                # Restore state
                state_machine.context = self._deserialize_context(backup.state_data)
                return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def _serialize_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Serialize a context object to a dictionary."""
        # Create a deep copy of the context data
        context_dict = context.model_dump()

        # Convert datetime objects to ISO format strings
        context_dict["last_state_change"] = context_dict["last_state_change"].isoformat()

        for i, transition in enumerate(context_dict["state_history"]):
            context_dict["state_history"][i]["timestamp"] = transition["timestamp"].isoformat()

        return context_dict

    def _deserialize_context(self, context_data: Dict[str, Any]) -> ConversationContext:
        """Deserialize a dictionary to a context object."""
        # Convert ISO format strings back to datetime objects
        context_data["last_state_change"] = datetime.fromisoformat(context_data["last_state_change"])

        for i, transition in enumerate(context_data["state_history"]):
            context_data["state_history"][i]["timestamp"] = datetime.fromisoformat(transition["timestamp"])

        return ConversationContext(**context_data)

    def _cleanup_old_backups(self):
        """Remove old backups to stay within max_backups limit."""
        backups = self.list_available_backups()

        if len(backups) > self.max_backups:
            # Remove oldest backups
            for backup in backups[self.max_backups:]:
                try:
                    os.remove(os.path.join(self.backup_dir, f"{backup.backup_id}.json"))
                except FileNotFoundError:
                    pass

    def _rollback_to_last_valid_state(self, state_machine: ConversationStateMachine):
        """Roll back to the last valid state in the history."""
        context = state_machine.context

        if not context.state_history or len(context.state_history) <= 1:
            # Not enough history to rollback, reset instead
            state_machine.reset()
            return

        # Try to find the last valid state
        valid_transitions = []

        for i, transition in enumerate(context.state_history):
            # Add the first transition (from "init") without validation
            if i == 0:
                valid_transitions.append(transition)
                continue

            # For subsequent transitions, validate
            prev_state = valid_transitions[-1].to_state
            if prev_state == transition.from_state and self._is_valid_transition(transition.from_state, transition.to_state):
                valid_transitions.append(transition)
            else:
                # Found an invalid transition, stop here
                break

        if not valid_transitions:
            # No valid transitions found, reset
            state_machine.reset()
            return

        # Rollback to the last valid state
        last_valid = valid_transitions[-1]

        # Create a new context with valid history
        new_context = ConversationContext(
            state=last_valid.to_state,
            state_history=valid_transitions,
            last_state_change=last_valid.timestamp,
            context_data=context.context_data,
            conversation_id=context.conversation_id,
            user_id=context.user_id
        )

        state_machine.context = new_context

    def _repair_state(self, state_machine: ConversationStateMachine):
        """
        Attempt to repair the state by fixing inconsistencies.

        This method tries to preserve as much state as possible while fixing issues.
        """
        context = state_machine.context

        # Check if current state is valid
        valid_states = [state.value for state in ConversationState]
        if context.state not in valid_states:
            # Invalid state, set to a default valid state
            context.state = ConversationState.GREETING.value

        # Ensure state history is consistent
        if context.state_history:
            # Fix the history to ensure transitions are consistent
            fixed_history = [context.state_history[0]]  # Keep the initial state

            for i, transition in enumerate(context.state_history[1:], 1):
                # Ensure from_state matches previous to_state
                corrected_transition = copy.deepcopy(transition)
                corrected_transition.from_state = fixed_history[-1].to_state

                # If to_state is invalid, use a valid next state
                if corrected_transition.to_state not in valid_states:
                    # Find a valid next state
                    try:
                        next_states = state_machine.validator.get_allowed_next_states(corrected_transition.from_state)
                        if next_states:
                            corrected_transition.to_state = next(iter(next_states))
                        else:
                            corrected_transition.to_state = ConversationState.GREETING.value
                    except ValueError:
                        corrected_transition.to_state = ConversationState.GREETING.value

                fixed_history.append(corrected_transition)

            # Update the context with fixed history
            context.state_history = fixed_history

            # Ensure current state matches the last state in history
            context.state = fixed_history[-1].to_state
            context.last_state_change = fixed_history[-1].timestamp

    def _restore_from_latest_backup(self, state_machine: ConversationStateMachine):
        """Restore from the latest backup for this conversation."""
        backups = self.list_available_backups(state_machine.context.conversation_id)

        if not backups:
            # No backups available, reset instead
            state_machine.reset()
            return

        # Use the most recent backup
        latest_backup = backups[0]
        self.restore_from_backup(state_machine, latest_backup.backup_id)

    def _force_transition_to_valid_state(self, state_machine: ConversationStateMachine):
        """Force a transition to a valid state based on the current state."""
        current_state = state_machine.get_current_state()
        valid_states = [state.value for state in ConversationState]

        # If current state is invalid, reset to GREETING
        if current_state not in valid_states:
            state_machine.force_transition(
                ConversationState.GREETING.value,
                reason="Recovery from invalid state"
            )
            return

        # Try to find a valid next state
        try:
            next_states = state_machine.get_allowed_next_states()
            if next_states:
                # Choose the first valid next state
                state_machine.force_transition(
                    next_states[0],
                    reason="Recovery to valid next state"
                )
            else:
                # No valid next states, reset to GREETING
                state_machine.force_transition(
                    ConversationState.GREETING.value,
                    reason="Recovery - no valid next states"
                )
        except Exception:
            # If anything goes wrong, reset to GREETING
            state_machine.force_transition(
                ConversationState.GREETING.value,
                reason="Recovery from exception"
            )

    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a transition is valid."""
        return self.validator.validator.validate_transition(from_state, to_state)


class RecoverableStateMachine(ConversationStateMachine):
    """
    Extension of ConversationStateMachine with built-in recovery capabilities.

    This class adds automatic state validation, backup, and recovery features
    to the base conversation state machine.
    """
    def __init__(self, initial_state: str = ConversationState.GREETING.value,
                validator: Optional[StateTransitionValidator] = None,
                conversation_id: str = "", user_id: Optional[str] = None,
                recovery_system: Optional[StateRecoverySystem] = None,
                auto_backup: bool = True,
                backup_frequency: int = 5,  # Backup every N transitions
                consistency_level: StateConsistencyLevel = StateConsistencyLevel.STANDARD,
                backup_dir: Optional[str] = None):
        """
        Initialize the recoverable state machine.

        Args:
            initial_state: The initial state of the conversation
            validator: Optional custom validator for state transitions
            conversation_id: Optional ID for the conversation
            user_id: Optional ID for the user
            recovery_system: Optional custom recovery system
            auto_backup: Whether to automatically create backups
            backup_frequency: How often to create backups (every N transitions)
            consistency_level: Level of consistency checking to perform
            backup_dir: Optional directory for backups
        """
        super().__init__(initial_state, validator, conversation_id, user_id)

        # Create recovery system if not provided
        if recovery_system is None and backup_dir is not None:
            self.recovery_system = StateRecoverySystem(backup_dir=backup_dir)
        else:
            self.recovery_system = recovery_system or StateRecoverySystem()

        self.auto_backup = auto_backup
        self.backup_frequency = backup_frequency
        self.consistency_level = consistency_level
        self.transition_count = 0

        # Create initial backup
        if self.auto_backup:
            self.recovery_system.create_backup(self, {"type": "initial"})

    def transition_to(self, new_state: str, reason: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Attempt to transition to a new state with validation and recovery.

        Args:
            new_state: The new state to transition to
            reason: Optional reason for the transition
            metadata: Optional metadata to associate with the transition

        Returns:
            bool: True if the transition was successful, False otherwise
        """
        # Validate current state before transition
        is_valid, errors = self.recovery_system.check_state_integrity(self, self.consistency_level)

        if not is_valid:
            # Current state is invalid, attempt recovery
            recovery_result = self.recovery_system.recover_state(
                self,
                strategy=RecoveryStrategy.REPAIR,
                error_details=f"Pre-transition validation failed: {'; '.join(errors)}"
            )

            if not recovery_result.success:
                return False

        # Attempt the transition
        result = super().transition_to(new_state, reason, metadata)

        if result:
            # Successful transition
            self.transition_count += 1

            # Create backup if needed
            if self.auto_backup and self.transition_count % self.backup_frequency == 0:
                self.recovery_system.create_backup(self, {
                    "type": "automatic",
                    "transition_count": self.transition_count
                })

        return result

    def force_transition(self, new_state: str, reason: str = "Forced transition",
                        metadata: Dict[str, Any] = None) -> None:
        """
        Force a transition to a new state, with backup.

        Args:
            new_state: The new state to transition to
            reason: Reason for the forced transition
            metadata: Optional metadata to associate with the transition
        """
        # Create backup before forced transition
        if self.auto_backup:
            self.recovery_system.create_backup(self, {
                "type": "pre_force",
                "forced_to": new_state,
                "reason": reason
            })

        # Perform the forced transition
        super().force_transition(new_state, reason, metadata)

        self.transition_count += 1

    def validate_state(self) -> Tuple[bool, List[str]]:
        """
        Validate the current state.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.recovery_system.check_state_integrity(self, self.consistency_level)

    def create_backup(self, metadata: Dict[str, Any] = None) -> StateBackup:
        """
        Create a manual backup of the current state.

        Args:
            metadata: Optional metadata to associate with the backup

        Returns:
            StateBackup object representing the backup
        """
        return self.recovery_system.create_backup(self, metadata or {"type": "manual"})

    def recover(self, strategy: RecoveryStrategy = RecoveryStrategy.REPAIR) -> StateRecoveryResult:
        """
        Manually trigger state recovery.

        Args:
            strategy: The recovery strategy to use

        Returns:
            StateRecoveryResult with details of the recovery operation
        """
        return self.recovery_system.recover_state(self, strategy)

    def list_backups(self) -> List[StateBackup]:
        """
        List available backups for this conversation.

        Returns:
            List of StateBackup objects
        """
        return self.recovery_system.list_available_backups(self.context.conversation_id)

    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore from a specific backup.

        Args:
            backup_id: ID of the backup to restore from

        Returns:
            True if successful, False otherwise
        """
        return self.recovery_system.restore_from_backup(self, backup_id)


# Example usage
if __name__ == "__main__":
    # Create a recoverable state machine
    state_machine = RecoverableStateMachine(
        conversation_id="demo-recovery",
        auto_backup=True,
        backup_frequency=2  # Backup every 2 transitions
    )

    print(f"Initial state: {state_machine.get_current_state()}")

    # Perform some valid transitions
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value, "User provided query")
    state_machine.transition_to(ConversationState.PROCESSING.value, "Processing query")
    state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value, "Providing results")

    # List backups
    backups = state_machine.list_backups()
    print(f"\nAvailable backups: {len(backups)}")
    for backup in backups:
        print(f"  - {backup.backup_id} ({backup.timestamp})")

    # Simulate state corruption
    print("\nSimulating state corruption...")
    state_machine.context.state = "invalid_state"  # This will corrupt the state

    # Validate the corrupted state
    is_valid, errors = state_machine.validate_state()
    print(f"State valid: {is_valid}")
    if not is_valid:
        print(f"Validation errors: {errors}")

    # Recover from corruption
    print("\nRecovering from corruption...")
    result = state_machine.recover()

    print(f"Recovery successful: {result.success}")
    print(f"Strategy used: {result.strategy_used.value}")
    print(f"Original state: {result.original_state}")
    print(f"Recovered state: {result.recovered_state}")
    print(f"Recovery time: {result.recovery_time:.4f} seconds")
    print(f"Changes made:")
    for change in result.changes_made:
        print(f"  - {change}")

    # Validate the recovered state
    is_valid, errors = state_machine.validate_state()
    print(f"\nRecovered state valid: {is_valid}")

    # Continue with valid transitions
    print("\nContinuing conversation after recovery...")
    state_machine.transition_to(ConversationState.FOLLOW_UP.value, "User asked follow-up")
    state_machine.transition_to(ConversationState.ENDING.value, "User ended conversation")

    print(f"Final state: {state_machine.get_current_state()}")

    # Create a manual backup
    backup = state_machine.create_backup({"type": "final", "note": "End of conversation"})
    print(f"\nCreated final backup: {backup.backup_id}")

    # Demonstrate restoring from a backup
    first_backup = backups[0] if backups else None
    if first_backup:
        print(f"\nRestoring from backup: {first_backup.backup_id}")
        success = state_machine.restore_backup(first_backup.backup_id)
        print(f"Restore successful: {success}")
        print(f"State after restore: {state_machine.get_current_state()}")
