"""
Tests for Exercise 4.6.3: State Recovery System
---------------------------------------------
This module contains tests for the state recovery system implementation.
"""

import unittest
import os
import shutil
import time
import json
from datetime import datetime, timedelta
from pydantic import ValidationError

from exercise4_6_2_conversation_state_machine import (
    ConversationState,
    StateTransitionValidator,
    StateTransition,
    ConversationContext,
    ConversationStateMachine
)

from exercise4_6_3_state_recovery_system import (
    StateIntegrityError,
    RecoveryStrategy,
    StateConsistencyLevel,
    StateBackup,
    StateRecoveryResult,
    StateValidator,
    StateRecoverySystem,
    RecoverableStateMachine
)


class TestStateRecoverySystem(unittest.TestCase):
    """Test cases for the state recovery system implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary backup directory for testing
        self.test_backup_dir = "test_backups"
        os.makedirs(self.test_backup_dir, exist_ok=True)

        # Create a recovery system
        self.recovery_system = StateRecoverySystem(backup_dir=self.test_backup_dir)

        # Create a state machine
        self.state_machine = ConversationStateMachine(
            conversation_id="test-conversation",
            user_id="test-user"
        )

        # Create a recoverable state machine
        self.recoverable_machine = RecoverableStateMachine(
            conversation_id="test-recoverable",
            user_id="test-user",
            recovery_system=self.recovery_system,
            auto_backup=True,
            backup_frequency=2
        )

    def tearDown(self):
        """Clean up after tests."""
        # Remove the test backup directory
        if os.path.exists(self.test_backup_dir):
            shutil.rmtree(self.test_backup_dir)

    def test_state_validator_basic(self):
        """Test basic state validation."""
        validator = StateValidator()
        context = ConversationContext()

        # Valid state
        errors = validator.validate_state(context, StateConsistencyLevel.BASIC)
        self.assertEqual(len(errors), 0)

        # Invalid state
        context.state = "invalid_state"
        errors = validator.validate_state(context, StateConsistencyLevel.BASIC)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Invalid state" in error for error in errors))

    def test_state_validator_standard(self):
        """Test standard state validation."""
        validator = StateValidator()
        context = ConversationContext()

        # Perform a valid transition
        transition_validator = StateTransitionValidator()
        context.transition_to(ConversationState.COLLECTING_INFO.value, transition_validator)

        # Valid state
        errors = validator.validate_state(context, StateConsistencyLevel.STANDARD)
        self.assertEqual(len(errors), 0)

        # Create inconsistency in state history
        context.state_history[1].from_state = "invalid_state"
        errors = validator.validate_state(context, StateConsistencyLevel.STANDARD)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Inconsistent state history" in error for error in errors))

    def test_state_validator_strict(self):
        """Test strict state validation."""
        validator = StateValidator()
        context = ConversationContext()

        # Perform a valid transition
        transition_validator = StateTransitionValidator()
        context.transition_to(ConversationState.COLLECTING_INFO.value, transition_validator)

        # Valid state
        errors = validator.validate_state(context, StateConsistencyLevel.STRICT)
        self.assertEqual(len(errors), 0)

        # Create timestamp inconsistency
        context.last_state_change = datetime.now() - timedelta(hours=1)
        errors = validator.validate_state(context, StateConsistencyLevel.STRICT)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("last_state_change" in error for error in errors))

    def test_create_backup(self):
        """Test creating a state backup."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.transition_to(ConversationState.PROCESSING.value)

        # Create a backup
        backup = self.recovery_system.create_backup(self.state_machine, {"test": True})

        # Check backup was created
        self.assertTrue(os.path.exists(os.path.join(self.test_backup_dir, f"{backup.backup_id}.json")))

        # Check backup properties
        self.assertEqual(backup.conversation_id, "test-conversation")
        self.assertEqual(backup.metadata["test"], True)

    def test_list_backups(self):
        """Test listing available backups."""
        # Create a separate test directory for this test
        test_dir = os.path.join(self.test_backup_dir, "list_backups_test")
        os.makedirs(test_dir, exist_ok=True)

        # Clear any existing backups
        for file in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, file))

        # Create a recovery system specific to this test
        recovery_system = StateRecoverySystem(backup_dir=test_dir)

        # Create some backups
        recovery_system.create_backup(self.state_machine, {"index": 1})
        time.sleep(0.1)  # Ensure different timestamps
        recovery_system.create_backup(self.state_machine, {"index": 2})
        time.sleep(0.1)  # Ensure different timestamps
        recovery_system.create_backup(self.state_machine, {"index": 3})

        # List all backups
        backups = recovery_system.list_available_backups()
        self.assertEqual(len(backups), 3)

        # List backups for specific conversation
        backups = recovery_system.list_available_backups("test-conversation")
        self.assertEqual(len(backups), 3)

        # List backups for non-existent conversation
        backups = recovery_system.list_available_backups("non-existent")
        self.assertEqual(len(backups), 0)

        # Clean up
        shutil.rmtree(test_dir)

    def test_restore_from_backup(self):
        """Test restoring from a backup."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Create a backup
        backup = self.recovery_system.create_backup(self.state_machine)

        # Perform more transitions
        self.state_machine.transition_to(ConversationState.PROCESSING.value)
        self.state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value)

        # Restore from backup
        success = self.recovery_system.restore_from_backup(self.state_machine, backup.backup_id)

        # Check restore was successful
        self.assertTrue(success)
        self.assertEqual(self.state_machine.get_current_state(), ConversationState.COLLECTING_INFO.value)

    def test_recover_state_reset(self):
        """Test recovering state using RESET strategy."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.transition_to(ConversationState.PROCESSING.value)

        # Corrupt the state
        self.state_machine.context.state = "invalid_state"

        # Recover using RESET strategy
        result = self.recovery_system.recover_state(
            self.state_machine,
            strategy=RecoveryStrategy.RESET
        )

        # Check recovery result
        self.assertTrue(result.success)
        self.assertEqual(result.strategy_used, RecoveryStrategy.RESET)
        self.assertEqual(result.original_state, "invalid_state")
        self.assertEqual(result.recovered_state, ConversationState.GREETING.value)

    def test_recover_state_rollback(self):
        """Test recovering state using ROLLBACK strategy."""
        # Perform some valid transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.transition_to(ConversationState.PROCESSING.value)

        # Force an invalid transition (corrupting the state)
        self.state_machine.force_transition(ConversationState.FOLLOW_UP.value)

        # Recover using ROLLBACK strategy
        result = self.recovery_system.recover_state(
            self.state_machine,
            strategy=RecoveryStrategy.ROLLBACK
        )

        # Check recovery result
        self.assertTrue(result.success)
        self.assertEqual(result.strategy_used, RecoveryStrategy.ROLLBACK)
        self.assertEqual(result.original_state, ConversationState.FOLLOW_UP.value)
        # Should rollback to PROCESSING which was the last valid state
        self.assertEqual(result.recovered_state, ConversationState.PROCESSING.value)

    def test_recover_state_repair(self):
        """Test recovering state using REPAIR strategy."""
        # Create a fresh state machine
        state_machine = ConversationStateMachine(
            conversation_id="test-repair",
            user_id="test-user"
        )

        # Perform some transitions
        state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Corrupt the state history but keep current state valid
        state_machine.context.state_history[0].to_state = "invalid_state"

        # Recover using REPAIR strategy
        result = self.recovery_system.recover_state(
            state_machine,
            strategy=RecoveryStrategy.REPAIR
        )

        # Check recovery result
        self.assertTrue(result.success)
        self.assertEqual(result.strategy_used, RecoveryStrategy.REPAIR)

        # The state should still be COLLECTING_INFO or a valid state
        valid_states = [state.value for state in ConversationState]
        self.assertIn(result.recovered_state, valid_states)

        # Validate the repaired state
        is_valid, _ = self.recovery_system.check_state_integrity(state_machine)
        self.assertTrue(is_valid)

    def test_recover_state_use_backup(self):
        """Test recovering state using USE_BACKUP strategy."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Create a backup
        self.recovery_system.create_backup(self.state_machine)

        # Perform more transitions and corrupt the state
        self.state_machine.transition_to(ConversationState.PROCESSING.value)
        self.state_machine.context.state = "invalid_state"

        # Recover using USE_BACKUP strategy
        result = self.recovery_system.recover_state(
            self.state_machine,
            strategy=RecoveryStrategy.USE_BACKUP
        )

        # Check recovery result
        self.assertTrue(result.success)
        self.assertEqual(result.strategy_used, RecoveryStrategy.USE_BACKUP)
        self.assertEqual(result.original_state, "invalid_state")
        self.assertEqual(result.recovered_state, ConversationState.COLLECTING_INFO.value)

    def test_recover_state_force_transition(self):
        """Test recovering state using FORCE_TRANSITION strategy."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Corrupt the state
        self.state_machine.context.state = "invalid_state"

        # Recover using FORCE_TRANSITION strategy
        result = self.recovery_system.recover_state(
            self.state_machine,
            strategy=RecoveryStrategy.FORCE_TRANSITION
        )

        # Check recovery result
        self.assertTrue(result.success)
        self.assertEqual(result.strategy_used, RecoveryStrategy.FORCE_TRANSITION)
        self.assertEqual(result.original_state, "invalid_state")
        self.assertEqual(result.recovered_state, ConversationState.GREETING.value)

    def test_recoverable_state_machine(self):
        """Test the RecoverableStateMachine class."""
        # Clear any existing backups
        for file in os.listdir(self.test_backup_dir):
            os.remove(os.path.join(self.test_backup_dir, file))

        # Create a new recoverable machine with a clean state
        recoverable_machine = RecoverableStateMachine(
            conversation_id="test-recoverable-new",
            user_id="test-user",
            backup_dir=self.test_backup_dir,
            auto_backup=True,
            backup_frequency=1  # Backup every transition for testing
        )

        # Perform some transitions
        recoverable_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        recoverable_machine.transition_to(ConversationState.PROCESSING.value)

        # Check that backups were created
        backups = recoverable_machine.list_backups()
        self.assertGreaterEqual(len(backups), 2)  # Initial + at least one auto backup

        # Corrupt the state
        recoverable_machine.context.state = "invalid_state"

        # Attempt a transition, which should trigger recovery
        result = recoverable_machine.transition_to(ConversationState.PROVIDING_RESULTS.value)

        # Check that the transition succeeded after recovery
        self.assertTrue(result)
        self.assertEqual(recoverable_machine.get_current_state(), ConversationState.PROVIDING_RESULTS.value)

    def test_manual_backup_and_restore(self):
        """Test manual backup and restore operations."""
        # Clear any existing backups
        for file in os.listdir(self.test_backup_dir):
            os.remove(os.path.join(self.test_backup_dir, file))

        # Create a new recoverable machine with a clean state
        recoverable_machine = RecoverableStateMachine(
            conversation_id="test-manual-backup",
            user_id="test-user",
            backup_dir=self.test_backup_dir,
            auto_backup=False  # Disable auto backup for this test
        )

        # Perform some transitions
        recoverable_machine.transition_to(ConversationState.COLLECTING_INFO.value)

        # Create a manual backup
        backup = recoverable_machine.create_backup({"type": "manual_test"})

        # Perform more transitions
        recoverable_machine.transition_to(ConversationState.PROCESSING.value)
        recoverable_machine.transition_to(ConversationState.PROVIDING_RESULTS.value)

        # Restore from the manual backup
        success = recoverable_machine.restore_backup(backup.backup_id)

        # Check restore was successful
        self.assertTrue(success)
        self.assertEqual(recoverable_machine.get_current_state(), ConversationState.COLLECTING_INFO.value)

    def test_manual_recovery(self):
        """Test manual recovery operation."""
        # Perform some transitions
        self.recoverable_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.recoverable_machine.transition_to(ConversationState.PROCESSING.value)

        # Corrupt the state
        self.recoverable_machine.context.state = "invalid_state"

        # Manually trigger recovery
        result = self.recoverable_machine.recover(RecoveryStrategy.REPAIR)

        # Check recovery result
        self.assertTrue(result.success)
        self.assertNotEqual(result.recovered_state, "invalid_state")

    def test_backup_cleanup(self):
        """Test cleanup of old backups."""
        # Create a separate test directory for this test
        test_dir = os.path.join(self.test_backup_dir, "backup_cleanup_test")
        os.makedirs(test_dir, exist_ok=True)

        # Clear any existing backups
        for file in os.listdir(test_dir):
            os.remove(os.path.join(test_dir, file))

        # Create a recovery system specific to this test with a low max_backups value
        recovery_system = StateRecoverySystem(backup_dir=test_dir, max_backups=3)

        # Create more than max_backups backups
        for i in range(5):
            recovery_system.create_backup(self.state_machine, {"index": i})
            time.sleep(0.1)  # Ensure different timestamps

        # Check that only max_backups backups are kept
        backups = recovery_system.list_available_backups()
        self.assertEqual(len(backups), 3)

        # Check that the newest backups are kept
        indices = [b.metadata["index"] for b in backups]
        self.assertEqual(sorted(indices, reverse=True), [4, 3, 2])

        # Clean up
        shutil.rmtree(test_dir)

    def test_serialization_deserialization(self):
        """Test serialization and deserialization of context."""
        # Perform some transitions
        self.state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
        self.state_machine.add_context_data("test_key", "test_value")

        # Serialize
        serialized = self.recovery_system._serialize_context(self.state_machine.context)

        # Deserialize
        deserialized = self.recovery_system._deserialize_context(serialized)

        # Check that deserialized context matches original
        self.assertEqual(deserialized.state, self.state_machine.context.state)
        self.assertEqual(deserialized.context_data, self.state_machine.context.context_data)
        self.assertEqual(len(deserialized.state_history), len(self.state_machine.context.state_history))


if __name__ == "__main__":
    unittest.main()
