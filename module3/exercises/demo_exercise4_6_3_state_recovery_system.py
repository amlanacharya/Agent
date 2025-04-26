"""
Demo for Exercise 4.6.3: State Recovery System
--------------------------------------------
This script demonstrates the usage of the State Recovery System implementation.
It shows how to detect and recover from various types of state corruption.
"""

import os
import time
import shutil
from datetime import datetime
from colorama import init, Fore, Style

from exercise4_6_2_conversation_state_machine import (
    ConversationState,
    ConversationStateMachine
)

from exercise4_6_3_state_recovery_system import (
    RecoveryStrategy,
    StateConsistencyLevel,
    StateRecoverySystem,
    RecoverableStateMachine
)

# Initialize colorama for colored terminal output
init()


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + f"{Style.RESET_ALL}\n")


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}" + "-" * 60)
    print(f" {text}")
    print("-" * 60 + f"{Style.RESET_ALL}\n")


def print_state_info(state_machine):
    """Print information about the current state."""
    current_state = state_machine.get_current_state()
    
    print(f"{Fore.CYAN}Current State: {Fore.WHITE}{current_state}")
    
    if hasattr(state_machine, "validate_state"):
        is_valid, errors = state_machine.validate_state()
        validity = f"{Fore.GREEN}Valid" if is_valid else f"{Fore.RED}Invalid"
        print(f"{Fore.CYAN}State Validity: {validity}{Style.RESET_ALL}")
        
        if not is_valid:
            print(f"{Fore.RED}Validation Errors:{Style.RESET_ALL}")
            for error in errors:
                print(f"  - {error}")


def print_system_message(text):
    """Print a system message."""
    print(f"{Fore.MAGENTA}System: {Fore.WHITE}{text}{Style.RESET_ALL}")


def print_recovery_result(result):
    """Print details of a recovery operation."""
    success = f"{Fore.GREEN}Success" if result.success else f"{Fore.RED}Failed"
    print(f"{Fore.CYAN}Recovery Result: {success}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Strategy Used: {Fore.WHITE}{result.strategy_used.value}")
    print(f"{Fore.CYAN}Original State: {Fore.WHITE}{result.original_state}")
    print(f"{Fore.CYAN}Recovered State: {Fore.WHITE}{result.recovered_state}")
    print(f"{Fore.CYAN}Recovery Time: {Fore.WHITE}{result.recovery_time:.4f} seconds")
    
    print(f"{Fore.CYAN}Changes Made:{Style.RESET_ALL}")
    for change in result.changes_made:
        print(f"  - {change}")


def print_backup_list(backups):
    """Print a list of available backups."""
    print(f"{Fore.CYAN}Available Backups ({len(backups)}):{Style.RESET_ALL}")
    for i, backup in enumerate(backups):
        metadata_str = ", ".join(f"{k}: {v}" for k, v in backup.metadata.items())
        print(f"  {i+1}. {backup.backup_id} - {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {metadata_str}")


def setup_demo():
    """Set up the demo environment."""
    # Create a clean backup directory
    backup_dir = "demo_backups"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    os.makedirs(backup_dir)
    
    return backup_dir


def demonstrate_basic_recovery():
    """Demonstrate basic state recovery functionality."""
    print_header("Basic State Recovery Demo")
    
    backup_dir = setup_demo()
    recovery_system = StateRecoverySystem(backup_dir=backup_dir)
    
    # Create a regular state machine
    state_machine = ConversationStateMachine(conversation_id="basic-demo")
    
    print_system_message("Created a regular state machine")
    print_state_info(state_machine)
    
    # Perform some transitions
    print_system_message("Performing some transitions...")
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value)
    state_machine.transition_to(ConversationState.PROCESSING.value)
    
    print_state_info(state_machine)
    
    # Create a backup
    print_system_message("Creating a backup...")
    backup = recovery_system.create_backup(state_machine, {"type": "manual"})
    print(f"{Fore.CYAN}Backup Created: {Fore.WHITE}{backup.backup_id}")
    
    # Corrupt the state
    print_system_message("Simulating state corruption...")
    state_machine.context.state = "invalid_state"
    
    # Check state integrity
    is_valid, errors = recovery_system.check_state_integrity(state_machine)
    validity = f"{Fore.GREEN}Valid" if is_valid else f"{Fore.RED}Invalid"
    print(f"{Fore.CYAN}State Validity: {validity}{Style.RESET_ALL}")
    
    if not is_valid:
        print(f"{Fore.RED}Validation Errors:{Style.RESET_ALL}")
        for error in errors:
            print(f"  - {error}")
    
    # Recover using different strategies
    for strategy in [
        RecoveryStrategy.RESET,
        RecoveryStrategy.ROLLBACK,
        RecoveryStrategy.REPAIR,
        RecoveryStrategy.USE_BACKUP,
        RecoveryStrategy.FORCE_TRANSITION
    ]:
        print_subheader(f"Recovery using {strategy.value.upper()} strategy")
        
        # Corrupt the state again
        state_machine.context.state = "invalid_state"
        
        # Recover
        result = recovery_system.recover_state(state_machine, strategy)
        print_recovery_result(result)
        print_state_info(state_machine)


def demonstrate_recoverable_state_machine():
    """Demonstrate the RecoverableStateMachine class."""
    print_header("Recoverable State Machine Demo")
    
    backup_dir = setup_demo()
    
    # Create a recoverable state machine
    state_machine = RecoverableStateMachine(
        conversation_id="recoverable-demo",
        backup_dir=backup_dir,
        auto_backup=True,
        backup_frequency=2,  # Backup every 2 transitions
        consistency_level=StateConsistencyLevel.STANDARD
    )
    
    print_system_message("Created a recoverable state machine with auto-backup")
    print_state_info(state_machine)
    
    # Perform some transitions
    print_system_message("Performing transitions (should create backups automatically)...")
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value, "User provided query")
    print_state_info(state_machine)
    
    state_machine.transition_to(ConversationState.PROCESSING.value, "Processing query")
    print_state_info(state_machine)
    
    state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value, "Providing results")
    print_state_info(state_machine)
    
    # List backups
    backups = state_machine.list_backups()
    print_backup_list(backups)
    
    # Corrupt the state
    print_system_message("Simulating state corruption...")
    state_machine.context.state = "invalid_state"
    print_state_info(state_machine)
    
    # Try to transition (should trigger automatic recovery)
    print_system_message("Attempting transition with corrupted state (should trigger recovery)...")
    result = state_machine.transition_to(ConversationState.FOLLOW_UP.value, "User asked follow-up")
    
    print(f"{Fore.CYAN}Transition Result: {Fore.GREEN if result else Fore.RED}{result}{Style.RESET_ALL}")
    print_state_info(state_machine)
    
    # Create a manual backup
    print_system_message("Creating a manual backup...")
    backup = state_machine.create_backup({"type": "manual", "note": "After recovery"})
    print(f"{Fore.CYAN}Backup Created: {Fore.WHITE}{backup.backup_id}")
    
    # List all backups
    backups = state_machine.list_backups()
    print_backup_list(backups)
    
    # Restore from a specific backup
    if len(backups) > 1:
        print_system_message(f"Restoring from backup: {backups[1].backup_id}...")
        success = state_machine.restore_backup(backups[1].backup_id)
        print(f"{Fore.CYAN}Restore Result: {Fore.GREEN if success else Fore.RED}{success}{Style.RESET_ALL}")
        print_state_info(state_machine)


def demonstrate_complex_recovery_scenario():
    """Demonstrate a more complex recovery scenario."""
    print_header("Complex Recovery Scenario Demo")
    
    backup_dir = setup_demo()
    
    # Create a recoverable state machine
    state_machine = RecoverableStateMachine(
        conversation_id="complex-demo",
        backup_dir=backup_dir,
        auto_backup=True,
        backup_frequency=3,
        consistency_level=StateConsistencyLevel.STRICT
    )
    
    print_system_message("Created a recoverable state machine with strict validation")
    
    # Simulate a conversation flow
    print_system_message("Simulating a conversation flow...")
    
    # GREETING -> COLLECTING_INFO
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value, "User asked about weather")
    state_machine.add_context_data("query_type", "weather")
    
    # COLLECTING_INFO -> COLLECTING_INFO (staying in same state)
    state_machine.transition_to(ConversationState.COLLECTING_INFO.value, "Collecting location")
    state_machine.add_context_data("location", "New York")
    
    # COLLECTING_INFO -> PROCESSING
    state_machine.transition_to(ConversationState.PROCESSING.value, "Processing weather query")
    
    # PROCESSING -> PROVIDING_RESULTS
    state_machine.transition_to(ConversationState.PROVIDING_RESULTS.value, "Providing weather results")
    state_machine.add_context_data("weather_data", {
        "temperature": 72,
        "condition": "Partly Cloudy"
    })
    
    print_state_info(state_machine)
    
    # List backups
    backups = state_machine.list_backups()
    print_backup_list(backups)
    
    # Simulate multiple types of corruption
    print_subheader("Simulating multiple types of corruption")
    
    # 1. Corrupt the state
    state_machine.context.state = "invalid_state"
    
    # 2. Corrupt the state history
    if state_machine.context.state_history:
        state_machine.context.state_history[1].from_state = "another_invalid_state"
    
    # 3. Corrupt timestamps
    state_machine.context.last_state_change = datetime.now().replace(year=2000)
    
    print_state_info(state_machine)
    
    # Manually trigger recovery
    print_system_message("Manually triggering recovery...")
    result = state_machine.recover(RecoveryStrategy.REPAIR)
    print_recovery_result(result)
    print_state_info(state_machine)
    
    # Continue the conversation after recovery
    print_system_message("Continuing conversation after recovery...")
    state_machine.transition_to(ConversationState.FOLLOW_UP.value, "User asked follow-up")
    state_machine.transition_to(ConversationState.ENDING.value, "User ended conversation")
    
    print_state_info(state_machine)


if __name__ == "__main__":
    try:
        # Run the basic recovery demo
        demonstrate_basic_recovery()
        
        # Wait a moment before the next demo
        time.sleep(1)
        
        # Run the recoverable state machine demo
        demonstrate_recoverable_state_machine()
        
        # Wait a moment before the next demo
        time.sleep(1)
        
        # Run the complex recovery scenario demo
        demonstrate_complex_recovery_scenario()
        
        print_header("Demo Complete")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Demo interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during demo: {e}{Style.RESET_ALL}")
