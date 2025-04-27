"""
Demo script for Exercise 4.6.5: Required Information Validator

This script demonstrates how to use the RequiredInfoValidator class to create
and manage task states that ensure all required information is collected.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from colorama import init, Fore, Style

# Initialize colorama
init()

# Import the RequiredInfoValidator module
from exercise4_6_5_required_info_validator import (
    FieldRequirement, FieldPriority, FieldDefinition, TaskDefinition,
    ValidationResult, FieldStatus, TaskState, RequiredInfoValidator,
    create_weather_query_task, create_booking_query_task, create_product_query_task
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


def print_field_status(field_name: str, status: FieldStatus) -> None:
    """Print field status with formatting."""
    if status.is_filled:
        if status.is_valid:
            print(f"{Fore.GREEN}✓ {field_name}: {status.value}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ {field_name}: {status.value} - {status.error_message}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}□ {field_name}: Not provided{Style.RESET_ALL}")


def print_task_state(task_state: TaskState, validator: RequiredInfoValidator) -> None:
    """Print task state with formatting."""
    # Get task definition
    task_def = validator.task_definitions.get(task_state.task_type)
    if not task_def:
        print(f"{Fore.RED}Unknown task type: {task_state.task_type}{Style.RESET_ALL}")
        return
    
    print(f"Task Type: {Fore.CYAN}{task_state.task_type}{Style.RESET_ALL}")
    print(f"Description: {task_def.description}")
    print(f"Created: {task_state.created_at}")
    print(f"Last Updated: {task_state.last_updated}")
    print(f"Complete: {Fore.GREEN if task_state.is_complete else Fore.RED}{task_state.is_complete}{Style.RESET_ALL}")
    
    print("\nField Status:")
    
    # Print required fields
    if task_def.get_required_fields():
        print(f"\n{Fore.MAGENTA}Required Fields:{Style.RESET_ALL}")
        for field in task_def.get_required_fields():
            status = task_state.field_status.get(field.name, FieldStatus(field_name=field.name))
            print_field_status(field.name, status)
    
    # Print conditional fields
    conditional_fields = task_def.get_conditional_fields()
    if conditional_fields:
        print(f"\n{Fore.MAGENTA}Conditional Fields:{Style.RESET_ALL}")
        for field in conditional_fields:
            # Check if dependencies are met
            dependencies_met = True
            if field.depends_on:
                for dep_field, dep_value in field.depends_on.items():
                    if task_state.get_field_value(dep_field) != dep_value:
                        dependencies_met = False
                        break
            
            status = task_state.field_status.get(field.name, FieldStatus(field_name=field.name))
            
            # Print dependency info
            if field.depends_on:
                dep_info = ", ".join(f"{k}={v}" for k, v in field.depends_on.items())
                print(f"{Fore.BLUE}[Depends on: {dep_info}] {Fore.GREEN if dependencies_met else Fore.RED}{'(Active)' if dependencies_met else '(Inactive)'}{Style.RESET_ALL}")
            
            print_field_status(field.name, status)
    
    # Print optional fields
    if task_def.get_optional_fields():
        print(f"\n{Fore.MAGENTA}Optional Fields:{Style.RESET_ALL}")
        for field in task_def.get_optional_fields():
            status = task_state.field_status.get(field.name, FieldStatus(field_name=field.name))
            print_field_status(field.name, status)


def print_completeness_info(completeness: Dict[str, Any]) -> None:
    """Print completeness information with formatting."""
    print(f"\nCompleteness Status:")
    print(f"Is Complete: {Fore.GREEN if completeness['is_complete'] else Fore.RED}{completeness['is_complete']}{Style.RESET_ALL}")
    
    if completeness["missing_required"]:
        print(f"\n{Fore.RED}Missing Required Fields:{Style.RESET_ALL}")
        for field in completeness["missing_required"]:
            print(f"  - {field}")
    
    if completeness["missing_conditional"]:
        print(f"\n{Fore.YELLOW}Missing Conditional Fields:{Style.RESET_ALL}")
        for field in completeness["missing_conditional"]:
            print(f"  - {field}")
    
    if completeness["invalid_fields"]:
        print(f"\n{Fore.RED}Invalid Fields:{Style.RESET_ALL}")
        for field in completeness["invalid_fields"]:
            print(f"  - {field}")
    
    if completeness["next_field"]:
        print(f"\nNext Field to Request: {Fore.CYAN}{completeness['next_field']}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}All required information collected!{Style.RESET_ALL}")


def demo_weather_query() -> None:
    """Demonstrate the weather query task."""
    print_header("Weather Query Task Demo")
    
    # Create validator and register task definition
    validator = RequiredInfoValidator()
    validator.register_task_definition(create_weather_query_task())
    
    # Create task state
    task_state = validator.create_task_state("weather_query")
    
    # Print initial state
    print_subheader("Initial State")
    print_task_state(task_state, validator)
    
    # Check completeness
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Get prompt for next field
    next_field = completeness["next_field"]
    if next_field:
        prompt = validator.get_field_prompt(task_state, next_field)
        print(f"\nPrompt: {Fore.CYAN}{prompt}{Style.RESET_ALL}")
    
    # Update location field
    print_subheader("After Providing Location")
    validator.update_field(task_state, "location", "New York")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Add optional field
    print_subheader("After Providing Optional Date")
    validator.update_field(task_state, "date", "tomorrow")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)


def demo_booking_query() -> None:
    """Demonstrate the booking query task."""
    print_header("Booking Query Task Demo")
    
    # Create validator and register task definition
    validator = RequiredInfoValidator()
    validator.register_task_definition(create_booking_query_task())
    
    # Create task state
    task_state = validator.create_task_state("booking_query")
    
    # Print initial state
    print_subheader("Initial State")
    print_task_state(task_state, validator)
    
    # Check completeness
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Update service_type field
    print_subheader("After Providing Service Type (Haircut)")
    validator.update_field(task_state, "service_type", "haircut")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Update date field
    print_subheader("After Providing Date")
    validator.update_field(task_state, "date", "tomorrow")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Get prompt for next field
    next_field = completeness["next_field"]
    if next_field:
        prompt = validator.get_field_prompt(task_state, next_field)
        print(f"\nPrompt: {Fore.CYAN}{prompt}{Style.RESET_ALL}")
    
    # Update time field
    print_subheader("After Providing Time")
    validator.update_field(task_state, "time", "3:00 PM")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Update stylist_preference field
    print_subheader("After Providing Stylist Preference")
    validator.update_field(task_state, "stylist_preference", "John")
    
    # Print updated state
    print_task_state(task_state, validator)
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)


def demo_conditional_fields() -> None:
    """Demonstrate conditional fields behavior."""
    print_header("Conditional Fields Demo")
    
    # Create validator and register task definition
    validator = RequiredInfoValidator()
    validator.register_task_definition(create_booking_query_task())
    
    # Create task state
    task_state = validator.create_task_state("booking_query")
    
    # Update required fields
    validator.update_field(task_state, "date", "tomorrow")
    validator.update_field(task_state, "time", "3:00 PM")
    
    # First scenario: haircut
    print_subheader("Scenario 1: Service Type = Haircut")
    validator.update_field(task_state, "service_type", "haircut")
    
    # Print state
    print_task_state(task_state, validator)
    
    # Check completeness
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Second scenario: massage
    print_subheader("Scenario 2: Service Type = Massage")
    validator.update_field(task_state, "service_type", "massage")
    
    # Print state
    print_task_state(task_state, validator)
    
    # Check completeness
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)
    
    # Update massage_type
    validator.update_field(task_state, "massage_type", "deep tissue")
    
    # Print state
    print_task_state(task_state, validator)
    
    # Check completeness
    completeness = validator.check_completeness(task_state)
    print_completeness_info(completeness)


def demo_validation_errors() -> None:
    """Demonstrate validation errors."""
    print_header("Validation Errors Demo")
    
    # Create validator and register task definition
    validator = RequiredInfoValidator()
    validator.register_task_definition(create_booking_query_task())
    
    # Create task state
    task_state = validator.create_task_state("booking_query")
    
    # Try to update with invalid value
    print_subheader("Invalid Service Type")
    result = validator.update_field(task_state, "service_type", "invalid_service")
    
    print(f"Validation Result:")
    print(f"Is Valid: {Fore.GREEN if result.is_valid else Fore.RED}{result.is_valid}{Style.RESET_ALL}")
    if result.error_message:
        print(f"Error Message: {Fore.RED}{result.error_message}{Style.RESET_ALL}")
    
    # Print state
    print_task_state(task_state, validator)
    
    # Get prompt with error
    prompt = validator.get_field_prompt(task_state, "service_type")
    print(f"\nPrompt with Error: {Fore.CYAN}{prompt}{Style.RESET_ALL}")
    
    # Update with valid value
    print_subheader("Valid Service Type")
    result = validator.update_field(task_state, "service_type", "haircut")
    
    print(f"Validation Result:")
    print(f"Is Valid: {Fore.GREEN if result.is_valid else Fore.RED}{result.is_valid}{Style.RESET_ALL}")
    
    # Print state
    print_task_state(task_state, validator)


def main() -> None:
    """Main function to run the demo."""
    print_header("Required Information Validator Demo")
    
    print(f"""
This demo showcases the RequiredInfoValidator, which ensures all required information
is collected before completing a task. The validator supports:

1. Required, optional, and conditional fields
2. Field validation with various constraints
3. Prioritization of missing fields
4. Tracking of field status and task completeness
5. Generation of prompts for requesting missing information

The demo includes the following scenarios:
""")
    
    print(f"1. {Fore.CYAN}Weather Query Task{Style.RESET_ALL} - Simple task with one required field")
    print(f"2. {Fore.CYAN}Booking Query Task{Style.RESET_ALL} - Complex task with required and conditional fields")
    print(f"3. {Fore.CYAN}Conditional Fields{Style.RESET_ALL} - Demonstration of how conditional fields work")
    print(f"4. {Fore.CYAN}Validation Errors{Style.RESET_ALL} - Handling of validation errors")
    
    while True:
        print(f"\n{Fore.YELLOW}Choose a demo to run (1-4, or q to quit):{Style.RESET_ALL}")
        choice = input("> ").strip().lower()
        
        if choice == "q":
            break
        elif choice == "1":
            demo_weather_query()
        elif choice == "2":
            demo_booking_query()
        elif choice == "3":
            demo_conditional_fields()
        elif choice == "4":
            demo_validation_errors()
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-4 or q.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
