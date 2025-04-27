"""
Demo script for Exercise 4.7.2: Task-Oriented Agent Validation
-----------------------------------------------------------
This script demonstrates the usage of the task-oriented agent validation patterns
through simulated task executions with various validation scenarios.
"""

import sys
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
from colorama import init, Fore, Style

# Initialize colorama
init()

# Import the task agent validator module
from exercise4_7_2_task_agent_validator import (
    ExecutionStatus, ErrorSeverity, TaskError, TaskParameters,
    CalendarEventParameters, ReminderParameters, EmailParameters,
    SearchParameters, WeatherParameters, TaskPrecondition,
    AuthenticationPrecondition, PermissionPrecondition,
    ResourceAvailabilityPrecondition, ConnectivityPrecondition,
    TaskResult, TaskAgent, CalendarEventAgent, ReminderAgent,
    EmailAgent, TaskAgentFactory, TaskAgentValidator
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


def print_task_info(task_agent: TaskAgent) -> None:
    """Print task agent information with formatting."""
    print(f"\n{Fore.MAGENTA}Task Information:{Style.RESET_ALL}")
    print(f"  Task ID: {task_agent.task_id}")
    print(f"  Task Type: {task_agent.task_type}")
    print(f"  Status: {Fore.CYAN}{task_agent.execution_status.value}{Style.RESET_ALL}")
    print(f"  Created: {task_agent.created_at}")
    print(f"  Updated: {task_agent.updated_at}")
    print(f"  Preconditions Met: {Fore.GREEN if task_agent.preconditions_met else Fore.RED}{task_agent.preconditions_met}{Style.RESET_ALL}")
    
    if hasattr(task_agent.parameters, "title"):
        print(f"  Title: {task_agent.parameters.title}")
    
    if hasattr(task_agent.parameters, "start_time") and hasattr(task_agent.parameters, "end_time"):
        print(f"  Time: {task_agent.parameters.start_time.strftime('%Y-%m-%d %H:%M')} to {task_agent.parameters.end_time.strftime('%H:%M')}")
    
    if hasattr(task_agent.parameters, "due_date"):
        print(f"  Due Date: {task_agent.parameters.due_date.strftime('%Y-%m-%d %H:%M')}")
    
    if hasattr(task_agent.parameters, "recipients"):
        print(f"  Recipients: {', '.join(task_agent.parameters.recipients)}")
    
    if hasattr(task_agent.parameters, "subject"):
        print(f"  Subject: {task_agent.parameters.subject}")


def print_preconditions(task_agent: TaskAgent) -> None:
    """Print task preconditions with formatting."""
    if not task_agent.preconditions:
        print(f"\n{Fore.MAGENTA}Preconditions: {Fore.YELLOW}None{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.MAGENTA}Preconditions:{Style.RESET_ALL}")
    for i, precondition in enumerate(task_agent.preconditions):
        status = f"{Fore.GREEN}Met{Style.RESET_ALL}" if precondition.is_met else f"{Fore.RED}Not Met{Style.RESET_ALL}"
        print(f"  {i+1}. {precondition.name}: {status}")
        print(f"     Description: {precondition.description}")
        
        if hasattr(precondition, "required_permissions") and precondition.required_permissions:
            print(f"     Required Permissions: {', '.join(precondition.required_permissions)}")
        
        if hasattr(precondition, "required_resources") and precondition.required_resources:
            print(f"     Required Resources: {', '.join(precondition.required_resources)}")
        
        if precondition.error_message:
            print(f"     Error: {Fore.RED}{precondition.error_message}{Style.RESET_ALL}")


def print_errors(errors: List[TaskError]) -> None:
    """Print task errors with formatting."""
    if not errors:
        print(f"\n{Fore.MAGENTA}Errors: {Fore.GREEN}None{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.MAGENTA}Errors:{Style.RESET_ALL}")
    for i, error in enumerate(errors):
        if error.severity == ErrorSeverity.INFO:
            color = Fore.BLUE
        elif error.severity == ErrorSeverity.WARNING:
            color = Fore.YELLOW
        elif error.severity == ErrorSeverity.ERROR:
            color = Fore.RED
        else:  # CRITICAL
            color = Fore.RED + Style.BRIGHT
        
        print(f"  {i+1}. [{color}{error.severity.value}{Style.RESET_ALL}] {error.error_code}: {error.message}")
        if error.details:
            print(f"     Details: {error.details}")


def print_result(result: TaskResult) -> None:
    """Print task result with formatting."""
    print(f"\n{Fore.MAGENTA}Task Result:{Style.RESET_ALL}")
    status = f"{Fore.GREEN}Success{Style.RESET_ALL}" if result.success else f"{Fore.RED}Failure{Style.RESET_ALL}"
    print(f"  Status: {status}")
    print(f"  Message: {result.message}")
    
    if result.data:
        print(f"  Data:")
        for key, value in result.data.items():
            print(f"    {key}: {value}")
    
    print_errors(result.errors)


def print_validation_errors(errors: List[str]) -> None:
    """Print validation errors with formatting."""
    if not errors:
        print(f"{Fore.GREEN}No validation errors{Style.RESET_ALL}")
        return
    
    print(f"{Fore.RED}Validation errors:{Style.RESET_ALL}")
    for i, error in enumerate(errors):
        print(f"  {i+1}. {error}")


def demo_parameter_validation() -> None:
    """Demonstrate parameter validation."""
    print_header("Parameter Validation Demo")
    
    validator = TaskAgentValidator()
    
    print_subheader("Valid Calendar Event Parameters")
    
    calendar_parameters = {
        "title": "Team Meeting",
        "start_time": datetime(2023, 12, 25, 10, 0),
        "end_time": datetime(2023, 12, 25, 11, 0),
        "attendees": ["alice@example.com", "bob@example.com"],
        "location": "Conference Room A"
    }
    
    print(f"Parameters:")
    for key, value in calendar_parameters.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_parameters("calendar_event", calendar_parameters)
    print("\nValidation result:")
    print_validation_errors(errors)
    
    print_subheader("Invalid Calendar Event Parameters (End Time Before Start Time)")
    
    invalid_calendar_parameters = {
        "title": "Invalid Meeting",
        "start_time": datetime(2023, 12, 25, 11, 0),
        "end_time": datetime(2023, 12, 25, 10, 0)
    }
    
    print(f"Parameters:")
    for key, value in invalid_calendar_parameters.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_parameters("calendar_event", invalid_calendar_parameters)
    print("\nValidation result:")
    print_validation_errors(errors)
    
    print_subheader("Valid Email Parameters")
    
    email_parameters = {
        "recipients": ["alice@example.com", "bob@example.com"],
        "subject": "Meeting Agenda",
        "body": "Here's the agenda for our meeting."
    }
    
    print(f"Parameters:")
    for key, value in email_parameters.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_parameters("email", email_parameters)
    print("\nValidation result:")
    print_validation_errors(errors)
    
    print_subheader("Invalid Email Parameters (Invalid Email Address)")
    
    invalid_email_parameters = {
        "recipients": ["alice@example.com", "invalid-email"],
        "subject": "Meeting Agenda",
        "body": "Here's the agenda for our meeting."
    }
    
    print(f"Parameters:")
    for key, value in invalid_email_parameters.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_parameters("email", invalid_email_parameters)
    print("\nValidation result:")
    print_validation_errors(errors)


def demo_precondition_validation() -> None:
    """Demonstrate precondition validation."""
    print_header("Precondition Validation Demo")
    
    validator = TaskAgentValidator()
    
    print_subheader("Valid Preconditions")
    
    preconditions = [
        {
            "type": "authentication",
            "name": "user_auth",
            "description": "User must be authenticated"
        },
        {
            "type": "permission",
            "name": "calendar_permission",
            "description": "User must have calendar write permission",
            "required_permissions": ["calendar.write"]
        },
        {
            "type": "resource_availability",
            "name": "calendar_api",
            "description": "Calendar API must be available",
            "required_resources": ["calendar_api"]
        }
    ]
    
    print(f"Preconditions:")
    for i, precondition in enumerate(preconditions):
        print(f"  {i+1}. Type: {precondition['type']}, Name: {precondition['name']}")
        if "required_permissions" in precondition:
            print(f"     Required Permissions: {', '.join(precondition['required_permissions'])}")
        if "required_resources" in precondition:
            print(f"     Required Resources: {', '.join(precondition['required_resources'])}")
    
    errors = validator.validate_preconditions(preconditions)
    print("\nValidation result:")
    print_validation_errors(errors)
    
    print_subheader("Invalid Preconditions (Missing Required Fields)")
    
    invalid_preconditions = [
        {
            "type": "permission",
            "name": "calendar_permission",
            "description": "User must have calendar write permission"
            # Missing required_permissions
        },
        {
            "type": "resource_availability",
            "name": "calendar_api",
            "description": "Calendar API must be available"
            # Missing required_resources
        }
    ]
    
    print(f"Preconditions:")
    for i, precondition in enumerate(invalid_preconditions):
        print(f"  {i+1}. Type: {precondition['type']}, Name: {precondition['name']}")
    
    errors = validator.validate_preconditions(invalid_preconditions)
    print("\nValidation result:")
    print_validation_errors(errors)


def demo_execution_context_validation() -> None:
    """Demonstrate execution context validation."""
    print_header("Execution Context Validation Demo")
    
    validator = TaskAgentValidator()
    
    # Create a task agent with various preconditions
    task_agent = TaskAgent(
        task_type="calendar_event",
        parameters={
            "title": "Team Meeting",
            "start_time": datetime(2023, 12, 25, 10, 0),
            "end_time": datetime(2023, 12, 25, 11, 0)
        },
        preconditions=[
            AuthenticationPrecondition(),
            PermissionPrecondition(required_permissions=["calendar.write"]),
            ResourceAvailabilityPrecondition(required_resources=["calendar_api"]),
            ConnectivityPrecondition()
        ]
    )
    
    print_task_info(task_agent)
    print_preconditions(task_agent)
    
    print_subheader("Valid Execution Context")
    
    valid_context = {
        "is_authenticated": True,
        "user_permissions": ["calendar.write", "calendar.read"],
        "available_resources": ["calendar_api", "notification_service"],
        "is_connected": True
    }
    
    print(f"Execution Context:")
    for key, value in valid_context.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_execution_context(task_agent, valid_context)
    print("\nValidation result:")
    print_validation_errors(errors)
    
    print_subheader("Invalid Execution Context (Missing Required Fields)")
    
    invalid_context = {
        "is_authenticated": True
        # Missing other required fields
    }
    
    print(f"Execution Context:")
    for key, value in invalid_context.items():
        print(f"  {key}: {value}")
    
    errors = validator.validate_execution_context(task_agent, invalid_context)
    print("\nValidation result:")
    print_validation_errors(errors)


def demo_calendar_event_task() -> None:
    """Demonstrate a calendar event task."""
    print_header("Calendar Event Task Demo")
    
    # Create calendar event parameters
    calendar_parameters = {
        "title": "Team Meeting",
        "start_time": datetime(2023, 12, 25, 10, 0),
        "end_time": datetime(2023, 12, 25, 11, 0),
        "attendees": ["alice@example.com", "bob@example.com"],
        "location": "Conference Room A"
    }
    
    # Create preconditions
    preconditions = [
        {
            "type": "authentication",
            "name": "user_auth",
            "description": "User must be authenticated"
        },
        {
            "type": "permission",
            "name": "calendar_permission",
            "description": "User must have calendar write permission",
            "required_permissions": ["calendar.write"]
        },
        {
            "type": "resource_availability",
            "name": "calendar_api",
            "description": "Calendar API must be available",
            "required_resources": ["calendar_api"]
        },
        {
            "type": "connectivity",
            "name": "network",
            "description": "Network connectivity is required"
        }
    ]
    
    # Create task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="calendar_event",
        parameters=calendar_parameters,
        preconditions=preconditions
    )
    
    print_task_info(task_agent)
    print_preconditions(task_agent)
    
    print_subheader("Execution with All Preconditions Met")
    
    # Create execution context
    context = {
        "is_authenticated": True,
        "user_permissions": ["calendar.write", "calendar.read"],
        "available_resources": ["calendar_api", "notification_service"],
        "is_connected": True,
        "has_conflicts": False
    }
    
    print(f"Execution Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context)
    print_result(result)
    
    print_subheader("Execution with Scheduling Conflicts (Warning)")
    
    # Reset task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="calendar_event",
        parameters=calendar_parameters,
        preconditions=preconditions
    )
    
    # Create execution context with conflicts
    context_with_conflicts = {
        "is_authenticated": True,
        "user_permissions": ["calendar.write", "calendar.read"],
        "available_resources": ["calendar_api", "notification_service"],
        "is_connected": True,
        "has_conflicts": True
    }
    
    print(f"Execution Context:")
    for key, value in context_with_conflicts.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context_with_conflicts)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context_with_conflicts)
    print_result(result)
    
    print_subheader("Execution with Preconditions Not Met")
    
    # Reset task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="calendar_event",
        parameters=calendar_parameters,
        preconditions=preconditions
    )
    
    # Create execution context with missing permissions
    context_missing_permissions = {
        "is_authenticated": True,
        "user_permissions": ["calendar.read"],  # Missing calendar.write
        "available_resources": ["calendar_api", "notification_service"],
        "is_connected": True
    }
    
    print(f"Execution Context:")
    for key, value in context_missing_permissions.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context_missing_permissions)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context_missing_permissions)
    print_result(result)


def demo_email_task() -> None:
    """Demonstrate an email task."""
    print_header("Email Task Demo")
    
    # Create email parameters
    email_parameters = {
        "recipients": ["alice@example.com", "bob@example.com"],
        "subject": "Meeting Agenda",
        "body": "Here's the agenda for our meeting.",
        "attachments": ["agenda.pdf"]
    }
    
    # Create preconditions
    preconditions = [
        {
            "type": "authentication",
            "name": "user_auth",
            "description": "User must be authenticated"
        },
        {
            "type": "permission",
            "name": "email_permission",
            "description": "User must have email send permission",
            "required_permissions": ["email.send"]
        },
        {
            "type": "connectivity",
            "name": "network",
            "description": "Network connectivity is required"
        }
    ]
    
    # Create task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="email",
        parameters=email_parameters,
        preconditions=preconditions
    )
    
    print_task_info(task_agent)
    print_preconditions(task_agent)
    
    print_subheader("Execution with All Preconditions Met")
    
    # Create execution context
    context = {
        "is_authenticated": True,
        "user_permissions": ["email.send", "email.read"],
        "is_connected": True,
        "attachment_size": 2 * 1024 * 1024  # 2 MB
    }
    
    print(f"Execution Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context)
    print_result(result)
    
    print_subheader("Execution with Large Attachments (Warning)")
    
    # Reset task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="email",
        parameters=email_parameters,
        preconditions=preconditions
    )
    
    # Create execution context with large attachments
    context_large_attachments = {
        "is_authenticated": True,
        "user_permissions": ["email.send", "email.read"],
        "is_connected": True,
        "attachment_size": 15 * 1024 * 1024  # 15 MB
    }
    
    print(f"Execution Context:")
    for key, value in context_large_attachments.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context_large_attachments)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context_large_attachments)
    print_result(result)
    
    print_subheader("Execution with No Connectivity")
    
    # Reset task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="email",
        parameters=email_parameters,
        preconditions=preconditions
    )
    
    # Create execution context with no connectivity
    context_no_connectivity = {
        "is_authenticated": True,
        "user_permissions": ["email.send", "email.read"],
        "is_connected": False
    }
    
    print(f"Execution Context:")
    for key, value in context_no_connectivity.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context_no_connectivity)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context_no_connectivity)
    print_result(result)


def demo_reminder_task() -> None:
    """Demonstrate a reminder task."""
    print_header("Reminder Task Demo")
    
    # Create reminder parameters with future due date
    future_date = datetime.now() + timedelta(days=7)
    reminder_parameters = {
        "title": "Submit Quarterly Report",
        "due_date": future_date,
        "priority": "high",
        "notes": "Don't forget to include the financial analysis."
    }
    
    # Create preconditions
    preconditions = [
        {
            "type": "authentication",
            "name": "user_auth",
            "description": "User must be authenticated"
        },
        {
            "type": "permission",
            "name": "reminder_permission",
            "description": "User must have reminder create permission",
            "required_permissions": ["reminder.create"]
        }
    ]
    
    # Create task agent
    task_agent = TaskAgentFactory.create_agent(
        task_type="reminder",
        parameters=reminder_parameters,
        preconditions=preconditions
    )
    
    print_task_info(task_agent)
    print_preconditions(task_agent)
    
    print_subheader("Execution with All Preconditions Met")
    
    # Create execution context
    context = {
        "is_authenticated": True,
        "user_permissions": ["reminder.create", "reminder.read"]
    }
    
    print(f"Execution Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
    
    # Check preconditions
    preconditions_met = task_agent.check_preconditions(context)
    print(f"\nPreconditions Met: {Fore.GREEN if preconditions_met else Fore.RED}{preconditions_met}{Style.RESET_ALL}")
    
    # Execute task
    result = task_agent.execute(context)
    print_result(result)
    
    print_subheader("Execution with Past Due Date (Warning)")
    
    # Create reminder parameters with past due date
    past_date = datetime.now() - timedelta(days=1)
    past_reminder_parameters = {
        "title": "Submit Quarterly Report",
        "due_date": past_date,
        "priority": "high",
        "notes": "Don't forget to include the financial analysis."
    }
    
    # Create task agent with past due date
    past_task_agent = TaskAgentFactory.create_agent(
        task_type="reminder",
        parameters=past_reminder_parameters,
        preconditions=preconditions
    )
    
    print_task_info(past_task_agent)
    
    # Execute task
    result = past_task_agent.execute(context)
    print_result(result)


def main() -> None:
    """Main function to run the demo."""
    print_header("Task-Oriented Agent Validation Demo")
    
    print(f"""
This demo showcases validation patterns specific to task-oriented agents, including:

1. Parameter validation for different task types
2. Precondition validation for task execution
3. Execution context validation
4. Task execution with various validation scenarios
5. Error handling and reporting

These validation patterns help ensure that task-oriented agents properly validate
inputs, check preconditions, track execution status, and handle errors appropriately.
""")
    
    while True:
        print(f"\n{Fore.YELLOW}Choose a demo to run (1-6, or q to quit):{Style.RESET_ALL}")
        print("1. Parameter Validation")
        print("2. Precondition Validation")
        print("3. Execution Context Validation")
        print("4. Calendar Event Task")
        print("5. Email Task")
        print("6. Reminder Task")
        print("q. Quit")
        
        choice = input("> ").strip().lower()
        
        if choice == "q":
            break
        elif choice == "1":
            demo_parameter_validation()
        elif choice == "2":
            demo_precondition_validation()
        elif choice == "3":
            demo_execution_context_validation()
        elif choice == "4":
            demo_calendar_event_task()
        elif choice == "5":
            demo_email_task()
        elif choice == "6":
            demo_reminder_task()
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-6 or q.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
