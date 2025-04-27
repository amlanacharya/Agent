# Exercise 4.7.2: Task-Oriented Agent Validation

## Overview

This exercise implements validation patterns specific to task-oriented agents, focusing on ensuring proper parameter validation, precondition checking, execution status tracking, and error handling. The validation system helps maintain high-quality task execution by enforcing domain-specific rules for task-oriented agents.

## Key Features

1. **Task Parameter Validation**: Validates parameters for different task types (calendar events, reminders, emails, etc.)
2. **Precondition Validation**: Ensures all required preconditions are met before task execution
3. **Execution Status Tracking**: Tracks and validates task execution states
4. **Error Handling**: Provides structured error reporting with severity levels
5. **Context Validation**: Validates execution context for different precondition types

## Components

### Task Parameter Models

- `TaskParameters`: Base class for all task parameters
- `CalendarEventParameters`: Parameters for calendar event tasks
- `ReminderParameters`: Parameters for reminder tasks
- `EmailParameters`: Parameters for email tasks
- `SearchParameters`: Parameters for search tasks
- `WeatherParameters`: Parameters for weather tasks

### Precondition Models

- `TaskPrecondition`: Base class for all task preconditions
- `AuthenticationPrecondition`: Ensures user is authenticated
- `PermissionPrecondition`: Ensures user has required permissions
- `ResourceAvailabilityPrecondition`: Ensures required resources are available
- `ConnectivityPrecondition`: Ensures network connectivity is available

### Task Agent Models

- `TaskAgent`: Generic task agent with validation for parameters, preconditions, and execution status
- `CalendarEventAgent`: Specialized agent for calendar event tasks
- `ReminderAgent`: Specialized agent for reminder tasks
- `EmailAgent`: Specialized agent for email tasks

### Support Classes

- `ExecutionStatus`: Enum for task execution status (pending, in_progress, completed, failed, etc.)
- `ErrorSeverity`: Enum for error severity levels (info, warning, error, critical)
- `TaskError`: Model for a task execution error
- `TaskResult`: Model for a task execution result
- `TaskAgentFactory`: Factory for creating task agents
- `TaskAgentValidator`: Validator for task agents

## Usage Example

```python
from exercise4_7_2_task_agent_validator import (
    TaskAgentFactory, TaskAgentValidator
)

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
    }
]

# Validate parameters
validator = TaskAgentValidator()
parameter_errors = validator.validate_parameters("calendar_event", calendar_parameters)
if parameter_errors:
    print("Parameter validation errors:", parameter_errors)
else:
    print("Parameters are valid")

# Validate preconditions
precondition_errors = validator.validate_preconditions(preconditions)
if precondition_errors:
    print("Precondition validation errors:", precondition_errors)
else:
    print("Preconditions are valid")

# Create task agent
task_agent = TaskAgentFactory.create_agent(
    task_type="calendar_event",
    parameters=calendar_parameters,
    preconditions=preconditions
)

# Create execution context
context = {
    "is_authenticated": True,
    "user_permissions": ["calendar.write", "calendar.read"],
    "available_resources": ["calendar_api"],
    "is_connected": True
}

# Validate execution context
context_errors = validator.validate_execution_context(task_agent, context)
if context_errors:
    print("Context validation errors:", context_errors)
else:
    print("Execution context is valid")

# Check preconditions
if task_agent.check_preconditions(context):
    print("All preconditions are met")
else:
    print("Some preconditions are not met")
    for error in task_agent.errors:
        print(f"- {error.message}")

# Execute task
result = task_agent.execute(context)
print(f"Task execution {'succeeded' if result.success else 'failed'}: {result.message}")
```

## Precondition Example

```python
from exercise4_7_2_task_agent_validator import (
    TaskAgent, AuthenticationPrecondition, PermissionPrecondition
)

# Create a task agent with preconditions
task_agent = TaskAgent(
    task_type="custom_task",
    parameters={"key": "value"},
    preconditions=[
        AuthenticationPrecondition(),
        PermissionPrecondition(required_permissions=["custom.permission"])
    ]
)

# Check preconditions with valid context
valid_context = {
    "is_authenticated": True,
    "user_permissions": ["custom.permission", "other.permission"]
}
task_agent.check_preconditions(valid_context)  # Returns True

# Check preconditions with invalid context
invalid_context = {
    "is_authenticated": False,
    "user_permissions": ["other.permission"]
}
task_agent.check_preconditions(invalid_context)  # Returns False
# task_agent.errors will contain error details
```

## Error Handling Example

```python
from exercise4_7_2_task_agent_validator import (
    TaskAgent, TaskError, ErrorSeverity
)

# Create a task agent
task_agent = TaskAgent(
    task_type="custom_task",
    parameters={"key": "value"}
)

# Add errors with different severity levels
task_agent._add_error(
    error_code="validation_error",
    message="Validation failed for parameter 'key'",
    severity=ErrorSeverity.ERROR
)

task_agent._add_error(
    error_code="resource_warning",
    message="Resource usage is high",
    severity=ErrorSeverity.WARNING
)

# Check errors
for error in task_agent.errors:
    print(f"[{error.severity.value}] {error.error_code}: {error.message}")
```

## Running the Demo

To run the interactive demo:

```bash
python demo_exercise4_7_2_task_agent_validator.py
```

The demo showcases:
1. Parameter Validation
2. Precondition Validation
3. Execution Context Validation
4. Calendar Event Task
5. Email Task
6. Reminder Task

## Running the Tests

To run the tests:

```bash
python -m unittest test_exercise4_7_2_task_agent_validator.py
```

## Integration with Agent Systems

This validator can be integrated with agent systems to:

1. **Validate Task Parameters**: Ensure task parameters are valid before execution
2. **Check Preconditions**: Verify that all required preconditions are met
3. **Track Execution Status**: Monitor and validate task execution states
4. **Handle Errors**: Provide structured error reporting with severity levels
5. **Validate Context**: Ensure execution context contains all required information

## Real-World Applications

- **Calendar Management**: Validate calendar event creation with proper time constraints
- **Email Systems**: Ensure emails have valid recipients and attachments
- **Reminder Services**: Validate reminder creation with appropriate due dates
- **Search Services**: Ensure search queries have valid parameters
- **Task Management**: Validate task creation and execution with proper preconditions
