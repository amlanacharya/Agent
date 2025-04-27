# Exercise 4.6.5: Required Information Validator

## Overview

The Required Information Validator is a state validation system that ensures all required information is collected before completing a task. It provides a flexible framework for defining required fields for different task types and validating the completeness of collected information.

This exercise implements the fifth practice exercise from Lesson 4.6: State Validation in Agent Systems.

## Key Features

1. **Task Definition**: Define tasks with required, optional, and conditional fields
2. **Field Validation**: Validate field values against various constraints
3. **Completeness Checking**: Track missing fields and determine task completeness
4. **Conditional Requirements**: Support for fields that are only required under certain conditions
5. **Field Prioritization**: Prioritize which missing fields to request first
6. **Prompt Generation**: Generate user-friendly prompts for requesting missing information

## Components

### Field Definition

- `FieldDefinition`: Defines a field with validation rules
- `FieldRequirement`: Enum for field requirement levels (REQUIRED, OPTIONAL, CONDITIONAL)
- `FieldPriority`: Enum for field priority levels (HIGH, MEDIUM, LOW)

### Task Definition

- `TaskDefinition`: Defines a task with required fields
- Methods for retrieving required, optional, and conditional fields

### Task State

- `TaskState`: Tracks the state of a task with collected information
- `FieldStatus`: Tracks the status of a field (filled, valid, error message)
- Methods for updating fields and checking completeness

### Validator

- `RequiredInfoValidator`: Main class for validating task states
- Methods for registering task definitions, creating task states, validating fields, and checking completeness

## Usage Example

```python
from exercise4_6_5_required_info_validator import (
    RequiredInfoValidator, TaskDefinition, FieldDefinition,
    FieldRequirement, FieldPriority
)

# Create validator
validator = RequiredInfoValidator()

# Register task definition
validator.register_task_definition(TaskDefinition(
    task_type="weather_query",
    description="Weather information request",
    fields=[
        FieldDefinition(
            name="location",
            description="the location for the weather forecast",
            requirement=FieldRequirement.REQUIRED,
            priority=FieldPriority.HIGH,
            field_type="string",
            min_length=2
        ),
        FieldDefinition(
            name="date",
            description="the date for the forecast",
            requirement=FieldRequirement.OPTIONAL,
            priority=FieldPriority.MEDIUM,
            field_type="string"
        )
    ]
))

# Create task state
task_state = validator.create_task_state("weather_query")

# Check completeness
completeness = validator.check_completeness(task_state)
print(f"Is complete: {completeness['is_complete']}")
print(f"Missing required fields: {completeness['missing_required']}")

# Get prompt for next field
next_field = completeness["next_field"]
if next_field:
    prompt = validator.get_field_prompt(task_state, next_field)
    print(f"Prompt: {prompt}")

# Update field
validator.update_field(task_state, "location", "New York")

# Check completeness again
completeness = validator.check_completeness(task_state)
print(f"Is complete: {completeness['is_complete']}")
```

## Conditional Fields Example

```python
# Register task definition with conditional fields
validator.register_task_definition(TaskDefinition(
    task_type="booking_query",
    description="Service booking request",
    fields=[
        FieldDefinition(
            name="service_type",
            description="the type of service to book",
            requirement=FieldRequirement.REQUIRED,
            priority=FieldPriority.HIGH,
            field_type="string",
            allowed_values=["haircut", "massage"]
        ),
        FieldDefinition(
            name="stylist_preference",
            description="your preferred stylist",
            requirement=FieldRequirement.CONDITIONAL,
            priority=FieldPriority.MEDIUM,
            field_type="string",
            depends_on={"service_type": "haircut"}
        ),
        FieldDefinition(
            name="massage_type",
            description="the type of massage",
            requirement=FieldRequirement.CONDITIONAL,
            priority=FieldPriority.MEDIUM,
            field_type="string",
            allowed_values=["swedish", "deep tissue", "hot stone"],
            depends_on={"service_type": "massage"}
        )
    ]
))

# Create task state
task_state = validator.create_task_state("booking_query")

# Update service_type to haircut
validator.update_field(task_state, "service_type", "haircut")

# Check completeness
completeness = validator.check_completeness(task_state)
print(f"Missing conditional fields: {completeness['missing_conditional']}")
# Output: Missing conditional fields: ['stylist_preference']

# Change service_type to massage
validator.update_field(task_state, "service_type", "massage")

# Check completeness again
completeness = validator.check_completeness(task_state)
print(f"Missing conditional fields: {completeness['missing_conditional']}")
# Output: Missing conditional fields: ['massage_type']
```

## Running the Demo

To run the interactive demo:

```bash
python demo_exercise4_6_5_required_info_validator.py
```

The demo showcases:
1. Weather Query Task - Simple task with one required field
2. Booking Query Task - Complex task with required and conditional fields
3. Conditional Fields - Demonstration of how conditional fields work
4. Validation Errors - Handling of validation errors

## Running the Tests

To run the tests:

```bash
python -m unittest test_exercise4_6_5_required_info_validator.py
```

## Integration with Agent Systems

This validator can be integrated with agent systems to:

1. **Track Conversation State**: Ensure all required information is collected during a conversation
2. **Guide Conversation Flow**: Determine what information to request next
3. **Validate User Inputs**: Ensure user inputs meet validation requirements
4. **Handle Conditional Logic**: Adapt required information based on user choices
5. **Generate Prompts**: Create context-aware prompts for requesting information

## Real-World Applications

- **Customer Service Bots**: Collect all necessary information before processing a request
- **Form-Filling Assistants**: Guide users through complex forms
- **Booking Systems**: Ensure all required booking details are provided
- **Information Retrieval**: Collect all parameters needed for a search query
- **Task Management**: Validate task creation with all required fields
