"""
Exercise 4.6.5: Required Information Validator

This module implements a state validation system that ensures all required information
is collected before completing a task. It provides a flexible framework for defining
required fields for different task types and validating the completeness of collected
information.

Key features:
1. Define required fields for different task types
2. Track collected information and missing fields
3. Support for optional and conditional fields
4. Validation of field values against constraints
5. Prioritization of missing fields for better user experience
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from enum import Enum
from datetime import datetime
import uuid
import re


class FieldRequirement(str, Enum):
    """Enum for field requirement levels."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"


class FieldPriority(int, Enum):
    """Enum for field priority levels."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class FieldDefinition(BaseModel):
    """Definition of a field with validation rules."""
    name: str
    description: str
    requirement: FieldRequirement = FieldRequirement.REQUIRED
    priority: FieldPriority = FieldPriority.MEDIUM
    field_type: str = "string"
    validation_pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    depends_on: Optional[Dict[str, Any]] = None
    
    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against the field's constraints.
        
        Args:
            value: The value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Skip validation for None values if field is optional
        if value is None:
            if self.requirement == FieldRequirement.REQUIRED:
                return False, "This field is required"
            return True, None
            
        # Type validation
        if self.field_type == "string":
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
                
            # String-specific validations
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"Value must be at least {self.min_length} characters"
                
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"Value must be at most {self.max_length} characters"
                
            if self.validation_pattern is not None and not re.match(self.validation_pattern, value):
                return False, f"Value does not match required pattern"
                
        elif self.field_type == "number":
            if not isinstance(value, (int, float)):
                return False, f"Expected number, got {type(value).__name__}"
                
            # Number-specific validations
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be at least {self.min_value}"
                
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be at most {self.max_value}"
                
        elif self.field_type == "boolean":
            if not isinstance(value, bool):
                return False, f"Expected boolean, got {type(value).__name__}"
                
        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Value must be one of: {', '.join(str(v) for v in self.allowed_values)}"
            
        return True, None


class TaskDefinition(BaseModel):
    """Definition of a task with required fields."""
    task_type: str
    description: str
    fields: List[FieldDefinition]
    
    def get_field(self, field_name: str) -> Optional[FieldDefinition]:
        """Get a field definition by name."""
        for field in self.fields:
            if field.name == field_name:
                return field
        return None
    
    def get_required_fields(self) -> List[FieldDefinition]:
        """Get all required fields."""
        return [field for field in self.fields if field.requirement == FieldRequirement.REQUIRED]
    
    def get_conditional_fields(self) -> List[FieldDefinition]:
        """Get all conditional fields."""
        return [field for field in self.fields if field.requirement == FieldRequirement.CONDITIONAL]
    
    def get_optional_fields(self) -> List[FieldDefinition]:
        """Get all optional fields."""
        return [field for field in self.fields if field.requirement == FieldRequirement.OPTIONAL]


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    field_name: Optional[str] = None


class FieldStatus(BaseModel):
    """Status of a field in the task state."""
    field_name: str
    is_filled: bool = False
    value: Optional[Any] = None
    is_valid: bool = True
    error_message: Optional[str] = None
    last_updated: Optional[datetime] = None


class TaskState(BaseModel):
    """State of a task with collected information."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    field_status: Dict[str, FieldStatus] = {}
    is_complete: bool = False
    
    def update_field(self, field_name: str, value: Any) -> ValidationResult:
        """
        Update a field value and validate it.
        
        Args:
            field_name: Name of the field to update
            value: New value for the field
            
        Returns:
            ValidationResult with validation status
        """
        # Create field status if it doesn't exist
        if field_name not in self.field_status:
            self.field_status[field_name] = FieldStatus(field_name=field_name)
            
        # Update field value
        self.field_status[field_name].value = value
        self.field_status[field_name].is_filled = True
        self.field_status[field_name].last_updated = datetime.now()
        
        # Update task timestamp
        self.last_updated = datetime.now()
        
        return ValidationResult(is_valid=True)
    
    def get_field_value(self, field_name: str) -> Optional[Any]:
        """Get a field value."""
        if field_name in self.field_status and self.field_status[field_name].is_filled:
            return self.field_status[field_name].value
        return None
    
    def get_filled_fields(self) -> Set[str]:
        """Get names of all filled fields."""
        return {name for name, status in self.field_status.items() if status.is_filled}
    
    def get_valid_fields(self) -> Set[str]:
        """Get names of all valid fields."""
        return {name for name, status in self.field_status.items() 
                if status.is_filled and status.is_valid}


class RequiredInfoValidator(BaseModel):
    """
    Validator for ensuring all required information is collected.
    
    This class manages task definitions and validates task states to ensure
    all required information is collected before a task can be completed.
    """
    task_definitions: Dict[str, TaskDefinition] = {}
    
    def register_task_definition(self, task_definition: TaskDefinition) -> None:
        """Register a task definition."""
        self.task_definitions[task_definition.task_type] = task_definition
    
    def create_task_state(self, task_type: str) -> TaskState:
        """
        Create a new task state.
        
        Args:
            task_type: Type of task to create
            
        Returns:
            New TaskState instance
            
        Raises:
            ValueError: If task_type is not registered
        """
        if task_type not in self.task_definitions:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return TaskState(task_type=task_type)
    
    def validate_field(self, task_state: TaskState, field_name: str, value: Any) -> ValidationResult:
        """
        Validate a field value against its definition.
        
        Args:
            task_state: Current task state
            field_name: Name of the field to validate
            value: Value to validate
            
        Returns:
            ValidationResult with validation status
            
        Raises:
            ValueError: If task_type is not registered or field is not defined
        """
        # Get task definition
        task_def = self.task_definitions.get(task_state.task_type)
        if not task_def:
            raise ValueError(f"Unknown task type: {task_state.task_type}")
            
        # Get field definition
        field_def = task_def.get_field(field_name)
        if not field_def:
            raise ValueError(f"Unknown field: {field_name}")
            
        # Validate value
        is_valid, error_message = field_def.validate_value(value)
        
        # Update field status
        if field_name not in task_state.field_status:
            task_state.field_status[field_name] = FieldStatus(field_name=field_name)
            
        task_state.field_status[field_name].is_valid = is_valid
        task_state.field_status[field_name].error_message = error_message
        
        return ValidationResult(
            is_valid=is_valid,
            error_message=error_message,
            field_name=field_name
        )
    
    def update_field(self, task_state: TaskState, field_name: str, value: Any) -> ValidationResult:
        """
        Update a field value and validate it.
        
        Args:
            task_state: Current task state
            field_name: Name of the field to update
            value: New value for the field
            
        Returns:
            ValidationResult with validation status
        """
        # Validate field
        result = self.validate_field(task_state, field_name, value)
        
        # Update field if valid
        if result.is_valid:
            task_state.update_field(field_name, value)
            
        return result
    
    def check_completeness(self, task_state: TaskState) -> Dict[str, Any]:
        """
        Check if all required information is collected.
        
        Args:
            task_state: Current task state
            
        Returns:
            Dictionary with completeness information
            
        Raises:
            ValueError: If task_type is not registered
        """
        # Get task definition
        task_def = self.task_definitions.get(task_state.task_type)
        if not task_def:
            raise ValueError(f"Unknown task type: {task_state.task_type}")
            
        # Get filled fields
        filled_fields = task_state.get_filled_fields()
        valid_fields = task_state.get_valid_fields()
        
        # Check required fields
        missing_required = []
        invalid_fields = []
        
        for field in task_def.get_required_fields():
            if field.name not in filled_fields:
                missing_required.append(field.name)
            elif field.name not in valid_fields:
                invalid_fields.append(field.name)
        
        # Check conditional fields
        missing_conditional = []
        
        for field in task_def.get_conditional_fields():
            # Skip if dependencies not met
            if not field.depends_on:
                continue
                
            # Check if dependencies are met
            dependencies_met = True
            for dep_field, dep_value in field.depends_on.items():
                if task_state.get_field_value(dep_field) != dep_value:
                    dependencies_met = False
                    break
                    
            # If dependencies are met, field is required
            if dependencies_met:
                if field.name not in filled_fields:
                    missing_conditional.append(field.name)
                elif field.name not in valid_fields:
                    invalid_fields.append(field.name)
        
        # Determine if task is complete
        is_complete = (
            len(missing_required) == 0 and
            len(missing_conditional) == 0 and
            len(invalid_fields) == 0
        )
        
        # Update task state
        task_state.is_complete = is_complete
        
        # Get next field to request
        next_field = self._get_next_field_to_request(
            task_def, missing_required, missing_conditional, invalid_fields
        )
        
        return {
            "is_complete": is_complete,
            "missing_required": missing_required,
            "missing_conditional": missing_conditional,
            "invalid_fields": invalid_fields,
            "next_field": next_field
        }
    
    def _get_next_field_to_request(
        self, 
        task_def: TaskDefinition,
        missing_required: List[str],
        missing_conditional: List[str],
        invalid_fields: List[str]
    ) -> Optional[str]:
        """
        Get the next field to request from the user.
        
        This method prioritizes fields based on:
        1. Invalid fields (need correction)
        2. Required fields (by priority)
        3. Conditional fields (by priority)
        
        Args:
            task_def: Task definition
            missing_required: List of missing required fields
            missing_conditional: List of missing conditional fields
            invalid_fields: List of invalid fields
            
        Returns:
            Name of the next field to request, or None if no fields need to be requested
        """
        # First, check for invalid fields
        if invalid_fields:
            return invalid_fields[0]
            
        # Next, check for missing required fields
        if missing_required:
            # Get field definitions for missing required fields
            missing_required_defs = [
                task_def.get_field(field_name) 
                for field_name in missing_required
            ]
            
            # Sort by priority
            missing_required_defs.sort(key=lambda f: f.priority.value)
            
            # Return highest priority field
            return missing_required_defs[0].name
            
        # Finally, check for missing conditional fields
        if missing_conditional:
            # Get field definitions for missing conditional fields
            missing_conditional_defs = [
                task_def.get_field(field_name) 
                for field_name in missing_conditional
            ]
            
            # Sort by priority
            missing_conditional_defs.sort(key=lambda f: f.priority.value)
            
            # Return highest priority field
            return missing_conditional_defs[0].name
            
        return None
    
    def get_field_prompt(self, task_state: TaskState, field_name: str) -> str:
        """
        Get a prompt for requesting a field value.
        
        Args:
            task_state: Current task state
            field_name: Name of the field to request
            
        Returns:
            Prompt string
            
        Raises:
            ValueError: If task_type is not registered or field is not defined
        """
        # Get task definition
        task_def = self.task_definitions.get(task_state.task_type)
        if not task_def:
            raise ValueError(f"Unknown task type: {task_state.task_type}")
            
        # Get field definition
        field_def = task_def.get_field(field_name)
        if not field_def:
            raise ValueError(f"Unknown field: {field_name}")
            
        # Check if field has an error
        error_message = None
        if field_name in task_state.field_status:
            error_message = task_state.field_status[field_name].error_message
            
        # Build prompt
        prompt = f"Please provide {field_def.description}"
        
        # Add constraints
        constraints = []
        
        if field_def.min_length is not None:
            constraints.append(f"at least {field_def.min_length} characters")
            
        if field_def.max_length is not None:
            constraints.append(f"at most {field_def.max_length} characters")
            
        if field_def.min_value is not None:
            constraints.append(f"minimum value of {field_def.min_value}")
            
        if field_def.max_value is not None:
            constraints.append(f"maximum value of {field_def.max_value}")
            
        if field_def.allowed_values is not None:
            values_str = ", ".join(str(v) for v in field_def.allowed_values)
            constraints.append(f"one of: {values_str}")
            
        if constraints:
            prompt += f" ({', '.join(constraints)})"
            
        # Add error message if present
        if error_message:
            prompt += f"\n\nError: {error_message}"
            
        return prompt


# Example usage
def create_weather_query_task() -> TaskDefinition:
    """Create a weather query task definition."""
    return TaskDefinition(
        task_type="weather_query",
        description="Weather information request",
        fields=[
            FieldDefinition(
                name="location",
                description="the location for the weather forecast",
                requirement=FieldRequirement.REQUIRED,
                priority=FieldPriority.HIGH,
                field_type="string",
                min_length=2,
                max_length=100
            ),
            FieldDefinition(
                name="date",
                description="the date for the forecast (e.g., 'today', 'tomorrow', or a specific date)",
                requirement=FieldRequirement.OPTIONAL,
                priority=FieldPriority.MEDIUM,
                field_type="string"
            ),
            FieldDefinition(
                name="forecast_type",
                description="the type of forecast",
                requirement=FieldRequirement.OPTIONAL,
                priority=FieldPriority.LOW,
                field_type="string",
                allowed_values=["hourly", "daily", "weekly"]
            )
        ]
    )


def create_booking_query_task() -> TaskDefinition:
    """Create a booking query task definition."""
    return TaskDefinition(
        task_type="booking_query",
        description="Service booking request",
        fields=[
            FieldDefinition(
                name="service_type",
                description="the type of service to book",
                requirement=FieldRequirement.REQUIRED,
                priority=FieldPriority.HIGH,
                field_type="string",
                allowed_values=["haircut", "massage", "consultation"]
            ),
            FieldDefinition(
                name="date",
                description="the date for the booking",
                requirement=FieldRequirement.REQUIRED,
                priority=FieldPriority.HIGH,
                field_type="string"
            ),
            FieldDefinition(
                name="time",
                description="the time for the booking",
                requirement=FieldRequirement.REQUIRED,
                priority=FieldPriority.MEDIUM,
                field_type="string"
            ),
            FieldDefinition(
                name="stylist_preference",
                description="your preferred stylist",
                requirement=FieldRequirement.CONDITIONAL,
                priority=FieldPriority.LOW,
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
    )


def create_product_query_task() -> TaskDefinition:
    """Create a product query task definition."""
    return TaskDefinition(
        task_type="product_query",
        description="Product information request",
        fields=[
            FieldDefinition(
                name="product_name",
                description="the name of the product",
                requirement=FieldRequirement.REQUIRED,
                priority=FieldPriority.HIGH,
                field_type="string",
                min_length=2
            ),
            FieldDefinition(
                name="category",
                description="the product category",
                requirement=FieldRequirement.OPTIONAL,
                priority=FieldPriority.MEDIUM,
                field_type="string"
            ),
            FieldDefinition(
                name="price_range",
                description="the price range (e.g., 'under $50', '$50-$100', 'over $100')",
                requirement=FieldRequirement.OPTIONAL,
                priority=FieldPriority.LOW,
                field_type="string"
            )
        ]
    )


if __name__ == "__main__":
    # Create validator and register task definitions
    validator = RequiredInfoValidator()
    validator.register_task_definition(create_weather_query_task())
    validator.register_task_definition(create_booking_query_task())
    validator.register_task_definition(create_product_query_task())
    
    # Create a task state
    task_state = validator.create_task_state("weather_query")
    
    # Check initial completeness
    completeness = validator.check_completeness(task_state)
    print(f"Initial completeness: {completeness}")
    
    # Get prompt for next field
    next_field = completeness["next_field"]
    if next_field:
        prompt = validator.get_field_prompt(task_state, next_field)
        print(f"\nPrompt: {prompt}")
    
    # Update field
    result = validator.update_field(task_state, "location", "New York")
    print(f"\nUpdate result: {result}")
    
    # Check completeness again
    completeness = validator.check_completeness(task_state)
    print(f"\nCompleteness after update: {completeness}")
    
    # Create a booking query task
    booking_task = validator.create_task_state("booking_query")
    
    # Update fields
    validator.update_field(booking_task, "service_type", "haircut")
    validator.update_field(booking_task, "date", "tomorrow")
    
    # Check completeness
    completeness = validator.check_completeness(booking_task)
    print(f"\nBooking task completeness: {completeness}")
    
    # Get prompt for next field
    next_field = completeness["next_field"]
    if next_field:
        prompt = validator.get_field_prompt(booking_task, next_field)
        print(f"\nPrompt: {prompt}")
