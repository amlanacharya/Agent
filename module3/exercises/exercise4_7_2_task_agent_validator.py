"""
Exercise 4.7.2: Task-Oriented Agent Validation
--------------------------------------------
This module implements validation patterns specific to task-oriented agents, focusing on:
1. Task parameter validation
2. Precondition validation
3. Execution status validation
4. Task completion validation
5. Error handling validation

These validation patterns help ensure that task-oriented agents properly validate
inputs, check preconditions, track execution status, and handle errors appropriately.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Literal, Set, Any, Union, Type, TypeVar, Generic, Callable
from enum import Enum
from datetime import datetime, date, time
import uuid
import re
from abc import ABC, abstractmethod


class ExecutionStatus(str, Enum):
    """Enum for task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ErrorSeverity(str, Enum):
    """Enum for error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TaskError(BaseModel):
    """Model for a task execution error."""
    error_code: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)


class TaskParameters(BaseModel):
    """Base class for task parameters."""
    pass


class CalendarEventParameters(TaskParameters):
    """Parameters for a calendar event task."""
    title: str
    start_time: datetime
    end_time: datetime
    attendees: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    description: Optional[str] = None
    is_all_day: bool = False

    @model_validator(mode='after')
    def validate_times(self):
        """Validate that end_time is after start_time."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self


class ReminderParameters(TaskParameters):
    """Parameters for a reminder task."""
    title: str
    due_date: datetime
    priority: Literal["low", "medium", "high"] = "medium"
    notes: Optional[str] = None


class EmailParameters(TaskParameters):
    """Parameters for an email task."""
    recipients: List[str]
    subject: str
    body: str
    attachments: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)

    @field_validator('recipients', 'cc', 'bcc')
    @classmethod
    def validate_email_addresses(cls, emails):
        """Validate that all email addresses are valid."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for email in emails:
            if not re.match(email_pattern, email):
                raise ValueError(f"Invalid email address: {email}")
        return emails


class SearchParameters(TaskParameters):
    """Parameters for a search task."""
    query: str
    sources: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = 10

    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v):
        """Validate that max_results is positive."""
        if v <= 0:
            raise ValueError("max_results must be positive")
        return v


class WeatherParameters(TaskParameters):
    """Parameters for a weather task."""
    location: str
    date: Optional[date] = None
    forecast_type: Literal["current", "hourly", "daily", "weekly"] = "current"


class TaskPrecondition(BaseModel):
    """Model for a task precondition."""
    name: str
    description: str
    is_met: bool = False
    error_message: Optional[str] = None

    def check(self, context: Dict[str, Any]) -> bool:
        """
        Check if the precondition is met.

        This is a placeholder method that should be overridden by subclasses.
        """
        return self.is_met


class AuthenticationPrecondition(TaskPrecondition):
    """Precondition for user authentication."""
    name: str = "authentication"
    description: str = "User must be authenticated"

    def check(self, context: Dict[str, Any]) -> bool:
        """Check if the user is authenticated."""
        self.is_met = context.get("is_authenticated", False)
        if not self.is_met:
            self.error_message = "User is not authenticated"
        return self.is_met


class PermissionPrecondition(TaskPrecondition):
    """Precondition for user permissions."""
    name: str = "permission"
    description: str = "User must have required permissions"
    required_permissions: List[str] = Field(default_factory=list)

    def check(self, context: Dict[str, Any]) -> bool:
        """Check if the user has the required permissions."""
        user_permissions = context.get("user_permissions", [])
        self.is_met = all(perm in user_permissions for perm in self.required_permissions)
        if not self.is_met:
            missing = [perm for perm in self.required_permissions if perm not in user_permissions]
            self.error_message = f"User is missing required permissions: {', '.join(missing)}"
        return self.is_met


class ResourceAvailabilityPrecondition(TaskPrecondition):
    """Precondition for resource availability."""
    name: str = "resource_availability"
    description: str = "Required resources must be available"
    required_resources: List[str] = Field(default_factory=list)

    def check(self, context: Dict[str, Any]) -> bool:
        """Check if the required resources are available."""
        available_resources = context.get("available_resources", [])
        self.is_met = all(res in available_resources for res in self.required_resources)
        if not self.is_met:
            missing = [res for res in self.required_resources if res not in available_resources]
            self.error_message = f"Missing required resources: {', '.join(missing)}"
        return self.is_met


class ConnectivityPrecondition(TaskPrecondition):
    """Precondition for network connectivity."""
    name: str = "connectivity"
    description: str = "Network connectivity is required"

    def check(self, context: Dict[str, Any]) -> bool:
        """Check if network connectivity is available."""
        self.is_met = context.get("is_connected", False)
        if not self.is_met:
            self.error_message = "Network connectivity is not available"
        return self.is_met


class TaskResult(BaseModel):
    """Model for a task execution result."""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[TaskError] = Field(default_factory=list)


T = TypeVar('T', bound=TaskParameters)


class TaskAgent(BaseModel, Generic[T]):
    """
    Model for a task-oriented agent with validation for parameters, preconditions, and execution status.

    Generic type T must be a subclass of TaskParameters.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    parameters: Union[T, Dict[str, Any]]
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    preconditions: List[TaskPrecondition] = Field(default_factory=list)
    preconditions_met: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    errors: List[TaskError] = Field(default_factory=list)
    result: Optional[TaskResult] = None

    @model_validator(mode='after')
    def validate_parameters_type(self):
        """Validate that parameters match the task type."""
        parameter_types = {
            "calendar_event": CalendarEventParameters,
            "reminder": ReminderParameters,
            "email": EmailParameters,
            "search": SearchParameters,
            "weather": WeatherParameters
        }

        # Skip validation for generic TaskAgent class (for testing purposes)
        if self.__class__ == TaskAgent:
            return self

        if self.task_type in parameter_types:
            expected_type = parameter_types[self.task_type]
            if not isinstance(self.parameters, expected_type):
                if isinstance(self.parameters, dict):
                    # Try to convert dict to proper type
                    try:
                        self.parameters = expected_type(**self.parameters)
                    except Exception as e:
                        raise ValueError(f"Invalid {self.task_type} parameters: {e}")
                else:
                    raise ValueError(f"{self.task_type} task requires {expected_type.__name__}")

        return self

    @model_validator(mode='after')
    def validate_execution_readiness(self):
        """Validate that the task can be executed."""
        if self.execution_status == ExecutionStatus.IN_PROGRESS and not self.preconditions_met:
            raise ValueError("Cannot execute task: preconditions not met")
        return self

    def check_preconditions(self, context: Dict[str, Any]) -> bool:
        """
        Check if all preconditions are met.

        Args:
            context: Context information for checking preconditions

        Returns:
            True if all preconditions are met, False otherwise
        """
        all_met = True
        for precondition in self.preconditions:
            if not precondition.check(context):
                all_met = False
                # Add error if not already present
                error_message = precondition.error_message or f"Precondition not met: {precondition.name}"
                self._add_error(
                    error_code=f"precondition_{precondition.name}",
                    message=error_message,
                    severity=ErrorSeverity.ERROR
                )

        self.preconditions_met = all_met
        self.updated_at = datetime.now()
        return all_met

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        """
        Execute the task.

        This is a placeholder method that should be overridden by subclasses.

        Args:
            context: Context information for task execution

        Returns:
            TaskResult with execution result
        """
        # Check preconditions
        if not self.check_preconditions(context):
            return TaskResult(
                success=False,
                message="Task execution failed: preconditions not met",
                errors=self.errors
            )

        # Update status
        self.execution_status = ExecutionStatus.IN_PROGRESS
        self.updated_at = datetime.now()

        # Execute task (placeholder)
        # In a real implementation, this would be overridden by subclasses
        success = True
        message = "Task executed successfully"
        data = {}

        # Update status based on result
        if success:
            self.execution_status = ExecutionStatus.COMPLETED
        else:
            self.execution_status = ExecutionStatus.FAILED

        self.updated_at = datetime.now()

        # Create result
        result = TaskResult(
            success=success,
            message=message,
            data=data,
            errors=self.errors
        )

        self.result = result
        return result

    def _add_error(self, error_code: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, details: Dict[str, Any] = None):
        """Add an error to the task."""
        error = TaskError(
            error_code=error_code,
            message=message,
            severity=severity,
            details=details or {}
        )
        self.errors.append(error)
        self.updated_at = datetime.now()


class CalendarEventAgent(TaskAgent[CalendarEventParameters]):
    """Task agent for creating calendar events."""
    task_type: str = "calendar_event"

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the calendar event task."""
        # Check preconditions
        if not self.check_preconditions(context):
            return TaskResult(
                success=False,
                message="Calendar event creation failed: preconditions not met",
                errors=self.errors
            )

        # Update status
        self.execution_status = ExecutionStatus.IN_PROGRESS
        self.updated_at = datetime.now()

        # Execute task (simulated)
        try:
            # Simulate calendar API call
            event_id = f"evt-{uuid.uuid4().hex[:8]}"

            # Check for scheduling conflicts (simulated)
            if context.get("has_conflicts", False):
                self._add_error(
                    error_code="scheduling_conflict",
                    message="Time slot conflicts with existing events",
                    severity=ErrorSeverity.WARNING
                )

            # Create result data
            data = {
                "event_id": event_id,
                "title": self.parameters.title,
                "start_time": self.parameters.start_time.isoformat(),
                "end_time": self.parameters.end_time.isoformat(),
                "attendees": self.parameters.attendees,
                "location": self.parameters.location,
                "has_warnings": len([e for e in self.errors if e.severity == ErrorSeverity.WARNING]) > 0
            }

            # Update status
            self.execution_status = ExecutionStatus.COMPLETED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=True,
                message="Calendar event created successfully",
                data=data,
                errors=self.errors
            )

            self.result = result
            return result

        except Exception as e:
            # Handle execution error
            self._add_error(
                error_code="execution_error",
                message=f"Error creating calendar event: {str(e)}",
                severity=ErrorSeverity.ERROR,
                details={"exception": str(e)}
            )

            # Update status
            self.execution_status = ExecutionStatus.FAILED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=False,
                message="Calendar event creation failed",
                errors=self.errors
            )

            self.result = result
            return result


class ReminderAgent(TaskAgent[ReminderParameters]):
    """Task agent for creating reminders."""
    task_type: str = "reminder"

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the reminder task."""
        # Check preconditions
        if not self.check_preconditions(context):
            return TaskResult(
                success=False,
                message="Reminder creation failed: preconditions not met",
                errors=self.errors
            )

        # Update status
        self.execution_status = ExecutionStatus.IN_PROGRESS
        self.updated_at = datetime.now()

        # Execute task (simulated)
        try:
            # Simulate reminder API call
            reminder_id = f"rem-{uuid.uuid4().hex[:8]}"

            # Check if due date is in the past
            if self.parameters.due_date < datetime.now():
                self._add_error(
                    error_code="past_due_date",
                    message="Due date is in the past",
                    severity=ErrorSeverity.WARNING
                )

            # Create result data
            data = {
                "reminder_id": reminder_id,
                "title": self.parameters.title,
                "due_date": self.parameters.due_date.isoformat(),
                "priority": self.parameters.priority,
                "has_warnings": len([e for e in self.errors if e.severity == ErrorSeverity.WARNING]) > 0
            }

            # Update status
            self.execution_status = ExecutionStatus.COMPLETED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=True,
                message="Reminder created successfully",
                data=data,
                errors=self.errors
            )

            self.result = result
            return result

        except Exception as e:
            # Handle execution error
            self._add_error(
                error_code="execution_error",
                message=f"Error creating reminder: {str(e)}",
                severity=ErrorSeverity.ERROR,
                details={"exception": str(e)}
            )

            # Update status
            self.execution_status = ExecutionStatus.FAILED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=False,
                message="Reminder creation failed",
                errors=self.errors
            )

            self.result = result
            return result


class EmailAgent(TaskAgent[EmailParameters]):
    """Task agent for sending emails."""
    task_type: str = "email"

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the email task."""
        # Check preconditions
        if not self.check_preconditions(context):
            return TaskResult(
                success=False,
                message="Email sending failed: preconditions not met",
                errors=self.errors
            )

        # Update status
        self.execution_status = ExecutionStatus.IN_PROGRESS
        self.updated_at = datetime.now()

        # Execute task (simulated)
        try:
            # Simulate email API call
            message_id = f"msg-{uuid.uuid4().hex[:8]}"

            # Check for large attachments
            if context.get("attachment_size", 0) > 10 * 1024 * 1024:  # 10 MB
                self._add_error(
                    error_code="large_attachment",
                    message="Attachments exceed recommended size limit",
                    severity=ErrorSeverity.WARNING
                )

            # Create result data
            data = {
                "message_id": message_id,
                "recipients": self.parameters.recipients,
                "subject": self.parameters.subject,
                "sent_at": datetime.now().isoformat(),
                "has_warnings": len([e for e in self.errors if e.severity == ErrorSeverity.WARNING]) > 0
            }

            # Update status
            self.execution_status = ExecutionStatus.COMPLETED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=True,
                message="Email sent successfully",
                data=data,
                errors=self.errors
            )

            self.result = result
            return result

        except Exception as e:
            # Handle execution error
            self._add_error(
                error_code="execution_error",
                message=f"Error sending email: {str(e)}",
                severity=ErrorSeverity.ERROR,
                details={"exception": str(e)}
            )

            # Update status
            self.execution_status = ExecutionStatus.FAILED
            self.updated_at = datetime.now()

            # Create result
            result = TaskResult(
                success=False,
                message="Email sending failed",
                errors=self.errors
            )

            self.result = result
            return result


class TaskAgentFactory:
    """Factory for creating task agents."""

    @staticmethod
    def create_agent(task_type: str, parameters: Dict[str, Any], preconditions: List[Dict[str, Any]] = None) -> TaskAgent:
        """
        Create a task agent of the specified type.

        Args:
            task_type: Type of task agent to create
            parameters: Parameters for the task
            preconditions: List of precondition configurations

        Returns:
            TaskAgent instance

        Raises:
            ValueError: If task_type is not supported
        """
        # Create precondition instances
        precondition_instances = []
        if preconditions:
            for precondition_config in preconditions:
                precondition_type = precondition_config.pop("type", None)
                if precondition_type == "authentication":
                    precondition_instances.append(AuthenticationPrecondition(**precondition_config))
                elif precondition_type == "permission":
                    precondition_instances.append(PermissionPrecondition(**precondition_config))
                elif precondition_type == "resource_availability":
                    precondition_instances.append(ResourceAvailabilityPrecondition(**precondition_config))
                elif precondition_type == "connectivity":
                    precondition_instances.append(ConnectivityPrecondition(**precondition_config))
                else:
                    # Generic precondition
                    precondition_instances.append(TaskPrecondition(**precondition_config))

        # Create agent based on task type
        if task_type == "calendar_event":
            return CalendarEventAgent(
                task_type=task_type,
                parameters=parameters,
                preconditions=precondition_instances
            )
        elif task_type == "reminder":
            return ReminderAgent(
                task_type=task_type,
                parameters=parameters,
                preconditions=precondition_instances
            )
        elif task_type == "email":
            return EmailAgent(
                task_type=task_type,
                parameters=parameters,
                preconditions=precondition_instances
            )
        else:
            # Generic task agent - don't convert parameters for custom task types
            agent = TaskAgent(
                task_type=task_type,
                parameters=parameters,
                preconditions=precondition_instances
            )
            # Store original parameters to avoid conversion
            if isinstance(parameters, dict):
                agent.parameters = parameters
            return agent


class TaskAgentValidator:
    """Validator for task agents."""

    @staticmethod
    def validate_parameters(task_type: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Validate parameters for a task type.

        Args:
            task_type: Type of task
            parameters: Parameters to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Get parameter class for task type
        parameter_types = {
            "calendar_event": CalendarEventParameters,
            "reminder": ReminderParameters,
            "email": EmailParameters,
            "search": SearchParameters,
            "weather": WeatherParameters
        }

        if task_type not in parameter_types:
            errors.append(f"Unsupported task type: {task_type}")
            return errors

        # Validate parameters
        parameter_class = parameter_types[task_type]
        try:
            parameter_class(**parameters)
        except Exception as e:
            errors.append(f"Invalid parameters: {str(e)}")

        return errors

    @staticmethod
    def validate_preconditions(preconditions: List[Dict[str, Any]]) -> List[str]:
        """
        Validate precondition configurations.

        Args:
            preconditions: List of precondition configurations

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for i, precondition in enumerate(preconditions):
            precondition_type = precondition.get("type")

            if not precondition_type:
                errors.append(f"Precondition {i+1} is missing 'type'")
                continue

            if precondition_type == "permission" and "required_permissions" not in precondition:
                errors.append(f"Permission precondition {i+1} is missing 'required_permissions'")

            if precondition_type == "resource_availability" and "required_resources" not in precondition:
                errors.append(f"Resource availability precondition {i+1} is missing 'required_resources'")

        return errors

    @staticmethod
    def validate_execution_context(task_agent: TaskAgent, context: Dict[str, Any]) -> List[str]:
        """
        Validate execution context for a task agent.

        Args:
            task_agent: Task agent to validate context for
            context: Execution context

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for required context based on preconditions
        for precondition in task_agent.preconditions:
            if isinstance(precondition, AuthenticationPrecondition) and "is_authenticated" not in context:
                errors.append("Context is missing 'is_authenticated' required by authentication precondition")

            if isinstance(precondition, PermissionPrecondition) and "user_permissions" not in context:
                errors.append("Context is missing 'user_permissions' required by permission precondition")

            if isinstance(precondition, ResourceAvailabilityPrecondition) and "available_resources" not in context:
                errors.append("Context is missing 'available_resources' required by resource availability precondition")

            if isinstance(precondition, ConnectivityPrecondition) and "is_connected" not in context:
                errors.append("Context is missing 'is_connected' required by connectivity precondition")

        return errors


# Example usage
if __name__ == "__main__":
    # Create a calendar event task
    calendar_parameters = {
        "title": "Team Meeting",
        "start_time": datetime(2023, 12, 25, 10, 0),
        "end_time": datetime(2023, 12, 25, 11, 0),
        "attendees": ["alice@example.com", "bob@example.com"],
        "location": "Conference Room A"
    }

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
        print("Parameter validation errors:")
        for error in parameter_errors:
            print(f"  - {error}")
    else:
        print("Parameters are valid")

    # Validate preconditions
    precondition_errors = validator.validate_preconditions(preconditions)
    if precondition_errors:
        print("Precondition validation errors:")
        for error in precondition_errors:
            print(f"  - {error}")
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
        "available_resources": ["calendar_api", "notification_service"],
        "is_connected": True,
        "has_conflicts": False
    }

    # Validate execution context
    context_errors = validator.validate_execution_context(task_agent, context)
    if context_errors:
        print("Context validation errors:")
        for error in context_errors:
            print(f"  - {error}")
    else:
        print("Execution context is valid")

    # Check preconditions
    if task_agent.check_preconditions(context):
        print("All preconditions are met")
    else:
        print("Some preconditions are not met:")
        for error in task_agent.errors:
            print(f"  - {error.message}")

    # Execute task
    result = task_agent.execute(context)
    print(f"Task execution {'succeeded' if result.success else 'failed'}: {result.message}")
    if result.data:
        print("Result data:")
        for key, value in result.data.items():
            print(f"  {key}: {value}")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  [{error.severity}] {error.message}")
