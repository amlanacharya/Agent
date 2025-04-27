"""
Tests for Exercise 4.7.2: Task-Oriented Agent Validation
-----------------------------------------------------
This module contains tests for the task-oriented agent validation patterns.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

from exercise4_7_2_task_agent_validator import (
    ExecutionStatus, ErrorSeverity, TaskError, TaskParameters,
    CalendarEventParameters, ReminderParameters, EmailParameters,
    SearchParameters, WeatherParameters, TaskPrecondition,
    AuthenticationPrecondition, PermissionPrecondition,
    ResourceAvailabilityPrecondition, ConnectivityPrecondition,
    TaskResult, TaskAgent, CalendarEventAgent, ReminderAgent,
    EmailAgent, TaskAgentFactory, TaskAgentValidator
)


class TestTaskParameters(unittest.TestCase):
    """Tests for the task parameter classes."""

    def test_calendar_event_parameters(self):
        """Test validation of calendar event parameters."""
        # Valid parameters
        valid_params = CalendarEventParameters(
            title="Team Meeting",
            start_time=datetime(2023, 12, 25, 10, 0),
            end_time=datetime(2023, 12, 25, 11, 0),
            attendees=["alice@example.com", "bob@example.com"],
            location="Conference Room A"
        )
        self.assertEqual(valid_params.title, "Team Meeting")
        self.assertEqual(len(valid_params.attendees), 2)

        # Invalid parameters (end_time before start_time)
        with self.assertRaises(ValueError):
            CalendarEventParameters(
                title="Invalid Meeting",
                start_time=datetime(2023, 12, 25, 11, 0),
                end_time=datetime(2023, 12, 25, 10, 0)
            )

    def test_email_parameters(self):
        """Test validation of email parameters."""
        # Valid parameters
        valid_params = EmailParameters(
            recipients=["alice@example.com", "bob@example.com"],
            subject="Meeting Agenda",
            body="Here's the agenda for our meeting."
        )
        self.assertEqual(valid_params.subject, "Meeting Agenda")
        self.assertEqual(len(valid_params.recipients), 2)

        # Invalid parameters (invalid email address)
        with self.assertRaises(ValueError):
            EmailParameters(
                recipients=["alice@example.com", "invalid-email"],
                subject="Meeting Agenda",
                body="Here's the agenda for our meeting."
            )

    def test_search_parameters(self):
        """Test validation of search parameters."""
        # Valid parameters
        valid_params = SearchParameters(
            query="task agent validation",
            sources=["web", "documentation"],
            max_results=20
        )
        self.assertEqual(valid_params.query, "task agent validation")
        self.assertEqual(valid_params.max_results, 20)

        # Invalid parameters (negative max_results)
        with self.assertRaises(ValueError):
            SearchParameters(
                query="task agent validation",
                max_results=-5
            )


class TestTaskPreconditions(unittest.TestCase):
    """Tests for the task precondition classes."""

    def test_authentication_precondition(self):
        """Test authentication precondition."""
        precondition = AuthenticationPrecondition()

        # Test with authenticated user
        context = {"is_authenticated": True}
        self.assertTrue(precondition.check(context))
        self.assertTrue(precondition.is_met)
        self.assertIsNone(precondition.error_message)

        # Test with unauthenticated user
        context = {"is_authenticated": False}
        self.assertFalse(precondition.check(context))
        self.assertFalse(precondition.is_met)
        self.assertEqual(precondition.error_message, "User is not authenticated")

    def test_permission_precondition(self):
        """Test permission precondition."""
        precondition = PermissionPrecondition(
            required_permissions=["calendar.write", "notification.send"]
        )

        # Test with all required permissions
        context = {"user_permissions": ["calendar.write", "notification.send", "user.read"]}
        self.assertTrue(precondition.check(context))
        self.assertTrue(precondition.is_met)
        self.assertIsNone(precondition.error_message)

        # Test with missing permissions
        context = {"user_permissions": ["calendar.read", "user.read"]}
        self.assertFalse(precondition.check(context))
        self.assertFalse(precondition.is_met)
        self.assertIn("missing required permissions", precondition.error_message)

    def test_resource_availability_precondition(self):
        """Test resource availability precondition."""
        precondition = ResourceAvailabilityPrecondition(
            required_resources=["calendar_api", "notification_service"]
        )

        # Test with all required resources
        context = {"available_resources": ["calendar_api", "notification_service", "email_service"]}
        self.assertTrue(precondition.check(context))
        self.assertTrue(precondition.is_met)
        self.assertIsNone(precondition.error_message)

        # Test with missing resources
        context = {"available_resources": ["email_service"]}
        self.assertFalse(precondition.check(context))
        self.assertFalse(precondition.is_met)
        self.assertIn("Missing required resources", precondition.error_message)

    def test_connectivity_precondition(self):
        """Test connectivity precondition."""
        precondition = ConnectivityPrecondition()

        # Test with connectivity
        context = {"is_connected": True}
        self.assertTrue(precondition.check(context))
        self.assertTrue(precondition.is_met)
        self.assertIsNone(precondition.error_message)

        # Test without connectivity
        context = {"is_connected": False}
        self.assertFalse(precondition.check(context))
        self.assertFalse(precondition.is_met)
        self.assertEqual(precondition.error_message, "Network connectivity is not available")


class TestTaskAgent(unittest.TestCase):
    """Tests for the TaskAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calendar_parameters = {
            "title": "Team Meeting",
            "start_time": datetime(2023, 12, 25, 10, 0),
            "end_time": datetime(2023, 12, 25, 11, 0),
            "attendees": ["alice@example.com", "bob@example.com"],
            "location": "Conference Room A"
        }

        self.preconditions = [
            AuthenticationPrecondition(),
            PermissionPrecondition(required_permissions=["calendar.write"])
        ]

        self.task_agent = TaskAgent(
            task_type="calendar_event",
            parameters=self.calendar_parameters,
            preconditions=self.preconditions
        )

    def test_validate_parameters_type(self):
        """Test validation of parameters type."""
        # Test with dictionary parameters
        agent = CalendarEventAgent(
            task_type="calendar_event",
            parameters=self.calendar_parameters
        )
        agent.validate_parameters_type()
        self.assertIsInstance(agent.parameters, CalendarEventParameters)

        # Test with already typed parameters
        typed_params = CalendarEventParameters(**self.calendar_parameters)
        agent = CalendarEventAgent(
            task_type="calendar_event",
            parameters=typed_params
        )
        agent.validate_parameters_type()
        self.assertIsInstance(agent.parameters, CalendarEventParameters)

        # Test with invalid parameters
        invalid_params = {"title": "Invalid Meeting"}
        with self.assertRaises(ValueError):
            agent = CalendarEventAgent(
                task_type="calendar_event",
                parameters=invalid_params
            )
            agent.validate_parameters_type()

    def test_validate_execution_readiness(self):
        """Test validation of execution readiness."""
        # Test with preconditions not met
        self.task_agent.preconditions_met = False
        self.task_agent.execution_status = ExecutionStatus.PENDING
        self.task_agent.validate_execution_readiness()  # Should not raise error

        # Test with preconditions not met and trying to execute
        self.task_agent.execution_status = ExecutionStatus.IN_PROGRESS
        with self.assertRaises(ValueError):
            self.task_agent.validate_execution_readiness()

        # Test with preconditions met
        self.task_agent.preconditions_met = True
        self.task_agent.execution_status = ExecutionStatus.IN_PROGRESS
        self.task_agent.validate_execution_readiness()  # Should not raise error

    def test_check_preconditions(self):
        """Test checking preconditions."""
        # Test with all preconditions met
        context = {
            "is_authenticated": True,
            "user_permissions": ["calendar.write", "calendar.read"]
        }
        self.assertTrue(self.task_agent.check_preconditions(context))
        self.assertTrue(self.task_agent.preconditions_met)
        self.assertEqual(len(self.task_agent.errors), 0)

        # Test with some preconditions not met
        context = {
            "is_authenticated": False,
            "user_permissions": ["calendar.read"]
        }
        self.assertFalse(self.task_agent.check_preconditions(context))
        self.assertFalse(self.task_agent.preconditions_met)
        self.assertEqual(len(self.task_agent.errors), 2)  # Two errors for two failed preconditions

    def test_execute(self):
        """Test task execution."""
        # Test execution with preconditions met
        context = {
            "is_authenticated": True,
            "user_permissions": ["calendar.write", "calendar.read"]
        }
        result = self.task_agent.execute(context)
        self.assertTrue(result.success)
        self.assertEqual(self.task_agent.execution_status, ExecutionStatus.COMPLETED)

        # Test execution with preconditions not met
        self.task_agent.execution_status = ExecutionStatus.PENDING
        self.task_agent.preconditions_met = False
        self.task_agent.errors = []

        context = {
            "is_authenticated": False,
            "user_permissions": []
        }
        result = self.task_agent.execute(context)
        self.assertFalse(result.success)
        self.assertEqual(self.task_agent.execution_status, ExecutionStatus.PENDING)
        self.assertIn("preconditions not met", result.message)


class TestCalendarEventAgent(unittest.TestCase):
    """Tests for the CalendarEventAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calendar_parameters = CalendarEventParameters(
            title="Team Meeting",
            start_time=datetime(2023, 12, 25, 10, 0),
            end_time=datetime(2023, 12, 25, 11, 0),
            attendees=["alice@example.com", "bob@example.com"],
            location="Conference Room A"
        )

        self.preconditions = [
            AuthenticationPrecondition(),
            PermissionPrecondition(required_permissions=["calendar.write"])
        ]

        self.agent = CalendarEventAgent(
            parameters=self.calendar_parameters,
            preconditions=self.preconditions
        )

    def test_execute_success(self):
        """Test successful execution."""
        context = {
            "is_authenticated": True,
            "user_permissions": ["calendar.write", "calendar.read"],
            "has_conflicts": False
        }

        result = self.agent.execute(context)
        self.assertTrue(result.success)
        self.assertEqual(self.agent.execution_status, ExecutionStatus.COMPLETED)
        self.assertIn("event_id", result.data)
        self.assertEqual(result.data["title"], "Team Meeting")
        self.assertEqual(result.data["has_warnings"], False)

    def test_execute_with_conflicts(self):
        """Test execution with scheduling conflicts."""
        context = {
            "is_authenticated": True,
            "user_permissions": ["calendar.write", "calendar.read"],
            "has_conflicts": True
        }

        result = self.agent.execute(context)
        self.assertTrue(result.success)  # Still succeeds but with warnings
        self.assertEqual(self.agent.execution_status, ExecutionStatus.COMPLETED)
        self.assertEqual(result.data["has_warnings"], True)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].severity, ErrorSeverity.WARNING)
        self.assertIn("conflicts", result.errors[0].message)

    def test_execute_failure(self):
        """Test execution failure due to preconditions."""
        context = {
            "is_authenticated": False,
            "user_permissions": ["calendar.read"]
        }

        result = self.agent.execute(context)
        self.assertFalse(result.success)
        self.assertEqual(self.agent.execution_status, ExecutionStatus.PENDING)
        self.assertIn("preconditions not met", result.message)
        self.assertEqual(len(result.errors), 2)  # Two errors for two failed preconditions


class TestTaskAgentFactory(unittest.TestCase):
    """Tests for the TaskAgentFactory class."""

    def test_create_calendar_agent(self):
        """Test creating a calendar event agent."""
        calendar_parameters = {
            "title": "Team Meeting",
            "start_time": datetime(2023, 12, 25, 10, 0),
            "end_time": datetime(2023, 12, 25, 11, 0),
            "attendees": ["alice@example.com", "bob@example.com"]
        }

        preconditions = [
            {
                "type": "authentication",
                "name": "user_auth",
                "description": "User must be authenticated"
            }
        ]

        agent = TaskAgentFactory.create_agent(
            task_type="calendar_event",
            parameters=calendar_parameters,
            preconditions=preconditions
        )

        self.assertIsInstance(agent, CalendarEventAgent)
        self.assertEqual(agent.task_type, "calendar_event")
        self.assertIsInstance(agent.parameters, CalendarEventParameters)
        self.assertEqual(len(agent.preconditions), 1)
        self.assertIsInstance(agent.preconditions[0], AuthenticationPrecondition)

    def test_create_reminder_agent(self):
        """Test creating a reminder agent."""
        reminder_parameters = {
            "title": "Submit Report",
            "due_date": datetime(2023, 12, 25, 17, 0),
            "priority": "high"
        }

        agent = TaskAgentFactory.create_agent(
            task_type="reminder",
            parameters=reminder_parameters
        )

        self.assertIsInstance(agent, ReminderAgent)
        self.assertEqual(agent.task_type, "reminder")
        self.assertIsInstance(agent.parameters, ReminderParameters)

    def test_create_email_agent(self):
        """Test creating an email agent."""
        email_parameters = {
            "recipients": ["alice@example.com"],
            "subject": "Meeting Agenda",
            "body": "Here's the agenda for our meeting."
        }

        agent = TaskAgentFactory.create_agent(
            task_type="email",
            parameters=email_parameters
        )

        self.assertIsInstance(agent, EmailAgent)
        self.assertEqual(agent.task_type, "email")
        self.assertIsInstance(agent.parameters, EmailParameters)

    def test_create_generic_agent(self):
        """Test creating a generic agent for unsupported task type."""
        parameters = {
            "query": "weather in New York"
        }

        agent = TaskAgentFactory.create_agent(
            task_type="custom_task",
            parameters=parameters
        )

        self.assertIsInstance(agent, TaskAgent)
        self.assertEqual(agent.task_type, "custom_task")
        self.assertEqual(agent.parameters, parameters)


class TestTaskAgentValidator(unittest.TestCase):
    """Tests for the TaskAgentValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = TaskAgentValidator()

    def test_validate_parameters(self):
        """Test parameter validation."""
        # Valid calendar parameters
        calendar_parameters = {
            "title": "Team Meeting",
            "start_time": datetime(2023, 12, 25, 10, 0),
            "end_time": datetime(2023, 12, 25, 11, 0),
            "attendees": ["alice@example.com", "bob@example.com"]
        }
        errors = self.validator.validate_parameters("calendar_event", calendar_parameters)
        self.assertEqual(len(errors), 0)

        # Invalid calendar parameters
        invalid_calendar_parameters = {
            "title": "Invalid Meeting",
            "start_time": datetime(2023, 12, 25, 11, 0),
            "end_time": datetime(2023, 12, 25, 10, 0)  # End time before start time
        }
        errors = self.validator.validate_parameters("calendar_event", invalid_calendar_parameters)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid parameters", errors[0])

        # Unsupported task type
        errors = self.validator.validate_parameters("unsupported_task", {})
        self.assertEqual(len(errors), 1)
        self.assertIn("Unsupported task type", errors[0])

    def test_validate_preconditions(self):
        """Test precondition validation."""
        # Valid preconditions
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
        errors = self.validator.validate_preconditions(preconditions)
        self.assertEqual(len(errors), 0)

        # Invalid preconditions (missing required_permissions)
        invalid_preconditions = [
            {
                "type": "permission",
                "name": "calendar_permission",
                "description": "User must have calendar write permission"
                # Missing required_permissions
            }
        ]
        errors = self.validator.validate_preconditions(invalid_preconditions)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing 'required_permissions'", errors[0])

    def test_validate_execution_context(self):
        """Test execution context validation."""
        # Create a task agent with various preconditions
        agent = TaskAgent(
            task_type="test_task",
            parameters={},
            preconditions=[
                AuthenticationPrecondition(),
                PermissionPrecondition(required_permissions=["test.permission"]),
                ResourceAvailabilityPrecondition(required_resources=["test_resource"]),
                ConnectivityPrecondition()
            ]
        )

        # Valid context
        valid_context = {
            "is_authenticated": True,
            "user_permissions": ["test.permission"],
            "available_resources": ["test_resource"],
            "is_connected": True
        }
        errors = self.validator.validate_execution_context(agent, valid_context)
        self.assertEqual(len(errors), 0)

        # Invalid context (missing fields)
        invalid_context = {
            "is_authenticated": True
            # Missing other required fields
        }
        errors = self.validator.validate_execution_context(agent, invalid_context)
        self.assertEqual(len(errors), 3)  # Three missing context fields


if __name__ == "__main__":
    unittest.main()
