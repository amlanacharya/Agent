"""
Exercise 4.4.3: Feedback System for Validation Failures

This exercise implements a feedback loop system that tracks validation failures
and suggests improvements to the validation logic.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
from collections import Counter


class ValidationFailureType(str, Enum):
    """Types of validation failures."""
    MISSING_ENTITY = "missing_entity"
    INVALID_ENTITY = "invalid_entity"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    MISSING_INTENT = "missing_intent"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_FORMAT = "invalid_format"
    UNSUPPORTED_ACTION = "unsupported_action"
    OTHER = "other"


class ValidationFailure(BaseModel):
    """Model for a single validation failure."""
    failure_id: str = Field(..., description="Unique identifier for the failure")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the failure occurred")
    failure_type: ValidationFailureType = Field(..., description="Type of validation failure")
    entity_type: Optional[str] = Field(None, description="Type of entity that failed validation")
    intent_type: Optional[str] = Field(None, description="Type of intent that failed validation")
    input_text: str = Field(..., description="Original user input text")
    details: str = Field(..., description="Detailed description of the failure")
    severity: int = Field(1, ge=1, le=5, description="Severity level (1-5)")
    resolved: bool = Field(False, description="Whether this failure has been resolved")


class ValidationFeedbackSystem(BaseModel):
    """
    Feedback system for tracking validation failures and suggesting improvements.
    
    This system records validation failures, analyzes patterns, and generates
    suggestions for improving validation logic.
    """
    failures: List[ValidationFailure] = Field(default_factory=list, description="List of recorded validation failures")
    
    def record_failure(self, failure: ValidationFailure) -> str:
        """
        Record a new validation failure.
        
        Args:
            failure: The validation failure to record
            
        Returns:
            The ID of the recorded failure
        """
        self.failures.append(failure)
        return failure.failure_id
    
    def get_failure_by_id(self, failure_id: str) -> Optional[ValidationFailure]:
        """
        Get a failure by its ID.
        
        Args:
            failure_id: The ID of the failure to retrieve
            
        Returns:
            The failure if found, None otherwise
        """
        for failure in self.failures:
            if failure.failure_id == failure_id:
                return failure
        return None
    
    def mark_as_resolved(self, failure_id: str) -> bool:
        """
        Mark a failure as resolved.
        
        Args:
            failure_id: The ID of the failure to mark as resolved
            
        Returns:
            True if the failure was found and marked as resolved, False otherwise
        """
        failure = self.get_failure_by_id(failure_id)
        if failure:
            failure.resolved = True
            return True
        return False
    
    def get_failures_by_type(self, failure_type: ValidationFailureType) -> List[ValidationFailure]:
        """
        Get all failures of a specific type.
        
        Args:
            failure_type: The type of failures to retrieve
            
        Returns:
            List of failures of the specified type
        """
        return [f for f in self.failures if f.failure_type == failure_type]
    
    def get_failures_by_entity(self, entity_type: str) -> List[ValidationFailure]:
        """
        Get all failures related to a specific entity type.
        
        Args:
            entity_type: The entity type to filter by
            
        Returns:
            List of failures related to the specified entity type
        """
        return [f for f in self.failures if f.entity_type == entity_type]
    
    def get_failures_by_intent(self, intent_type: str) -> List[ValidationFailure]:
        """
        Get all failures related to a specific intent type.
        
        Args:
            intent_type: The intent type to filter by
            
        Returns:
            List of failures related to the specified intent type
        """
        return [f for f in self.failures if f.intent_type == intent_type]
    
    def get_unresolved_failures(self) -> List[ValidationFailure]:
        """
        Get all unresolved failures.
        
        Returns:
            List of unresolved failures
        """
        return [f for f in self.failures if not f.resolved]
    
    def get_high_severity_failures(self, min_severity: int = 4) -> List[ValidationFailure]:
        """
        Get failures with high severity.
        
        Args:
            min_severity: Minimum severity level (default: 4)
            
        Returns:
            List of failures with severity >= min_severity
        """
        return [f for f in self.failures if f.severity >= min_severity]
    
    def get_common_failure_types(self, top_n: int = 3) -> List[Tuple[ValidationFailureType, int]]:
        """
        Get the most common failure types.
        
        Args:
            top_n: Number of top failure types to return
            
        Returns:
            List of (failure_type, count) tuples, sorted by count in descending order
        """
        counter = Counter([f.failure_type for f in self.failures])
        return counter.most_common(top_n)
    
    def get_common_entity_failures(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """
        Get the most common entity types that fail validation.
        
        Args:
            top_n: Number of top entity types to return
            
        Returns:
            List of (entity_type, count) tuples, sorted by count in descending order
        """
        # Filter out None values
        entity_types = [f.entity_type for f in self.failures if f.entity_type is not None]
        counter = Counter(entity_types)
        return counter.most_common(top_n)
    
    def get_improvement_suggestions(self) -> Dict[str, List[str]]:
        """
        Generate suggestions for improving validation based on failure patterns.
        
        Returns:
            Dictionary mapping improvement categories to lists of suggestions
        """
        suggestions = {
            "entity_extraction": [],
            "intent_classification": [],
            "ambiguity_resolution": [],
            "general": []
        }
        
        # Analyze failure types
        failure_types = self.get_common_failure_types(5)
        for failure_type, count in failure_types:
            if failure_type == ValidationFailureType.MISSING_ENTITY:
                suggestions["entity_extraction"].append(
                    f"Improve entity extraction (found {count} missing entity failures)"
                )
            elif failure_type == ValidationFailureType.INVALID_ENTITY:
                suggestions["entity_extraction"].append(
                    f"Enhance entity validation rules (found {count} invalid entity failures)"
                )
            elif failure_type == ValidationFailureType.AMBIGUOUS_INTENT:
                suggestions["intent_classification"].append(
                    f"Refine intent disambiguation (found {count} ambiguous intent failures)"
                )
            elif failure_type == ValidationFailureType.MISSING_INTENT:
                suggestions["intent_classification"].append(
                    f"Expand intent recognition patterns (found {count} missing intent failures)"
                )
            elif failure_type == ValidationFailureType.LOW_CONFIDENCE:
                suggestions["general"].append(
                    f"Improve confidence scoring (found {count} low confidence failures)"
                )
        
        # Analyze entity failures
        entity_failures = self.get_common_entity_failures(5)
        for entity_type, count in entity_failures:
            suggestions["entity_extraction"].append(
                f"Improve extraction of '{entity_type}' entities (failed {count} times)"
            )
        
        # Check for high severity unresolved issues
        high_severity = self.get_high_severity_failures()
        if high_severity:
            suggestions["general"].append(
                f"Address {len(high_severity)} high severity validation failures"
            )
        
        # Add ambiguity suggestions if needed
        ambiguity_failures = self.get_failures_by_type(ValidationFailureType.AMBIGUOUS_INTENT)
        if ambiguity_failures:
            suggestions["ambiguity_resolution"].append(
                f"Add more clarification questions for ambiguous inputs ({len(ambiguity_failures)} failures)"
            )
        
        return suggestions
    
    def generate_improvement_report(self) -> str:
        """
        Generate a human-readable report with improvement suggestions.
        
        Returns:
            Formatted report string
        """
        suggestions = self.get_improvement_suggestions()
        
        report = "# Validation Improvement Report\n\n"
        
        # Add summary statistics
        report += "## Summary\n\n"
        report += f"- Total validation failures: {len(self.failures)}\n"
        report += f"- Unresolved failures: {len(self.get_unresolved_failures())}\n"
        report += f"- High severity failures: {len(self.get_high_severity_failures())}\n\n"
        
        # Add common failure types
        report += "## Common Failure Types\n\n"
        for failure_type, count in self.get_common_failure_types(5):
            report += f"- {failure_type.value}: {count} occurrences\n"
        report += "\n"
        
        # Add improvement suggestions
        report += "## Improvement Suggestions\n\n"
        
        for category, category_suggestions in suggestions.items():
            if category_suggestions:
                report += f"### {category.replace('_', ' ').title()}\n\n"
                for suggestion in category_suggestions:
                    report += f"- {suggestion}\n"
                report += "\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create a feedback system
    feedback_system = ValidationFeedbackSystem()
    
    # Record some validation failures
    feedback_system.record_failure(
        ValidationFailure(
            failure_id="fail-001",
            failure_type=ValidationFailureType.MISSING_ENTITY,
            entity_type="date",
            input_text="Schedule a meeting tomorrow",
            details="Failed to extract date entity from 'tomorrow'",
            severity=3
        )
    )
    
    feedback_system.record_failure(
        ValidationFailure(
            failure_id="fail-002",
            failure_type=ValidationFailureType.MISSING_ENTITY,
            entity_type="location",
            input_text="What's the weather like?",
            details="Missing required location entity for weather intent",
            severity=2
        )
    )
    
    feedback_system.record_failure(
        ValidationFailure(
            failure_id="fail-003",
            failure_type=ValidationFailureType.AMBIGUOUS_INTENT,
            intent_type="booking",
            input_text="I need to make a reservation",
            details="Ambiguous between restaurant, hotel, and appointment booking",
            severity=4
        )
    )
    
    feedback_system.record_failure(
        ValidationFailure(
            failure_id="fail-004",
            failure_type=ValidationFailureType.INVALID_ENTITY,
            entity_type="date",
            input_text="Schedule for February 30th",
            details="Invalid date: February 30th does not exist",
            severity=2
        )
    )
    
    feedback_system.record_failure(
        ValidationFailure(
            failure_id="fail-005",
            failure_type=ValidationFailureType.MISSING_ENTITY,
            entity_type="location",
            input_text="How far is it?",
            details="Missing source and destination locations for distance query",
            severity=3
        )
    )
    
    # Mark one as resolved
    feedback_system.mark_as_resolved("fail-001")
    
    # Generate and print improvement report
    report = feedback_system.generate_improvement_report()
    print(report)
    
    # Print specific improvement suggestions
    print("Specific Improvement Suggestions:")
    suggestions = feedback_system.get_improvement_suggestions()
    for category, category_suggestions in suggestions.items():
        if category_suggestions:
            print(f"\n{category.replace('_', ' ').title()}:")
            for suggestion in category_suggestions:
                print(f"- {suggestion}")
