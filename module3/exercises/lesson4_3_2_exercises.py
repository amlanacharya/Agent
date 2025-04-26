"""
Lesson 4.3.2: Change Tracking Mixin

This exercise implements a mixin for tracking model changes that records
the previous and new values of fields when they're updated.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import copy


class ChangeRecord(BaseModel):
    """Record of a single field change."""
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime = Field(default_factory=datetime.now)


class ChangeTrackingMixin(BaseModel):
    """Mixin that tracks changes to model fields."""
    change_history: List[ChangeRecord] = Field(default_factory=list, exclude=True)
    previous_state: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data):
        """Initialize the model and store the initial state."""
        super().__init__(**data)
        # Store initial state after initialization
        self._store_current_state()

    def _store_current_state(self):
        """Store the current state of the model."""
        # Get all fields except private ones (starting with _)
        self.previous_state = {
            field: copy.deepcopy(getattr(self, field))
            for field in self.__class__.model_fields
            if not field.startswith('_') and hasattr(self, field)
        }

    def update(self, **kwargs):
        """Update fields and track changes."""
        changes = []

        # Record changes for each field
        for field, new_value in kwargs.items():
            if field in self.__class__.model_fields and hasattr(self, field):
                old_value = getattr(self, field)

                # Only record if the value actually changed
                if old_value != new_value:
                    changes.append(ChangeRecord(
                        field_name=field,
                        old_value=old_value,
                        new_value=new_value,
                        timestamp=datetime.now()
                    ))

        # Update the model
        for field, value in kwargs.items():
            if field in self.__class__.model_fields:
                setattr(self, field, value)

        # Add changes to history
        self.change_history.extend(changes)

        # Update previous state
        self._store_current_state()

        return changes

    def get_change_history(self):
        """Get the complete change history."""
        return self.change_history

    def get_field_history(self, field_name: str):
        """Get the change history for a specific field."""
        return [
            change for change in self.change_history
            if change.field_name == field_name
        ]

    def revert_last_change(self):
        """Revert the most recent change."""
        if not self.change_history:
            return False

        # Get the most recent change
        last_change = self.change_history.pop()

        # Revert the field to its previous value
        setattr(self, last_change.field_name, last_change.old_value)

        # Update previous state
        self._store_current_state()

        return True

    def revert_all_changes(self):
        """Revert all changes to the initial state."""
        if not self.change_history:
            return False

        # Group changes by field, keeping only the earliest change for each field
        field_to_earliest_change = {}
        for change in self.change_history:
            if change.field_name not in field_to_earliest_change:
                field_to_earliest_change[change.field_name] = change

        # Revert each field to its original value
        for field, change in field_to_earliest_change.items():
            setattr(self, field, change.old_value)

        # Clear change history
        self.change_history = []

        # Update previous state
        self._store_current_state()

        return True


# Example usage with a user profile model
class UserProfile(ChangeTrackingMixin):
    """User profile model with change tracking."""
    username: str
    email: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    age: Optional[int] = None
    is_active: bool = True


def main():
    """Demonstrate the change tracking mixin."""
    # Create a user profile
    user = UserProfile(
        username="johndoe",
        email="john@example.com",
        display_name="John Doe"
    )
    print(f"Initial user profile: {user.model_dump_json(indent=2)}")

    # Make some changes
    changes = user.update(
        display_name="John D.",
        bio="Software developer",
        age=30
    )

    print("\nChanges made:")
    for change in changes:
        print(f"  {change.field_name}: {change.old_value} -> {change.new_value}")

    print("\nUpdated user profile:")
    print(user.model_dump_json(indent=2))

    # View change history
    print("\nComplete change history:")
    for i, change in enumerate(user.get_change_history()):
        print(f"  {i+1}. {change.field_name}: {change.old_value} -> {change.new_value} ({change.timestamp})")

    # View field history
    print("\nChange history for 'age' field:")
    for i, change in enumerate(user.get_field_history("age")):
        print(f"  {i+1}. {change.old_value} -> {change.new_value} ({change.timestamp})")

    # Make more changes
    user.update(
        display_name="John Doe",
        age=31
    )

    print("\nUser profile after more changes:")
    print(user.model_dump_json(indent=2))

    # Revert last change
    user.revert_last_change()
    print("\nUser profile after reverting last change:")
    print(user.model_dump_json(indent=2))

    # Revert all changes
    user.revert_all_changes()
    print("\nUser profile after reverting all changes:")
    print(user.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
