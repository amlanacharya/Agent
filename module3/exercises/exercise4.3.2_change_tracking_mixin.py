"""
Exercise 4.3.2: Change Tracking Mixin

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
    _change_history: List[ChangeRecord] = Field(default_factory=list, exclude=True)
    _previous_state: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, **data):
        """Initialize the model and store the initial state."""
        super().__init__(**data)
        # Store initial state after initialization
        self._store_current_state()
    
    def _store_current_state(self):
        """Store the current state of the model."""
        # Get all fields except private ones (starting with _)
        self._previous_state = {
            field: copy.deepcopy(getattr(self, field))
            for field in self.model_fields
            if not field.startswith('_') and hasattr(self, field)
        }
    
    def update(self, **kwargs):
        """Update fields and track changes."""
        # Record changes
        changes = []
        for field, new_value in kwargs.items():
            if field in self.model_fields and hasattr(self, field):
                old_value = getattr(self, field)
                if old_value != new_value:
                    changes.append(ChangeRecord(
                        field_name=field,
                        old_value=old_value,
                        new_value=new_value
                    ))
        
        # Update fields
        for field, value in kwargs.items():
            if field in self.model_fields:
                setattr(self, field, value)
        
        # Add changes to history
        self._change_history.extend(changes)
        
        # Update previous state
        self._store_current_state()
        
        return changes
    
    def get_change_history(self) -> List[ChangeRecord]:
        """Get the complete change history."""
        return self._change_history
    
    def get_field_history(self, field_name: str) -> List[ChangeRecord]:
        """Get change history for a specific field."""
        return [
            change for change in self._change_history
            if change.field_name == field_name
        ]
    
    def revert_last_change(self) -> Optional[ChangeRecord]:
        """Revert the last change made to the model."""
        if not self._change_history:
            return None
        
        # Get the last change
        last_change = self._change_history[-1]
        
        # Revert the field to its previous value
        setattr(self, last_change.field_name, last_change.old_value)
        
        # Remove the change from history
        self._change_history.pop()
        
        # Update previous state
        self._store_current_state()
        
        return last_change
    
    def revert_all_changes(self):
        """Revert all changes to the initial state."""
        if not self._change_history:
            return
        
        # Get the initial state for each changed field
        initial_values = {}
        for field in set(change.field_name for change in self._change_history):
            # Find the first change for this field
            first_change = next(
                (change for change in self._change_history if change.field_name == field),
                None
            )
            if first_change:
                initial_values[field] = first_change.old_value
        
        # Apply the initial values
        for field, value in initial_values.items():
            setattr(self, field, value)
        
        # Clear change history
        self._change_history.clear()
        
        # Update previous state
        self._store_current_state()


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
    print("Initial user profile:")
    print(user.model_dump_json(indent=2))
    
    # Make some changes
    print("\nUpdating user profile...")
    changes = user.update(
        display_name="John D.",
        bio="Software developer from New York",
        age=30
    )
    
    print("\nChanges made:")
    for change in changes:
        print(f"  {change.field_name}: {change.old_value} -> {change.new_value}")
    
    print("\nUpdated user profile:")
    print(user.model_dump_json(indent=2))
    
    # Make more changes
    print("\nUpdating user profile again...")
    changes = user.update(
        email="john.doe@example.com",
        age=31
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
    
    # Revert last change
    print("\nReverting last change...")
    reverted = user.revert_last_change()
    print(f"Reverted: {reverted.field_name}: {reverted.new_value} -> {reverted.old_value}")
    
    print("\nUser profile after revert:")
    print(user.model_dump_json(indent=2))
    
    # Revert all changes
    print("\nReverting all changes...")
    user.revert_all_changes()
    
    print("\nUser profile after reverting all changes:")
    print(user.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
