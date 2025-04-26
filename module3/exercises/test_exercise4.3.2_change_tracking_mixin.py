"""
Tests for Exercise 4.3.2: Change Tracking Mixin
----------------------------------------------
This module contains tests for the change tracking mixin implementation.
"""

import unittest
from datetime import datetime
from typing import Optional, List

from exercise4.3.2_change_tracking_mixin import (
    ChangeRecord,
    ChangeTrackingMixin,
    UserProfile
)


class TestChangeTrackingMixin(unittest.TestCase):
    """Test cases for the change tracking mixin implementation."""
    
    def test_change_record(self):
        """Test the ChangeRecord model."""
        record = ChangeRecord(
            field_name="username",
            old_value="old_user",
            new_value="new_user"
        )
        self.assertEqual(record.field_name, "username")
        self.assertEqual(record.old_value, "old_user")
        self.assertEqual(record.new_value, "new_user")
        self.assertIsInstance(record.timestamp, datetime)
    
    def test_initial_state(self):
        """Test the initial state of a model with the mixin."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(len(user.get_change_history()), 0)
        self.assertEqual(user._previous_state["username"], "johndoe")
        self.assertEqual(user._previous_state["email"], "john@example.com")
    
    def test_update_with_changes(self):
        """Test updating fields with changes."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Update with changes
        changes = user.update(
            display_name="John Doe",
            bio="A software developer"
        )
        
        # Check that changes were recorded
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0].field_name, "display_name")
        self.assertEqual(changes[0].old_value, None)
        self.assertEqual(changes[0].new_value, "John Doe")
        self.assertEqual(changes[1].field_name, "bio")
        self.assertEqual(changes[1].old_value, None)
        self.assertEqual(changes[1].new_value, "A software developer")
        
        # Check that change history was updated
        history = user.get_change_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].field_name, "display_name")
        self.assertEqual(history[1].field_name, "bio")
        
        # Check that previous state was updated
        self.assertEqual(user._previous_state["display_name"], "John Doe")
        self.assertEqual(user._previous_state["bio"], "A software developer")
    
    def test_update_without_changes(self):
        """Test updating fields without actual changes."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Update with the same values
        changes = user.update(
            username="johndoe",
            email="john@example.com"
        )
        
        # Check that no changes were recorded
        self.assertEqual(len(changes), 0)
        self.assertEqual(len(user.get_change_history()), 0)
    
    def test_get_field_history(self):
        """Test getting history for a specific field."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make multiple changes to the same field
        user.update(email="john.doe@example.com")
        user.update(email="johndoe@company.com")
        
        # Check field history
        email_history = user.get_field_history("email")
        self.assertEqual(len(email_history), 2)
        self.assertEqual(email_history[0].old_value, "john@example.com")
        self.assertEqual(email_history[0].new_value, "john.doe@example.com")
        self.assertEqual(email_history[1].old_value, "john.doe@example.com")
        self.assertEqual(email_history[1].new_value, "johndoe@company.com")
    
    def test_revert_last_change(self):
        """Test reverting the last change."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make multiple changes
        user.update(email="john.doe@example.com")
        user.update(display_name="John Doe")
        
        # Revert last change
        reverted = user.revert_last_change()
        
        # Check that the last change was reverted
        self.assertEqual(reverted.field_name, "display_name")
        self.assertEqual(user.display_name, None)
        self.assertEqual(len(user.get_change_history()), 1)
        
        # Revert another change
        reverted = user.revert_last_change()
        self.assertEqual(reverted.field_name, "email")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(len(user.get_change_history()), 0)
        
        # Try to revert when there are no changes
        reverted = user.revert_last_change()
        self.assertIsNone(reverted)
    
    def test_revert_all_changes(self):
        """Test reverting all changes."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make multiple changes to different fields
        user.update(
            email="john.doe@example.com",
            display_name="John Doe",
            bio="A developer",
            age=30
        )
        
        # Make more changes
        user.update(
            email="johndoe@company.com",
            age=31
        )
        
        # Revert all changes
        user.revert_all_changes()
        
        # Check that all fields are back to their initial values
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertIsNone(user.display_name)
        self.assertIsNone(user.bio)
        self.assertIsNone(user.age)
        self.assertEqual(len(user.get_change_history()), 0)
    
    def test_complex_object_changes(self):
        """Test tracking changes to complex objects like lists and dicts."""
        # Create a model with complex fields
        class ComplexModel(ChangeTrackingMixin):
            tags: List[str] = []
            metadata: dict = {}
        
        model = ComplexModel(
            tags=["tag1", "tag2"],
            metadata={"key1": "value1"}
        )
        
        # Update complex fields
        model.update(
            tags=["tag1", "tag2", "tag3"],
            metadata={"key1": "value1", "key2": "value2"}
        )
        
        # Check that changes were recorded correctly
        history = model.get_change_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].old_value, ["tag1", "tag2"])
        self.assertEqual(history[0].new_value, ["tag1", "tag2", "tag3"])
        self.assertEqual(history[1].old_value, {"key1": "value1"})
        self.assertEqual(history[1].new_value, {"key1": "value1", "key2": "value2"})
        
        # Revert changes
        model.revert_all_changes()
        self.assertEqual(model.tags, ["tag1", "tag2"])
        self.assertEqual(model.metadata, {"key1": "value1"})


if __name__ == "__main__":
    unittest.main()
