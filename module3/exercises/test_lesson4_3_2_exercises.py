"""
Tests for Lesson 4.3.2: Change Tracking Mixin
----------------------------------------------
This module contains tests for the change tracking mixin implementation.
"""

import unittest
from datetime import datetime
from typing import Optional, List

from module3.exercises.lesson4_3_2_exercises import (
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
            old_value="johndoe",
            new_value="john_doe"
        )
        self.assertEqual(record.field_name, "username")
        self.assertEqual(record.old_value, "johndoe")
        self.assertEqual(record.new_value, "john_doe")
        self.assertIsInstance(record.timestamp, datetime)
    
    def test_initialization(self):
        """Test initialization of a model with the mixin."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(len(user.get_change_history()), 0)
    
    def test_update_method(self):
        """Test the update method."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make changes
        changes = user.update(
            display_name="John Doe",
            bio="Software developer"
        )
        
        # Check that changes were recorded
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0].field_name, "display_name")
        self.assertEqual(changes[0].old_value, None)
        self.assertEqual(changes[0].new_value, "John Doe")
        self.assertEqual(changes[1].field_name, "bio")
        self.assertEqual(changes[1].old_value, None)
        self.assertEqual(changes[1].new_value, "Software developer")
        
        # Check that model was updated
        self.assertEqual(user.display_name, "John Doe")
        self.assertEqual(user.bio, "Software developer")
        
        # Check that change history was updated
        history = user.get_change_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].field_name, "display_name")
        self.assertEqual(history[1].field_name, "bio")
    
    def test_update_with_no_changes(self):
        """Test update with values that don't change."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Update with same values
        changes = user.update(
            username="johndoe",
            email="john@example.com"
        )
        
        # No changes should be recorded
        self.assertEqual(len(changes), 0)
        self.assertEqual(len(user.get_change_history()), 0)
    
    def test_get_field_history(self):
        """Test getting history for a specific field."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make multiple changes to the same field
        user.update(display_name="John")
        user.update(display_name="John Doe")
        user.update(display_name="J. Doe")
        
        # Check field history
        field_history = user.get_field_history("display_name")
        self.assertEqual(len(field_history), 3)
        self.assertEqual(field_history[0].old_value, None)
        self.assertEqual(field_history[0].new_value, "John")
        self.assertEqual(field_history[1].old_value, "John")
        self.assertEqual(field_history[1].new_value, "John Doe")
        self.assertEqual(field_history[2].old_value, "John Doe")
        self.assertEqual(field_history[2].new_value, "J. Doe")
    
    def test_revert_last_change(self):
        """Test reverting the last change."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make changes
        user.update(display_name="John Doe")
        user.update(age=30)
        
        # Revert last change
        result = user.revert_last_change()
        
        # Check that revert was successful
        self.assertTrue(result)
        self.assertEqual(user.age, None)
        self.assertEqual(user.display_name, "John Doe")
        
        # Check that change history was updated
        self.assertEqual(len(user.get_change_history()), 1)
        self.assertEqual(user.get_change_history()[0].field_name, "display_name")
    
    def test_revert_all_changes(self):
        """Test reverting all changes."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Make multiple changes
        user.update(display_name="John Doe")
        user.update(bio="Software developer")
        user.update(age=30)
        user.update(display_name="J. Doe")
        
        # Revert all changes
        result = user.revert_all_changes()
        
        # Check that revert was successful
        self.assertTrue(result)
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(user.display_name, None)
        self.assertEqual(user.bio, None)
        self.assertEqual(user.age, None)
        
        # Check that change history was cleared
        self.assertEqual(len(user.get_change_history()), 0)
    
    def test_revert_with_no_changes(self):
        """Test reverting when there are no changes."""
        user = UserProfile(
            username="johndoe",
            email="john@example.com"
        )
        
        # Try to revert with no changes
        result = user.revert_last_change()
        self.assertFalse(result)
        
        result = user.revert_all_changes()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
