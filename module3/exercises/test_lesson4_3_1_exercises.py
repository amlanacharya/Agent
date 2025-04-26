"""
Tests for Lesson 4.3.1: User Hierarchy with Inheritance
--------------------------------------------------------
This module contains tests for the user hierarchy implementation.
"""

import unittest
from datetime import datetime, timedelta
from pydantic import ValidationError

from lesson4_3_1_exercises import (
    BaseUser,
    GuestUser,
    RegisteredUser,
    AdminUser
)


class TestUserHierarchy(unittest.TestCase):
    """Test cases for the user hierarchy implementation."""

    def test_base_user(self):
        """Test the BaseUser model."""
        # Valid base user
        user = BaseUser(
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        self.assertIsNotNone(user.id)
        self.assertEqual(user.ip_address, "192.168.1.1")

        # Invalid IP address
        with self.assertRaises(ValidationError):
            BaseUser(
                ip_address="invalid_ip",
                user_agent="Mozilla/5.0"
            )

        # Invalid IP address with octets out of range
        with self.assertRaises(ValidationError):
            BaseUser(
                ip_address="192.168.1.300",
                user_agent="Mozilla/5.0"
            )

    def test_guest_user(self):
        """Test the GuestUser model."""
        # Valid guest user
        guest = GuestUser(
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            session_id="sess_12345"
        )
        self.assertEqual(guest.visit_count, 1)

        # Test increment_visit method
        original_time = guest.last_active
        guest.increment_visit()
        self.assertEqual(guest.visit_count, 2)
        self.assertGreaterEqual(guest.last_active, original_time)

    def test_registered_user(self):
        """Test the RegisteredUser model."""
        # Valid registered user
        user = RegisteredUser(
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            username="johndoe",
            email="john@example.com"
        )
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertTrue(user.is_active)

        # Invalid username
        with self.assertRaises(ValidationError):
            RegisteredUser(
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                username="j",  # Too short
                email="john@example.com"
            )

        # Invalid email
        with self.assertRaises(ValidationError):
            RegisteredUser(
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                username="johndoe",
                email="invalid-email"
            )

        # Test update_login method
        original_login = user.last_login
        # Wait a tiny bit to ensure time difference
        user.update_login()
        self.assertGreaterEqual(user.last_login, original_login)

    def test_admin_user(self):
        """Test the AdminUser model."""
        # Valid admin user
        admin = AdminUser(
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            username="admin",
            email="admin@example.com",
            admin_level=2,
            permissions=["manage_users"]
        )
        self.assertEqual(admin.admin_level, 2)
        self.assertEqual(admin.permissions, ["manage_users"])
        self.assertFalse(admin.is_super_admin)

        # Invalid admin level
        with self.assertRaises(ValidationError):
            AdminUser(
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                username="admin",
                email="admin@example.com",
                admin_level=5  # Out of range
            )

        # Test grant_permission method
        admin.grant_permission("edit_content")
        self.assertIn("edit_content", admin.permissions)
        # Granting the same permission again shouldn't duplicate it
        admin.grant_permission("edit_content")
        self.assertEqual(admin.permissions.count("edit_content"), 1)

        # Test revoke_permission method
        admin.revoke_permission("manage_users")
        self.assertNotIn("manage_users", admin.permissions)
        # Revoking a non-existent permission should not raise an error
        admin.revoke_permission("non_existent")

        # Test promote method
        admin.promote()
        self.assertEqual(admin.admin_level, 3)
        self.assertTrue(admin.is_super_admin)
        # Cannot promote beyond level 3
        admin.promote()
        self.assertEqual(admin.admin_level, 3)

        # Test demote method
        admin.demote()
        self.assertEqual(admin.admin_level, 2)
        self.assertFalse(admin.is_super_admin)
        admin.demote()
        self.assertEqual(admin.admin_level, 1)
        # Cannot demote below level 1
        admin.demote()
        self.assertEqual(admin.admin_level, 1)


if __name__ == "__main__":
    unittest.main()
