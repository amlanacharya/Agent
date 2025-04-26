"""
Tests for Lesson 1 Exercise Solutions
----------------------------------
This module contains tests for the lesson1_exercises module.
"""

import unittest
from pydantic import ValidationError

from module3.exercises.lesson1_exercises import (
    UserProfile,
    UserProfileWithSkills,
    Skill,
    ProficiencyLevel,
    validate_user_data
)


class TestLesson1Exercises(unittest.TestCase):
    """Test cases for lesson1_exercises module."""

    def test_user_profile(self):
        """Test the UserProfile model."""
        # Valid user profile
        user = UserProfile(
            name="John Doe",
            email="john.doe@example.com",
            age=30,
            bio="A software developer with a passion for Python."
        )
        self.assertEqual(user.name, "John Doe")
        self.assertEqual(user.email, "john.doe@example.com")
        self.assertEqual(user.age, 30)
        self.assertEqual(user.bio, "A software developer with a passion for Python.")

        # Test with minimal fields
        user = UserProfile(name="Jane Smith", email="jane@example.com")
        self.assertEqual(user.name, "Jane Smith")
        self.assertEqual(user.email, "jane@example.com")
        self.assertIsNone(user.age)
        self.assertIsNone(user.bio)

        # Test name too short
        with self.assertRaises(ValidationError):
            UserProfile(name="J", email="j@example.com")

        # Test invalid email (missing @)
        with self.assertRaises(ValidationError):
            UserProfile(name="John Doe", email="invalid-email")

        # Test age too low
        with self.assertRaises(ValidationError):
            UserProfile(name="John Doe", email="john@example.com", age=10)

        # Test age too high
        with self.assertRaises(ValidationError):
            UserProfile(name="John Doe", email="john@example.com", age=150)

    def test_user_profile_with_skills(self):
        """Test the UserProfileWithSkills model."""
        # Valid user profile with skills
        user = UserProfileWithSkills(
            name="Jane Smith",
            email="jane.smith@example.com",
            age=28,
            bio="Full-stack developer and open source contributor.",
            skills=[
                {"name": "Python", "proficiency": "expert"},
                {"name": "JavaScript", "proficiency": "intermediate"},
                {"name": "Docker", "proficiency": "beginner"}
            ],
            years_experience=5
        )
        self.assertEqual(user.name, "Jane Smith")
        self.assertEqual(len(user.skills), 3)
        self.assertEqual(user.skills[0].name, "Python")
        self.assertEqual(user.skills[0].proficiency, ProficiencyLevel.EXPERT)
        self.assertEqual(user.years_experience, 5)

        # Test with empty skills list
        user = UserProfileWithSkills(
            name="Bob Johnson",
            email="bob@example.com"
        )
        self.assertEqual(user.skills, [])

        # Test with duplicate skill names
        with self.assertRaises(ValidationError):
            UserProfileWithSkills(
                name="Alice Brown",
                email="alice@example.com",
                skills=[
                    {"name": "Python", "proficiency": "expert"},
                    {"name": "python", "proficiency": "beginner"}  # Same name, different case
                ]
            )

        # Test with invalid proficiency level
        with self.assertRaises(ValidationError):
            UserProfileWithSkills(
                name="Charlie Davis",
                email="charlie@example.com",
                skills=[
                    {"name": "Python", "proficiency": "master"}  # Invalid level
                ]
            )

    def test_validate_user_data(self):
        """Test the validate_user_data function."""
        # Valid basic profile
        valid_data = {
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 35,
            "bio": "Data scientist and machine learning enthusiast."
        }

        result = validate_user_data(valid_data)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["name"], "Alice Johnson")
        self.assertEqual(result["model_used"], "UserProfile")

        # Valid profile with skills
        valid_data_with_skills = {
            "name": "Bob Smith",
            "email": "bob@example.com",
            "skills": [
                {"name": "Python", "proficiency": "expert"},
                {"name": "SQL", "proficiency": "intermediate"}
            ]
        }

        result = validate_user_data(valid_data_with_skills)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["name"], "Bob Smith")
        self.assertEqual(len(result["data"]["skills"]), 2)
        self.assertEqual(result["model_used"], "UserProfileWithSkills")

        # Invalid data - name too short
        invalid_data = {
            "name": "B",  # Too short
            "email": "b@example.com"
        }

        result = validate_user_data(invalid_data)
        self.assertEqual(result["status"], "error")
        # Check that there's an error message
        self.assertIn("name", result["message"].lower())

        # Invalid data - missing email
        invalid_data = {
            "name": "Charlie Davis"
            # Missing email
        }

        result = validate_user_data(invalid_data)
        self.assertEqual(result["status"], "error")
        self.assertIn("email", result["message"].lower())

        # Invalid data - multiple errors
        invalid_data = {
            "name": "D",  # Too short
            "email": "invalid-email",  # Missing @ symbol
            "age": 150,  # Too high
            "skills": [
                {"name": "Python", "proficiency": "expert"},
                {"name": "python", "proficiency": "beginner"}  # Duplicate skill name
            ]
        }

        result = validate_user_data(invalid_data)
        self.assertEqual(result["status"], "error")
        self.assertTrue(len(result["fields_with_errors"]) > 0)


if __name__ == "__main__":
    unittest.main()
