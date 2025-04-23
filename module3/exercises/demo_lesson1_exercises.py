"""
Demonstration Script for Lesson 1 Exercise Solutions
------------------------------------------------
This script demonstrates the usage of the exercise solutions from Lesson 1.
"""

from module3.exercises.lesson1_exercises import (
    UserProfile,
    UserProfileWithSkills,
    Skill,
    ProficiencyLevel,
    validate_user_data
)


def demonstrate_user_profile():
    """Demonstrate the UserProfile model."""
    print("\n=== UserProfile Model ===")

    # Create a valid user profile
    user = UserProfile(
        name="John Doe",
        email="john.doe@example.com",
        age=30,
        bio="A software developer with a passion for Python."
    )
    print(f"Valid user profile: {user}")

    # Create a minimal user profile
    minimal_user = UserProfile(name="Jane Smith", email="jane@example.com")
    print(f"Minimal user profile: {minimal_user}")

    # Try creating invalid user profiles
    try:
        # Name too short
        user = UserProfile(name="J", email="j@example.com")
    except Exception as e:
        print(f"Validation error (name too short): {e}")

    try:
        # Invalid email (missing @)
        user = UserProfile(name="John Doe", email="invalid-email")
    except Exception as e:
        print(f"Validation error (invalid email): {e}")

    try:
        # Age too low
        user = UserProfile(name="John Doe", email="john@example.com", age=10)
    except Exception as e:
        print(f"Validation error (age too low): {e}")


def demonstrate_user_profile_with_skills():
    """Demonstrate the UserProfileWithSkills model."""
    print("\n=== UserProfileWithSkills Model ===")

    # Create a valid user profile with skills
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
    print(f"Valid user profile with skills: {user}")

    # Create a user profile with empty skills list
    user = UserProfileWithSkills(
        name="Bob Johnson",
        email="bob@example.com"
    )
    print(f"User profile with empty skills list: {user}")

    # Try creating invalid user profiles with skills
    try:
        # Duplicate skill names
        user = UserProfileWithSkills(
            name="Alice Brown",
            email="alice@example.com",
            skills=[
                {"name": "Python", "proficiency": "expert"},
                {"name": "python", "proficiency": "beginner"}  # Same name, different case
            ]
        )
    except Exception as e:
        print(f"Validation error (duplicate skill names): {e}")

    try:
        # Invalid proficiency level
        user = UserProfileWithSkills(
            name="Charlie Davis",
            email="charlie@example.com",
            skills=[
                {"name": "Python", "proficiency": "master"}  # Invalid level
            ]
        )
    except Exception as e:
        print(f"Validation error (invalid proficiency level): {e}")


def demonstrate_validate_user_data():
    """Demonstrate the validate_user_data function."""
    print("\n=== validate_user_data Function ===")

    # Valid basic profile
    valid_data = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 35,
        "bio": "Data scientist and machine learning enthusiast."
    }

    result = validate_user_data(valid_data)
    print(f"Validation result for valid basic profile: {result}")

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
    print(f"Validation result for valid profile with skills: {result}")

    # Invalid data - name too short
    invalid_data = {
        "name": "B",  # Too short
        "email": "b@example.com"
    }

    result = validate_user_data(invalid_data)
    print(f"Validation result for invalid data (name too short): {result}")

    # Invalid data - missing email
    invalid_data = {
        "name": "Charlie Davis"
        # Missing email
    }

    result = validate_user_data(invalid_data)
    print(f"Validation result for invalid data (missing email): {result}")

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
    print(f"Validation result for invalid data (multiple errors): {result}")


if __name__ == "__main__":
    print("=== Lesson 1 Exercise Solutions Demonstration ===")

    demonstrate_user_profile()
    demonstrate_user_profile_with_skills()
    demonstrate_validate_user_data()

    print("\nDemonstration complete!")
