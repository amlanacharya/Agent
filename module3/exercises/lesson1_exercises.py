"""
Lesson 1 Exercise Solutions
-------------------------
This module contains solutions for the exercises in Lesson 1: Pydantic Fundamentals.
"""

from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import List, Optional, Dict, Any
from enum import Enum


class ProficiencyLevel(str, Enum):
    """Enum for skill proficiency levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Skill(BaseModel):
    """Model for a skill with name and proficiency level."""
    name: str = Field(..., min_length=2, max_length=50)
    proficiency: ProficiencyLevel = ProficiencyLevel.BEGINNER


class UserProfile(BaseModel):
    """
    Exercise 1: Create a UserProfile model with fields for name, email, age, and bio.
    Add appropriate validation rules.
    """
    name: str = Field(..., min_length=2, max_length=100)
    email: str
    age: Optional[int] = Field(None, ge=13, lt=120)
    bio: Optional[str] = Field(None, max_length=500)

    @field_validator('email')
    def validate_email(cls, v):
        """
        Exercise 3: Add a custom validator that ensures the email field contains an @ symbol.
        """
        if '@' not in v:
            raise ValueError('Email must contain an @ symbol')
        return v


class UserProfileWithSkills(UserProfile):
    """
    Exercise 2: Extend the model to include a list of skills,
    where each skill has a name and proficiency level.
    """
    skills: List[Skill] = []
    years_experience: Optional[int] = Field(None, ge=0)

    @field_validator('skills')
    def validate_skills(cls, skills):
        """Validate that skill names are unique."""
        skill_names = [skill.name.lower() for skill in skills]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError('Skill names must be unique')
        return skills


def validate_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exercise 4: Create a function that takes a dictionary of user data,
    validates it against your model, and handles any validation errors gracefully.

    Args:
        user_data: Dictionary containing user profile data

    Returns:
        Dictionary with validation result and either the validated data or error messages
    """
    try:
        # Try to validate against the UserProfileWithSkills model first
        if 'skills' in user_data:
            user = UserProfileWithSkills(**user_data)
            return {
                "status": "success",
                "data": user.model_dump(),
                "model_used": "UserProfileWithSkills"
            }
        # If no skills provided, use the basic UserProfile model
        else:
            user = UserProfile(**user_data)
            return {
                "status": "success",
                "data": user.model_dump(),
                "model_used": "UserProfile"
            }
    except Exception as e:
        # Handle validation errors
        return {
            "status": "error",
            "message": str(e),
            "fields_with_errors": _extract_error_fields(str(e))
        }


def _extract_error_fields(error_message: str) -> List[str]:
    """
    Extract field names from a Pydantic validation error message.

    Args:
        error_message: The error message string

    Returns:
        List of field names that had validation errors
    """
    # This is a simple implementation - in a real application,
    # you would parse the ValidationError object directly
    fields = []

    # Common patterns in error messages
    if "validation error for" in error_message.lower():
        # Extract field name after "validation error for"
        parts = error_message.split("validation error for ")
        if len(parts) > 1:
            field_part = parts[1].split("\n")[0]  # Get everything up to the newline
            # Extract just the field name without the model name
            if field_part.strip().endswith(':'):
                field_part = field_part.strip()[:-1]  # Remove the colon

            # If it contains a model name, extract just the field
            if '\n' in field_part:
                field_part = field_part.split('\n')[0]

            # Clean up any remaining model name
            if ' ' in field_part:
                field_part = field_part.split(' ')[-1]

            fields.append(field_part.strip())

    # Look for field names in brackets
    import re
    bracket_fields = re.findall(r'\[\'([^\']+)\'\]', error_message)
    fields.extend(bracket_fields)

    # If we couldn't extract specific fields, check for common field names
    if not fields:
        common_fields = ["name", "email", "age", "bio", "skills", "proficiency"]
        for field in common_fields:
            if field in error_message.lower():
                fields.append(field)

    # Clean up field names - remove any model prefixes
    cleaned_fields = []
    for field in fields:
        if '\n' in field:
            parts = field.split('\n')
            cleaned_fields.append(parts[-1])
        else:
            cleaned_fields.append(field)

    # Further clean up to handle "UserProfile\nname" format
    final_fields = []
    for field in cleaned_fields:
        if '\\n' in field:
            final_fields.append(field.split('\\n')[-1])
        else:
            final_fields.append(field)

    return list(set(final_fields))  # Remove duplicates


if __name__ == "__main__":
    # Test the UserProfile model
    try:
        user = UserProfile(
            name="John Doe",
            email="john.doe@example.com",
            age=30,
            bio="A software developer with a passion for Python."
        )
        print(f"Valid user profile: {user}")
    except Exception as e:
        print(f"Validation error: {e}")

    # Test the UserProfileWithSkills model
    try:
        user_with_skills = UserProfileWithSkills(
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
        print(f"Valid user profile with skills: {user_with_skills}")
    except Exception as e:
        print(f"Validation error: {e}")

    # Test the validate_user_data function
    valid_data = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 35,
        "bio": "Data scientist and machine learning enthusiast."
    }

    result = validate_user_data(valid_data)
    print(f"Validation result for valid data: {result}")

    # Test with invalid data
    invalid_data = {
        "name": "B",  # Too short
        "email": "invalid-email",  # Missing @ symbol
        "age": 150,  # Too high
        "skills": [
            {"name": "Python", "proficiency": "expert"},
            {"name": "python", "proficiency": "beginner"}  # Duplicate skill name
        ]
    }

    result = validate_user_data(invalid_data)
    print(f"Validation result for invalid data: {result}")
