"""
Standalone Demonstration Script for User Profile Validation
-------------------------------------------------------
This script demonstrates the UserProfile model and validation from the exercises.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
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
    """User profile model with validation rules."""
    name: str = Field(..., min_length=2, max_length=100)
    email: str
    age: Optional[int] = Field(None, ge=13, lt=120)
    bio: Optional[str] = Field(None, max_length=500)
    
    @field_validator('email')
    def validate_email(cls, v):
        """Ensure email contains @ symbol."""
        if '@' not in v:
            raise ValueError('Email must contain an @ symbol')
        return v


class UserProfileWithSkills(UserProfile):
    """Extended user profile with skills."""
    skills: List[Skill] = []
    years_experience: Optional[int] = Field(None, ge=0)
    
    @field_validator('skills')
    def validate_skills(cls, skills):
        """Validate that skill names are unique."""
        skill_names = [skill.name.lower() for skill in skills]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError('Skill names must be unique')
        return skills


def validate_user_data(user_data):
    """Validate user data against the appropriate model."""
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
            "fields_with_errors": extract_error_fields(str(e))
        }


def extract_error_fields(error_message):
    """Extract field names from a validation error message."""
    fields = []
    
    # Common patterns in error messages
    if "validation error for" in error_message.lower():
        parts = error_message.split("validation error for ")
        if len(parts) > 1:
            field_part = parts[1].split(" ")[0]
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
    
    return list(set(fields))  # Remove duplicates


def main():
    """Main demonstration function."""
    print("=== User Profile Validation Demonstration ===")
    
    # Basic user profile
    print("\n1. Basic User Profile")
    print("--------------------")
    
    # Valid user profile
    user = UserProfile(
        name="John Doe",
        email="john.doe@example.com",
        age=30,
        bio="A software developer with a passion for Python."
    )
    print(f"Valid user profile: {user}")
    
    # Minimal user profile
    minimal_user = UserProfile(name="Jane Smith", email="jane@example.com")
    print(f"Minimal user profile: {minimal_user}")
    
    # Invalid user profiles
    try:
        # Name too short
        user = UserProfile(name="J", email="j@example.com")
    except Exception as e:
        print(f"Validation error (name too short): {e}")
    
    try:
        # Invalid email
        user = UserProfile(name="John Doe", email="invalid-email")
    except Exception as e:
        print(f"Validation error (invalid email): {e}")
    
    # User profile with skills
    print("\n2. User Profile with Skills")
    print("-------------------------")
    
    # Valid user profile with skills
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
    
    # Invalid user profile with skills
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
    
    # Data validation function
    print("\n3. Data Validation Function")
    print("-------------------------")
    
    # Valid data
    valid_data = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 35,
        "bio": "Data scientist and machine learning enthusiast."
    }
    
    result = validate_user_data(valid_data)
    print(f"Validation result for valid data: {result}")
    
    # Invalid data
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
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
