"""
Module 3 Exercises
----------------
This package contains exercise solutions for Module 3: Data Validation & Structured Outputs with Pydantic.
"""

# Make solution classes available at the package level
try:
    from .lesson1_exercises import UserProfile, UserProfileWithSkills, validate_user_data
except ImportError:
    pass

try:
    from .lesson2_exercises import SchemaEvolution, ConfigModel, NestedModel
except ImportError:
    pass

try:
    from .lesson3_exercises import OutputParser, JobApplicationParser
except ImportError:
    pass

try:
    from .lesson4_1_exercises import PaymentSystem, TravelBooking, ProductInventory, SurveyForm
except ImportError:
    pass
