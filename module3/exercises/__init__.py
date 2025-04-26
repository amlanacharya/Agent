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

try:
    from .lesson4_3_1_exercises import BaseUser, GuestUser, RegisteredUser, AdminUser
except ImportError:
    pass

try:
    from .lesson4_3_2_exercises import ChangeTrackingMixin, ChangeRecord, UserProfile
except ImportError:
    pass

try:
    from .lesson4_3_3_exercises import DBModelGenerator, DBTable, DBColumn
except ImportError:
    pass

try:
    from .lesson4_3_4_exercises import ModelAdapter, AdapterRegistry, CreateUserRequest, UserDB, UserResponse
except ImportError:
    pass

try:
    from .lesson4_3_5_exercises import FormDefinition, StringField, NumberField, BooleanField, SelectField, FieldType
except ImportError:
    pass
