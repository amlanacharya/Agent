"""
Module 3 Code Examples
---------------------
This package contains code examples for Module 3: Data Validation & Structured Outputs with Pydantic.
"""

# Make key classes available at the package level
from .pydantic_basics import User, Product, AdvancedUser, SignupForm, TaskInput
from .output_parsers import PydanticOutputParser, StructuredOutputParser
# Import validation patterns if the module exists
try:
    from .validation_patterns import DateRange, PasswordReset, Payment
except ImportError:
    pass
