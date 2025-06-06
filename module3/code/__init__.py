"""
Module 3 Code Examples
---------------------
This package contains code examples for Module 3: Data Validation & Structured Outputs with Pydantic.
"""

# Make key classes available at the package level
from .pydantic_basics import User, Product, AdvancedUser, SignupForm, TaskInput
from .output_parsers import PydanticOutputParser, StructuredOutputParser, llm_call, simulate_llm_call

# Import GroqClient if available
try:
    from .groq_client import GroqClient
except ImportError:
    pass
# Import validation patterns if the module exists
try:
    from .validation_patterns import DateRange, PasswordReset, Payment
except ImportError:
    pass
# Import error handling if the module exists
try:
    from .error_handling import (
        validate_user_input, validate_multiple_items,
        safe_parse_with_defaults, attempt_error_correction,
        user_friendly_errors, FormValidator
    )
except ImportError:
    pass
