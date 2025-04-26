"""
Demo script for the QualityValidator.

This script demonstrates how to use the QualityValidator to analyze
different types of text and interpret the results.
"""

import sys
import os
from typing import Dict, Any
import json
from colorama import init, Fore, Style

# Import the QualityValidator module
# Assuming the module is in the same directory as this test file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exercise4_5_5_quality_validator import (
    QualityValidator, QualityValidatorConfig, QualityDimension, QualityLevel
)

# Initialize colorama for colored console output
init()


def print_colored_score(name: str, score: float) -> None:
    """Print a colored score based on its value."""
    if score >= 0.8:
        color = Fore.GREEN
    elif score >= 0.6:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    
    print(f"{name}: {color}{score:.2f}{Style.RESET_ALL}")


def print_quality_report(title: str, text: str, context: Dict[str, Any] = None) -> None:
    """Print a quality report for the given text."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}QUALITY ANALYSIS: {title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Sample Text:{Style.RESET_ALL}")
    # Print only the first 300 characters of the text, with ellipsis if longer
    preview = text[:300] + ("..." if len(text) > 300 else "")
    print(preview)
    
    # Create validator and analyze text
    validator = QualityValidator()
    metrics, suggestions = validator.validate_with_suggestions(text, context)
    
    # Print summary
    print(f"\n{Fore.YELLOW}Quality Summary:{Style.RESET_ALL}")
    quality_level = metrics.overall_quality_level.value.upper()
    quality_color = Fore.GREEN if metrics.overall_quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD] else \
                   Fore.YELLOW if metrics.overall_quality_level == QualityLevel.ADEQUATE else Fore.RED
    
    print(f"Overall Quality: {quality_color}{quality_level}{Style.RESET_ALL} ({metrics.overall_score:.2f})")
    
    print(f"\n{Fore.YELLOW}Dimension Scores:{Style.RESET_ALL}")
    print_colored_score("Clarity", metrics.clarity_score)
    print_colored_score("Conciseness", metrics.conciseness_score)
    print_colored_score("Helpfulness", metrics.helpfulness_score)
    print_colored_score("Coherence", metrics.coherence_score)
    print_colored_score("Engagement", metrics.engagement_score)
    
    print(f"\n{Fore.YELLOW}Issues ({len(metrics.issues)}):{Style.RESET_ALL}")
    if metrics.issues:
        for i, issue in enumerate(metrics.issues, 1):
            severity_color = Fore.RED if issue.severity > 0.7 else \
                           Fore.YELLOW if issue.severity > 0.4 else Fore.WHITE
            
            print(f"{i}. {issue.dimension.value}: {severity_color}{issue.description}{Style.RESET_ALL}")
            if issue.suggestion:
                print(f"   {Fore.GREEN}Suggestion:{Style.RESET_ALL} {issue.suggestion}")
    else:
        print("No issues detected!")
    
    print(f"\n{Fore.YELLOW}Top Improvement Suggestions:{Style.RESET_ALL}")
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("No suggestions needed!")


def main():
    """Run the demo with various sample texts."""
    # Sample 1: Technical documentation (should score well)
    technical_doc = """
    # Pydantic Data Validation
    
    Pydantic provides data validation through Python type annotations. This guide will show you how to use it effectively.
    
    ## Basic Usage
    
    To use Pydantic, define your data models as classes that inherit from `BaseModel`:
    
    ```python
    from pydantic import BaseModel, Field
    
    class User(BaseModel):
        name: str = Field(..., min_length=1)
        age: int = Field(..., ge=0)
    ```
    
    This ensures that:
    - The `name` field must be a non-empty string
    - The `age` field must be a non-negative integer
    
    ## Validation in Action
    
    When you create a model instance, validation happens automatically:
    
    ```python
    # This works fine
    user = User(name="John", age=30)
    
    # This raises a ValidationError
    user = User(name="", age=-5)
    ```
    
    ## Error Handling
    
    When validation fails, Pydantic raises a `ValidationError` with detailed information:
    
    ```python
    try:
        user = User(name="", age=-5)
    except ValidationError as e:
        print(e.json())
    ```
    
    This helps you identify exactly what went wrong and fix it appropriately.
    
    Have questions about using Pydantic? Let me know in the comments!
    """
    
    # Sample 2: Very verbose and repetitive text (should score poorly on conciseness)
    verbose_text = """
    Pydantic is a validation library. It's a library used for validation in Python. The validation 
    happens with Python type annotations. Type annotations are a feature of Python that allows you 
    to specify types. The types are used by Pydantic for validation. The validation happens 
    automatically. You don't need to write validation code manually. Manual validation code would 
    be tedious to write. It would also be error-prone. That's why automatic validation with 
    Pydantic is helpful. It's very helpful indeed. Extremely helpful. You define models in Pydantic. 
    The models inherit from BaseModel. BaseModel is provided by Pydantic. The BaseModel class has 
    many helpful methods. These methods are automatically available to your models. Your models will 
    have these methods because they inherit from BaseModel. Inheritance is how you get these methods. 
    Inheritance is a feature of object-oriented programming. Python supports object-oriented 
    programming. That's why Pydantic can use inheritance in Python. Pydantic makes validation easy. 
    Validation is important for data integrity. Data integrity is crucial for applications. 
    Applications need valid data to function properly. Pydantic helps ensure data validity. That's 
    why Pydantic is useful. It's a useful library. Many people use it. Many applications use it too.
    """
    
    # Sample 3: Text with poor structure and coherence
    incoherent_text = """
    Pydantic validates data. Type annotations help. Python is typed. Models inherit BaseModel.
    
    ValidationError happens when invalid. JSON schema generation is possible. Export to JSON.
    Python 3.6+ required for Pydantic. Dataclasses are supported too. Types are important.
    
    Field has many parameters. Min length can be set. Max length too. Many other constraints.
    
    Config classes customize behavior. Advantages over dataclasses exist. Speed is good.
    DateTime parsing works well. List validation works. Dict validation too.
    """
    
    # Sample 4: Clear but not engaging text
    boring_text = """
    Pydantic provides data validation using Python type annotations. Models are defined as classes 
    inheriting from BaseModel. Fields are defined with type annotations. Additional constraints can 
    be added with the Field function. Validation occurs on instance creation. Invalid data raises
    ValidationError. Validated data can be accessed as attributes. Models can be converted to 
    dictionaries. They can also be converted to JSON. Configuration is done through Config class.
    Nested models are supported. List and dictionary field types are supported. Custom validators
    can be created with validator decorator. Field names can be aliased. Models can be made immutable.
    """
    
    # Sample 5: Unbalanced text (good coherence, poor clarity due to complexity)
    complex_text = """
    The implementation of object validation methodologies within the Pydantic framework leverages 
    the intrinsic type annotation capabilities inherent in Python's syntax to facilitate the automatic 
    verification of data structures against predefined constraints. The utilization of such mechanisms 
    obviates the necessity for manual validation code, thereby reducing the cognitive overhead 
    associated with ensuring data integrity. Furthermore, the inheritance-based architecture employed 
    by Pydantic, wherein user-defined models extend the functionality of the BaseModel class, enables 
    the propagation of validation behaviors across the object hierarchy. This paradigm significantly 
    enhances code modularity and reusability while maintaining a consistent approach to data validation.
    
    # Primary Components
    
    - Type Annotations
    - Field Constraints
    - Validation Execution
    
    # Error Management
    
    - Exception Propagation
    - Detailed Error Reporting
    """
    
    # Sample 6: Very short and unhelpful text
    unhelpful_text = """
    Pydantic is a Python library. It does validation. It's pretty good.
    You should try it if you need validation.
    """
    
    # Run the demo with all samples
    contexts = {
        "query": "How does Pydantic validation work?",
        "expected_topics": ["validation", "type hints", "error handling", "models"]
    }
    
    print_quality_report("Technical Documentation (Good Example)", technical_doc, contexts)
    print_quality_report("Verbose and Repetitive Text", verbose_text, contexts)
    print_quality_report("Poorly Structured Text", incoherent_text, contexts)
    print_quality_report("Clear but Not Engaging Text", boring_text, contexts)
    print_quality_report("Complex Language Text", complex_text, contexts)
    print_quality_report("Unhelpful Brief Text", unhelpful_text, contexts)
    
    # Custom configuration example
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}CUSTOM CONFIGURATION EXAMPLE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    # Create custom config that prioritizes clarity and helpfulness
    custom_config = QualityValidatorConfig(
        dimension_weights={
            QualityDimension.CLARITY: 0.4,        # Increased weight
            QualityDimension.HELPFULNESS: 0.3,    # Increased weight
            QualityDimension.CONCISENESS: 0.15,   # Decreased weight
            QualityDimension.COHERENCE: 0.1,      # Decreased weight
            QualityDimension.ENGAGEMENT: 0.05     # Decreased weight
        }
    )
    
    validator = QualityValidator(custom_config)
    metrics, suggestions = validator.validate_with_suggestions(technical_doc, contexts)
    
    print("\nStandard weights vs. Custom weights (prioritizing clarity and helpfulness):")
    print(f"Standard overall score: {0.85:.2f}")  # Example score from earlier run
    print(f"Custom weights score: {metrics.overall_score:.2f}")
    
    # Summary of the validator's capabilities
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}QUALITY VALIDATOR SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    print("""
The Quality Validator evaluates text across five dimensions:

1. Clarity - Readability, sentence complexity, word choice
2. Conciseness - Length, repetition, information density
3. Helpfulness - Relevance, actionable advice, examples
4. Coherence - Organization, logical flow, structure
5. Engagement - Tone, variety, connection with reader

For each dimension, specific metrics are calculated to determine scores.
Issues are identified when metrics fall below thresholds, and suggestions
for improvement are provided based on the most severe issues.

The validator can be configured to prioritize different dimensions based
on the specific requirements of your content.
    """)


if __name__ == "__main__":
    main()