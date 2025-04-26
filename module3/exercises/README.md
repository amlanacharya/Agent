# Module 3: Exercise Solutions 🧪

This directory contains solutions for the exercises in Module 3: Data Validation & Structured Outputs with Pydantic.

## 📁 Directory Structure

```
exercises/
├── lesson1_exercises.py                      # Solutions for Lesson 1 exercises
├── test_lesson1_exercises.py                 # Tests for Lesson 1 solutions
├── demo_lesson1_exercises.py                 # Demonstration script for Lesson 1 solutions
├── lesson2_exercises.py                      # Solutions for Lesson 2 exercises (Schema Design)
├── test_lesson2_exercises.py                 # Tests for Lesson 2 solutions
├── lesson3_exercises.py                      # Solutions for Lesson 3 exercises (Output Parsing)
├── test_lesson3_exercises.py                 # Tests for Lesson 3 solutions
├── lesson4_1_exercises.py                    # Solutions for Lesson 4.1 exercises (Cross-Field Validation)
├── test_lesson4_1_exercises.py               # Tests for Lesson 4.1 solutions
├── lesson4_2_exercises.py                    # Solutions for Lesson 4.2 exercises (Error Handling)
├── test_lesson4_2_exercises.py               # Tests for Lesson 4.2 solutions
├── exercise4.3.1_user_hierarchy.py           # Exercise 4.3.1: User Hierarchy with Inheritance
├── test_exercise4.3.1_user_hierarchy.py      # Tests for Exercise 4.3.1
├── exercise4.3.2_change_tracking_mixin.py    # Exercise 4.3.2: Change Tracking Mixin
├── test_exercise4.3.2_change_tracking_mixin.py # Tests for Exercise 4.3.2
├── exercise4.3.3_db_model_generator.py       # Exercise 4.3.3: Database Model Generator
├── test_exercise4.3.3_db_model_generator.py  # Tests for Exercise 4.3.3
├── exercise4.3.4_model_adapter_system.py     # Exercise 4.3.4: Model Adapter System
├── test_exercise4.3.4_model_adapter_system.py # Tests for Exercise 4.3.4
├── exercise4.3.5_form_builder.py             # Exercise 4.3.5: Form Builder
├── README.md                                 # This file
└── README_exercises4.3.md                    # Detailed README for Lesson 4.3 exercises
```

## 🚀 Getting Started

### Prerequisites

To run these examples, you'll need:

- Python 3.8 or higher
- Pydantic 2.0 or higher

Install the required packages:

```bash
pip install pydantic
```

### Running the Exercise Solutions

Each lesson has a corresponding demonstration script that shows the solutions in action:

```bash
# Run the Lesson 1 exercise solutions demonstration
python -m module3.exercises.demo_lesson1_exercises
```

### Running the Tests

Each solution has corresponding test files that verify the functionality:

```bash
# Run the tests for Lesson 1 solutions
python -m unittest module3.exercises.test_lesson1_exercises
```

## 📚 Exercise Solutions by Lesson

### Lesson 1: Pydantic Fundamentals

- `lesson1_exercises.py`: Solutions for exercises on basic Pydantic usage:
  - Creating a `UserProfile` model with validation rules
  - Extending the model to include a list of skills
  - Adding custom validators for email validation
  - Creating a function to validate user data against the model

### Lesson 2: Schema Design & Evolution

- `lesson2_exercises.py`: Solutions for exercises on schema design:
  - Implementing schema evolution strategies
  - Creating nested models for complex data
  - Implementing model composition patterns
  - Handling backward compatibility

### Lesson 3: Structured Output Parsing

- `lesson3_exercises.py`: Solutions for exercises on output parsing:
  - Creating a job application form parser
  - Implementing retry mechanisms for parsing failures
  - Adding custom validators for dates and emails
  - Creating a two-stage parsing approach for complex data

### Lesson 4: Advanced Validation Patterns

#### Lesson 4.1: Cross-Field Validation

- `lesson4_1_exercises.py`: Solutions for exercises on cross-field validation:
  - Implementing dependent field validation
  - Creating conditional validation rules
  - Handling complex validation scenarios
  - Implementing custom error messages

#### Lesson 4.2: Error Handling and Recovery

- `lesson4_2_exercises.py`: Solutions for exercises on error handling:
  - Implementing multi-step form validation
  - Creating suggestion systems for common errors
  - Building validation error logging
  - Implementing validation middleware
  - Creating partial submission forms with draft management

#### Lesson 4.3: Advanced Model Composition

- `exercise4.3.1_user_hierarchy.py`: User hierarchy with inheritance
  - Creating a model hierarchy for different types of users
  - Implementing type-specific functionality
  - Adding field validation with field_validator

- `exercise4.3.2_change_tracking_mixin.py`: Change tracking mixin
  - Implementing a mixin for tracking model changes
  - Recording previous and new values of fields
  - Adding reversion capabilities

- `exercise4.3.3_db_model_generator.py`: Database model generator
  - Creating Pydantic models from database table schemas
  - Mapping database types to Python types
  - Generating field constraints

- `exercise4.3.4_model_adapter_system.py`: Model adapter system
  - Converting between API request models, database models, and API response models
  - Implementing field mapping and transformation
  - Creating an adapter registry

- `exercise4.3.5_form_builder.py`: Form builder
  - Generating both Pydantic models and HTML form elements
  - Creating field type mapping and constraints
  - Implementing validator generation

## 🔍 Key Concepts

- **Model Design**: Creating well-structured data models
- **Validation Rules**: Implementing appropriate constraints
- **Custom Validators**: Adding domain-specific validation logic
- **Error Handling**: Gracefully handling validation failures
- **User Experience**: Providing helpful error messages
