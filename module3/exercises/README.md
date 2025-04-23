# Module 3: Exercise Solutions 🧪

This directory contains solutions for the exercises in Module 3: Data Validation & Structured Outputs with Pydantic.

## 📁 Directory Structure

```
exercises/
├── lesson1_exercises.py         # Solutions for Lesson 1 exercises
├── test_lesson1_exercises.py    # Tests for Lesson 1 solutions
├── demo_lesson1_exercises.py    # Demonstration script for Lesson 1 solutions
├── lesson2_exercises.py         # Solutions for Lesson 2 exercises (Schema Design)
├── lesson3_exercises.py         # Solutions for Lesson 3 exercises (Output Parsing)
├── lesson4_exercises.py         # Solutions for Lesson 4 exercises (Advanced Validation)
└── README.md                    # This file
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

- `lesson4_exercises.py`: Solutions for exercises on advanced validation:
  - Implementing dependent field validation
  - Creating conditional validation rules
  - Handling complex validation scenarios
  - Implementing custom error messages

## 🔍 Key Concepts

- **Model Design**: Creating well-structured data models
- **Validation Rules**: Implementing appropriate constraints
- **Custom Validators**: Adding domain-specific validation logic
- **Error Handling**: Gracefully handling validation failures
- **User Experience**: Providing helpful error messages
