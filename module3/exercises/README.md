# 🧪 Module 3: Exercise Solutions

## 📚 Overview

This directory contains solutions for the exercises in Module 3: Data Validation & Structured Outputs with Pydantic. These exercises are designed to reinforce the concepts covered in the lessons and provide hands-on practice with implementing data validation and structured output parsing.

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
├── lesson4_3_1_exercises.py                  # Lesson 4.3.1: User Hierarchy with Inheritance
├── test_lesson4_3_1_exercises.py             # Tests for Lesson 4.3.1
├── lesson4_3_2_exercises.py                  # Lesson 4.3.2: Change Tracking Mixin
├── test_lesson4_3_2_exercises.py             # Tests for Lesson 4.3.2
├── lesson4_3_3_exercises.py                  # Lesson 4.3.3: Database Model Generator
├── test_lesson4_3_3_exercises.py             # Tests for Lesson 4.3.3
├── lesson4_3_4_exercises.py                  # Lesson 4.3.4: Model Adapter System
├── test_lesson4_3_4_exercises.py             # Tests for Lesson 4.3.4
├── lesson4_3_5_exercises.py                  # Lesson 4.3.5: Form Builder
└── README_lesson4_3.md                       # Detailed README for Lesson 4.3 exercises
```

## 🔍 Exercise Descriptions

### Lesson 1: Pydantic Fundamentals

- **lesson1_exercises.py**: Solutions for exercises on basic Pydantic usage:
  - 📋 Creating a `UserProfile` model with validation rules
  - 🔒 Extending the model to include a list of skills
  - 🧩 Adding custom validators for email validation
  - 🔄 Creating a function to validate user data against the model

### Lesson 2: Schema Design & Evolution

- **lesson2_exercises.py**: Solutions for exercises on schema design:
  - 📊 Implementing schema evolution strategies
  - 🔄 Creating nested models for complex data
  - 🧩 Implementing model composition patterns
  - 📝 Handling backward compatibility

### Lesson 3: Structured Output Parsing

- **lesson3_exercises.py**: Solutions for exercises on output parsing:
  - 📄 Creating a job application form parser
  - 🛡️ Implementing retry mechanisms for parsing failures
  - 🔍 Adding custom validators for dates and emails
  - 🧠 Creating a two-stage parsing approach for complex data

### Lesson 4: Advanced Validation Patterns

#### Lesson 4.1: Cross-Field Validation

- **lesson4_1_exercises.py**: Solutions for exercises on cross-field validation:
  - ✅ Implementing dependent field validation
  - 🔄 Creating conditional validation rules
  - 🧩 Handling complex validation scenarios
  - 📊 Implementing custom error messages

#### Lesson 4.2: Error Handling and Recovery

- **lesson4_2_exercises.py**: Solutions for exercises on error handling:
  - 📋 Implementing multi-step form validation
  - 🔍 Creating suggestion systems for common errors
  - 📊 Building validation error logging
  - 🛡️ Implementing validation middleware
  - 🧩 Creating partial submission forms with draft management

#### Lesson 4.3: Advanced Model Composition

- **lesson4_3_1_exercises.py**: User hierarchy with inheritance:
  - 👤 Creating a model hierarchy for different types of users
  - 🔄 Implementing type-specific functionality
  - ✅ Adding field validation with field_validator

- **lesson4_3_2_exercises.py**: Change tracking mixin:
  - 📝 Implementing a mixin for tracking model changes
  - 🔄 Recording previous and new values of fields
  - 🧩 Adding reversion capabilities

- **lesson4_3_3_exercises.py**: Database model generator:
  - 🗃️ Creating Pydantic models from database table schemas
  - 🔄 Mapping database types to Python types
  - 🔒 Generating field constraints

- **lesson4_3_4_exercises.py**: Model adapter system:
  - 🔄 Converting between API request models, database models, and API response models
  - 🧩 Implementing field mapping and transformation
  - 📊 Creating an adapter registry

- **lesson4_3_5_exercises.py**: Form builder:
  - 📋 Generating both Pydantic models and HTML form elements
  - 🔄 Creating field type mapping and constraints
  - ✅ Implementing validator generation

## 🚀 Running the Exercises

You can run any of the exercise solutions directly from the command line:

```bash
# Run from the project root
python -m module3.exercises.lesson1_exercises
python -m module3.exercises.lesson2_exercises
python -m module3.exercises.lesson3_exercises
python -m module3.exercises.lesson4_1_exercises
python -m module3.exercises.lesson4_2_exercises
```

To run the tests:

```bash
# Run from the project root
python -m module3.exercises.test_lesson1_exercises
python -m module3.exercises.test_lesson2_exercises
python -m module3.exercises.test_lesson3_exercises
python -m module3.exercises.test_lesson4_1_exercises
python -m module3.exercises.test_lesson4_2_exercises
```

To run the interactive demos:

```bash
# Run from the project root
python -m module3.exercises.demo_lesson1_exercises
```

## 📝 Exercise Completion Checklist

- [ ] Lesson 1 Exercises
  - [ ] UserProfile Model Creation
  - [ ] Skills List Extension
  - [ ] Email Validator Implementation
  - [ ] Data Validation Function
- [ ] Lesson 2 Exercises
  - [ ] Schema Evolution Strategy
  - [ ] Nested Models for Complex Data
  - [ ] Model Composition Patterns
  - [ ] Backward Compatibility Handling
- [ ] Lesson 3 Exercises
  - [ ] Job Application Form Parser
  - [ ] Retry Mechanism Implementation
  - [ ] Custom Date and Email Validators
  - [ ] Two-Stage Parsing Approach
- [ ] Lesson 4.1 Exercises
  - [ ] Dependent Field Validation
  - [ ] Conditional Validation Rules
  - [ ] Complex Validation Scenarios
  - [ ] Custom Error Messages
- [ ] Lesson 4.2 Exercises
  - [ ] Multi-Step Form Validation
  - [ ] Suggestion System for Errors
  - [ ] Validation Error Logging
  - [ ] Validation Middleware
  - [ ] Partial Submission Forms
- [ ] Lesson 4.3 Exercises
  - [ ] User Hierarchy with Inheritance
  - [ ] Change Tracking Mixin
  - [ ] Database Model Generator
  - [ ] Model Adapter System
  - [ ] Form Builder

## 🧠 Learning Outcomes

By completing these exercises, you will:
- 🔍 Understand how to implement robust data validation with Pydantic
- 🧩 Master the creation of flexible and maintainable schema designs
- 🔄 Learn how to parse and validate structured outputs from LLMs
- 📊 Develop skills in implementing advanced validation patterns
- 🛠️ Practice building complete validation systems for real-world applications

## 🤔 Need Help?

If you get stuck on any exercise:
- Review the relevant lesson material
- Check the test files for expected behavior
- Experiment with different approaches
- Compare your solution with the provided examples
- Refer to the Pydantic documentation for specific features
