"""
Demo script for Exercise 4.6.1: Multi-Step Form Validator

This script demonstrates how to use the MultiStepForm class to create
and manage a multi-step form with validation.
"""

import sys
import os
from typing import Dict, Any, List
from colorama import init, Fore, Style

# Initialize colorama
init()

# Import the MultiStepForm module
from exercise4_6_1_multi_step_form_validator import (
    FormStepStatus,
    FieldType,
    ValidationRule,
    FormField,
    FormStepDefinition,
    ValidationError,
    FormStepState,
    MultiStepForm
)


def print_header(text: str) -> None:
    """Print a header with formatting."""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text.center(80)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")


def print_subheader(text: str) -> None:
    """Print a subheader with formatting."""
    print(f"\n{Fore.YELLOW}{text}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * len(text)}{Style.RESET_ALL}")


def print_step_info(form: MultiStepForm) -> None:
    """Print information about the current step."""
    current_step = form.get_current_step()
    current_state = form.get_current_step_state()

    print(f"\n{Fore.GREEN}Current Step: {current_step.title} ({current_step.step_id}){Style.RESET_ALL}")
    print(f"Status: {current_state.status.value}")

    if current_step.description:
        print(f"Description: {current_step.description}")

    if current_step.fields:
        print("\nFields:")
        for field in current_step.fields:
            required = f"{Fore.RED}*{Style.RESET_ALL}" if field.required else ""
            print(f"  - {field.label}{required} ({field.field_type.value})")
            if field.help_text:
                print(f"    Help: {field.help_text}")

    if current_state.data:
        print("\nCurrent Data:")
        for key, value in current_state.data.items():
            print(f"  {key}: {value}")

    if current_state.errors:
        print(f"\n{Fore.RED}Validation Errors:{Style.RESET_ALL}")
        for error in current_state.errors:
            print(f"  {error.field}: {error.error}")

    print("\nNavigation Options:")
    if current_step.previous_step:
        print(f"  - Previous: {current_step.previous_step}")
    if current_step.next_steps:
        print(f"  - Next: {', '.join(current_step.next_steps)}")


def print_form_summary(form: MultiStepForm) -> None:
    """Print a summary of the form state."""
    print_subheader("Form Summary")

    print(f"Title: {form.title}")
    if form.description:
        print(f"Description: {form.description}")

    print(f"\nSteps: {len(form.steps)}")
    for step_id, state in form.step_states.items():
        status_color = Fore.GREEN if state.status == FormStepStatus.COMPLETED else \
                      Fore.YELLOW if state.status == FormStepStatus.IN_PROGRESS else \
                      Fore.RED if state.status == FormStepStatus.INVALID else \
                      Fore.WHITE

        print(f"  - {step_id}: {status_color}{state.status.value}{Style.RESET_ALL}")

    print(f"\nForm Complete: {Fore.GREEN if form.is_form_complete() else Fore.RED}{form.is_form_complete()}{Style.RESET_ALL}")
    print(f"Form Submitted: {Fore.GREEN if form.is_submitted else Fore.RED}{form.is_submitted}{Style.RESET_ALL}")


def create_registration_form() -> MultiStepForm:
    """Create a sample registration form."""
    # Personal Information Step
    personal_info_step = FormStepDefinition(
        step_id="personal_info",
        title="Personal Information",
        description="Please provide your personal details",
        fields=[
            FormField(
                name="first_name",
                label="First Name",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_length(2),
                    ValidationRule.max_length(50)
                ],
                help_text="Your legal first name"
            ),
            FormField(
                name="last_name",
                label="Last Name",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_length(2),
                    ValidationRule.max_length(50)
                ],
                help_text="Your legal last name"
            ),
            FormField(
                name="email",
                label="Email Address",
                field_type=FieldType.EMAIL,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.email()
                ],
                help_text="We'll use this to contact you"
            ),
            FormField(
                name="age",
                label="Age",
                field_type=FieldType.NUMBER,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.min_value(18),
                    ValidationRule.max_value(120)
                ],
                help_text="You must be at least 18 years old"
            )
        ],
        next_steps=["address"]
    )

    # Address Step
    address_step = FormStepDefinition(
        step_id="address",
        title="Address Information",
        description="Please provide your address details",
        fields=[
            FormField(
                name="street",
                label="Street Address",
                field_type=FieldType.TEXT,
                validation_rules=[ValidationRule.required()],
                help_text="Street number and name"
            ),
            FormField(
                name="city",
                label="City",
                field_type=FieldType.TEXT,
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="state",
                label="State/Province",
                field_type=FieldType.SELECT,
                options=[
                    {"value": "CA", "label": "California"},
                    {"value": "NY", "label": "New York"},
                    {"value": "TX", "label": "Texas"},
                    {"value": "FL", "label": "Florida"},
                    {"value": "IL", "label": "Illinois"}
                ],
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="zip_code",
                label="ZIP/Postal Code",
                field_type=FieldType.TEXT,
                validation_rules=[
                    ValidationRule.required(),
                    ValidationRule.pattern(r'^\d{5}(-\d{4})?$', "Invalid ZIP code format")
                ],
                help_text="5-digit ZIP code or ZIP+4"
            )
        ],
        next_steps=["preferences"],
        previous_step="personal_info"
    )

    # Preferences Step
    preferences_step = FormStepDefinition(
        step_id="preferences",
        title="Preferences",
        description="Please tell us about your preferences",
        fields=[
            FormField(
                name="interests",
                label="Interests",
                field_type=FieldType.CHECKBOX,
                options=[
                    {"value": "tech", "label": "Technology"},
                    {"value": "sports", "label": "Sports"},
                    {"value": "music", "label": "Music"},
                    {"value": "travel", "label": "Travel"},
                    {"value": "food", "label": "Food & Cooking"},
                    {"value": "art", "label": "Art & Design"}
                ],
                required=False,
                help_text="Select all that apply"
            ),
            FormField(
                name="contact_method",
                label="Preferred Contact Method",
                field_type=FieldType.RADIO,
                options=[
                    {"value": "email", "label": "Email"},
                    {"value": "phone", "label": "Phone"},
                    {"value": "mail", "label": "Mail"}
                ],
                validation_rules=[ValidationRule.required()]
            ),
            FormField(
                name="comments",
                label="Additional Comments",
                field_type=FieldType.TEXTAREA,
                required=False,
                help_text="Tell us anything else you'd like us to know"
            )
        ],
        next_steps=["review"],
        previous_step="address"
    )

    # Review Step
    review_step = FormStepDefinition(
        step_id="review",
        title="Review",
        description="Please review your information before submitting",
        fields=[],  # No input fields on review step
        previous_step="preferences"
    )

    # Create the multi-step form
    return MultiStepForm(
        title="User Registration",
        description="Complete this form to register for our service",
        steps={
            "personal_info": personal_info_step,
            "address": address_step,
            "preferences": preferences_step,
            "review": review_step
        },
        start_step_id="personal_info"
    )


def demo_happy_path() -> None:
    """Demonstrate the happy path through the form."""
    print_header("Multi-Step Form Validator - Happy Path Demo")

    print("This demo shows a successful path through a multi-step registration form.")

    # Create the form
    form = create_registration_form()

    # Step 1: Personal Information
    print_step_info(form)

    print_subheader("Filling out Personal Information")
    personal_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30
    }

    print(f"Submitting data: {personal_data}")
    result = form.update_step_data("personal_info", personal_data)

    if result["success"] and not result["errors"]:
        print(f"{Fore.GREEN}✓ Personal information accepted{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Personal information has errors{Style.RESET_ALL}")
        for error in result["errors"]:
            print(f"  - {error.field}: {error.error}")

    # Complete step and move to next
    form.complete_step("personal_info")
    form.next_step()

    # Step 2: Address
    print_step_info(form)

    print_subheader("Filling out Address Information")
    address_data = {
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip_code": "94105"
    }

    print(f"Submitting data: {address_data}")
    result = form.update_step_data("address", address_data)

    if result["success"] and not result["errors"]:
        print(f"{Fore.GREEN}✓ Address information accepted{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Address information has errors{Style.RESET_ALL}")
        for error in result["errors"]:
            print(f"  - {error.field}: {error.error}")

    # Complete step and move to next
    form.complete_step("address")
    form.next_step()

    # Step 3: Preferences
    print_step_info(form)

    print_subheader("Filling out Preferences")
    preferences_data = {
        "interests": ["tech", "travel", "food"],
        "contact_method": "email",
        "comments": "Looking forward to using your service!"
    }

    print(f"Submitting data: {preferences_data}")
    result = form.update_step_data("preferences", preferences_data)

    if result["success"] and not result["errors"]:
        print(f"{Fore.GREEN}✓ Preferences accepted{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Preferences have errors{Style.RESET_ALL}")
        for error in result["errors"]:
            print(f"  - {error.field}: {error.error}")

    # Complete step and move to next
    form.complete_step("preferences")
    form.next_step()

    # Step 4: Review
    print_step_info(form)

    print_subheader("Reviewing Form Data")
    all_data = form.get_form_data()

    for key, value in all_data.items():
        print(f"  {key}: {value}")

    # Complete the review step
    print_subheader("Completing Review Step")
    form.complete_step("review")
    print(f"{Fore.GREEN}✓ Review step completed{Style.RESET_ALL}")

    # Submit the form
    print_subheader("Submitting Form")
    result = form.submit_form()

    if result["success"]:
        print(f"{Fore.GREEN}✓ Form submitted successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Form submission failed{Style.RESET_ALL}")
        print(f"  Reason: {result['message']}")
        if "incomplete_steps" in result:
            print(f"  Incomplete steps: {', '.join(result['incomplete_steps'])}")

    # Print form summary
    print_form_summary(form)


def demo_validation_errors() -> None:
    """Demonstrate validation errors in the form."""
    print_header("Multi-Step Form Validator - Validation Errors Demo")

    print("This demo shows how validation errors are handled in the form.")

    # Create the form
    form = create_registration_form()

    # Step 1: Personal Information with errors
    print_step_info(form)

    print_subheader("Submitting Invalid Personal Information")
    invalid_personal_data = {
        "first_name": "J",  # Too short
        "last_name": "",    # Missing
        "email": "not-an-email",  # Invalid email
        "age": 15  # Below minimum
    }

    print(f"Submitting data: {invalid_personal_data}")
    result = form.update_step_data("personal_info", invalid_personal_data)

    print(f"{Fore.RED}Expected validation errors:{Style.RESET_ALL}")
    for error in result["errors"]:
        print(f"  - {error.field}: {error.error}")

    # Try to complete step with errors
    print_subheader("Attempting to Complete Step with Errors")
    result = form.complete_step("personal_info")

    if not result["success"]:
        print(f"{Fore.GREEN}✓ System correctly prevented completion with errors{Style.RESET_ALL}")

    # Try to move to next step
    print_subheader("Attempting to Move to Next Step")
    result = form.next_step()

    if not result["success"]:
        print(f"{Fore.GREEN}✓ System correctly prevented navigation with errors{Style.RESET_ALL}")
        print(f"  Reason: {result['message']}")

    # Fix the errors
    print_subheader("Fixing the Errors")
    valid_personal_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30
    }

    print(f"Submitting corrected data: {valid_personal_data}")
    result = form.update_step_data("personal_info", valid_personal_data)

    if result["success"] and not result["errors"]:
        print(f"{Fore.GREEN}✓ Personal information accepted after correction{Style.RESET_ALL}")

    # Now complete step and move to next
    form.complete_step("personal_info")
    form.next_step()

    print_step_info(form)
    print(f"{Fore.GREEN}✓ Successfully moved to next step after fixing errors{Style.RESET_ALL}")


def demo_navigation() -> None:
    """Demonstrate navigation between form steps."""
    print_header("Multi-Step Form Validator - Navigation Demo")

    print("This demo shows how navigation works between form steps.")

    # Create the form
    form = create_registration_form()

    # Fill out first step
    form.update_step_data("personal_info", {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30
    })
    form.complete_step("personal_info")

    print_subheader("Initial State")
    print_step_info(form)

    # Move to next step
    print_subheader("Moving to Next Step")
    form.next_step()
    print_step_info(form)

    # Try to skip to review (should fail)
    print_subheader("Attempting to Skip to Review Step")
    result = form.go_to_step("review")

    if not result["success"]:
        print(f"{Fore.GREEN}✓ System correctly prevented skipping steps{Style.RESET_ALL}")
        print(f"  Reason: {result['message']}")

    # Go back to first step
    print_subheader("Going Back to Previous Step")
    form.previous_step()
    print_step_info(form)

    # Go forward again
    print_subheader("Going Forward Again")
    form.next_step()
    print_step_info(form)

    # Complete current step
    form.update_step_data("address", {
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip_code": "94105"
    })
    form.complete_step("address")

    # Move to next step
    print_subheader("Moving to Preferences Step")
    form.next_step()
    print_step_info(form)

    # Now we can go back to any previous step
    print_subheader("Going Back to Personal Info Step")
    form.go_to_step("personal_info")
    print_step_info(form)

    print(f"{Fore.GREEN}✓ Successfully demonstrated navigation between steps{Style.RESET_ALL}")


def demo_form_reset() -> None:
    """Demonstrate resetting the form."""
    print_header("Multi-Step Form Validator - Form Reset Demo")

    print("This demo shows how to reset a form to its initial state.")

    # Create the form
    form = create_registration_form()

    # Fill out and complete the form
    form.update_step_data("personal_info", {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30
    })
    form.complete_step("personal_info")
    form.next_step()

    form.update_step_data("address", {
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip_code": "94105"
    })
    form.complete_step("address")
    form.next_step()

    form.update_step_data("preferences", {
        "interests": ["tech", "travel"],
        "contact_method": "email",
        "comments": "Looking forward to using your service!"
    })
    form.complete_step("preferences")
    form.next_step()

    # Complete the review step
    form.complete_step("review")

    # Submit the form
    form.submit_form()

    print_subheader("Form Before Reset")
    print_form_summary(form)

    # Reset the form
    print_subheader("Resetting the Form")
    form.reset_form()

    print_subheader("Form After Reset")
    print_form_summary(form)
    print_step_info(form)

    print(f"{Fore.GREEN}✓ Successfully reset the form to its initial state{Style.RESET_ALL}")


def main() -> None:
    """Run the demo."""
    print_header("Multi-Step Form Validator Demo")

    print("""
This demo showcases a multi-step form validation system built with Pydantic.
The system allows defining form steps with specific fields and validation rules,
and ensures each step is valid before allowing progression to the next step.

Key features demonstrated:
1. Form step definition and validation
2. Field-level validation rules
3. Navigation between steps
4. Form submission and reset
5. Handling validation errors
    """)

    while True:
        print_subheader("Demo Options")
        print("1. Happy Path Demo (successful form completion)")
        print("2. Validation Errors Demo")
        print("3. Navigation Demo")
        print("4. Form Reset Demo")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            demo_happy_path()
        elif choice == "2":
            demo_validation_errors()
        elif choice == "3":
            demo_navigation()
        elif choice == "4":
            demo_form_reset()
        elif choice == "5":
            print("\nExiting demo. Goodbye!")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 5.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
