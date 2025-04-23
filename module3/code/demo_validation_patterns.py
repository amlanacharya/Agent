"""
Demo for Advanced Validation Patterns
----------------------------------
This script demonstrates the advanced validation patterns from the validation_patterns module.
"""

from validation_patterns import (
    DateRange,
    PasswordReset,
    PaymentMethod,
    CreditCardInfo,
    PayPalInfo,
    BankTransferInfo,
    Payment,
    ContentType,
    Content,
    BookingRequest,
    BookingSystem,
    DynamicModel,
    create_range_validator,
    create_list_length_validator,
    EducationLevel,
    EmploymentStatus,
    EmploymentInfo,
    Address,
    LoanApplication
)
from datetime import datetime, date, timedelta
from pydantic import field_validator


def demo_cross_field_validation():
    """Demonstrate cross-field validation."""
    print("\n=== Cross-Field Validation ===\n")

    # Valid date range
    try:
        date_range = DateRange(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 15)
        )
        print(f"Valid date range: {date_range}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid date range
    try:
        date_range = DateRange(
            start_date=date(2023, 1, 15),
            end_date=date(2023, 1, 1)
        )
        print(f"Valid date range: {date_range}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Valid password reset
    try:
        password_reset = PasswordReset(
            password="StrongP@ss123",
            password_confirm="StrongP@ss123"
        )
        print(f"Valid password reset: {password_reset}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid password reset (mismatch)
    try:
        password_reset = PasswordReset(
            password="StrongP@ss123",
            password_confirm="DifferentP@ss123"
        )
        print(f"Valid password reset: {password_reset}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid password reset (weak password)
    try:
        password_reset = PasswordReset(
            password="weak",
            password_confirm="weak"
        )
        print(f"Valid password reset: {password_reset}")
    except ValueError as e:
        print(f"Validation error: {e}")


def demo_conditional_validation():
    """Demonstrate conditional validation."""
    print("\n=== Conditional Validation ===\n")

    # Valid credit card payment
    try:
        payment = Payment(
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethod.CREDIT_CARD,
            credit_card_info=CreditCardInfo(
                card_number="4111 1111 1111 1111",
                expiration_date="12/25",
                cvv="123"
            )
        )
        print(f"Valid credit card payment: {payment}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Valid PayPal payment
    try:
        payment = Payment(
            amount=50.0,
            currency="EUR",
            payment_method=PaymentMethod.PAYPAL,
            paypal_info=PayPalInfo(
                email="john@example.com"
            )
        )
        print(f"Valid PayPal payment: {payment}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Valid bank transfer payment
    try:
        payment = Payment(
            amount=200.0,
            currency="GBP",
            payment_method=PaymentMethod.BANK_TRANSFER,
            bank_transfer_info=BankTransferInfo(
                account_number="12345678",
                routing_number="123456789",
                account_name="John Doe"
            )
        )
        print(f"Valid bank transfer payment: {payment}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid payment (missing required info)
    try:
        payment = Payment(
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethod.CREDIT_CARD,
            # Missing credit_card_info
        )
        print(f"Valid payment: {payment}")
    except ValueError as e:
        print(f"Validation error: {e}")


def demo_content_based_validation():
    """Demonstrate content-based validation."""
    print("\n=== Content-Based Validation ===\n")

    # Valid text content
    try:
        content = Content(
            content_type=ContentType.TEXT,
            title="My Text",
            text_content="This is some text content."
        )
        print(f"Valid text content: {content}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Valid image content
    try:
        content = Content(
            content_type=ContentType.IMAGE,
            title="My Image",
            file_path="image.jpg"
        )
        print(f"Valid image content: {content}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid content (missing required field)
    try:
        content = Content(
            content_type=ContentType.TEXT,
            title="My Text",
            # Missing text_content
        )
        print(f"Valid content: {content}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid content (invalid file extension)
    try:
        content = Content(
            content_type=ContentType.IMAGE,
            title="My Image",
            file_path="image.txt"  # Invalid extension for image
        )
        print(f"Valid content: {content}")
    except ValueError as e:
        print(f"Validation error: {e}")


def demo_context_dependent_validation():
    """Demonstrate context-dependent validation."""
    print("\n=== Context-Dependent Validation ===\n")

    # Create booking system
    booking_system = BookingSystem()

    # Valid booking
    booking = BookingRequest(
        room_id=101,
        check_in_date=date(2023, 6, 6),
        check_out_date=date(2023, 6, 9),
        guest_count=2
    )
    errors = booking_system.validate_booking(booking)
    if errors:
        print(f"Booking validation errors: {errors}")
    else:
        print(f"Valid booking: {booking}")

    # Invalid booking (exceeds capacity)
    booking = BookingRequest(
        room_id=101,
        check_in_date=date(2023, 6, 6),
        check_out_date=date(2023, 6, 9),
        guest_count=3  # Room 101 capacity is 2
    )
    errors = booking_system.validate_booking(booking)
    if errors:
        print(f"Booking validation errors: {errors}")
    else:
        print(f"Valid booking: {booking}")

    # Invalid booking (unavailable dates)
    booking = BookingRequest(
        room_id=101,
        check_in_date=date(2023, 6, 3),  # Conflicts with existing booking
        check_out_date=date(2023, 6, 7),
        guest_count=2
    )
    errors = booking_system.validate_booking(booking)
    if errors:
        print(f"Booking validation errors: {errors}")
    else:
        print(f"Valid booking: {booking}")


def demo_dynamic_validation():
    """Demonstrate dynamic validation."""
    print("\n=== Dynamic Validation ===\n")

    # Add validators dynamically
    setattr(DynamicModel, 'validate_value', field_validator('value')(create_range_validator(0, 100)))
    setattr(DynamicModel, 'validate_items', field_validator('items')(create_list_length_validator(1, 5)))

    # Valid model
    try:
        model = DynamicModel(value=50, items=["item1", "item2"])
        print(f"Valid dynamic model: {model}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid model (value out of range)
    try:
        model = DynamicModel(value=150, items=["item1"])
        print(f"Valid dynamic model: {model}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid model (empty items list)
    try:
        model = DynamicModel(value=50, items=[])
        print(f"Valid dynamic model: {model}")
    except ValueError as e:
        print(f"Validation error: {e}")


def demo_advanced_form_validation():
    """Demonstrate advanced form validation."""
    print("\n=== Advanced Form Validation ===\n")

    # Valid loan application
    try:
        loan_application = LoanApplication(
            applicant_name="John Doe",
            email="john@example.com",
            phone="1234567890",
            date_of_birth=date(1990, 1, 1),
            address=Address(
                street="123 Main St",
                city="Anytown",
                state="CA",
                zip_code="12345",
                country="USA"
            ),
            education_level=EducationLevel.BACHELOR,
            employment_info=EmploymentInfo(
                status=EmploymentStatus.EMPLOYED,
                employer="Acme Inc",
                position="Software Engineer",
                years_at_job=3,
                annual_income=80000.0
            ),
            loan_amount=30000.0,  # Less than 50% of income
            loan_purpose="Home renovation",
            credit_score=700,
            has_existing_loans=True,
            existing_loan_amount=10000.0
        )
        print(f"Valid loan application: {loan_application}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid loan application (loan amount exceeds 50% of income)
    try:
        loan_application = LoanApplication(
            applicant_name="John Doe",
            email="john@example.com",
            phone="1234567890",
            date_of_birth=date(1990, 1, 1),
            address=Address(
                street="123 Main St",
                city="Anytown",
                state="CA",
                zip_code="12345",
                country="USA"
            ),
            education_level=EducationLevel.BACHELOR,
            employment_info=EmploymentInfo(
                status=EmploymentStatus.EMPLOYED,
                employer="Acme Inc",
                position="Software Engineer",
                years_at_job=3,
                annual_income=80000.0
            ),
            loan_amount=50000.0,  # Exceeds 50% of income
            loan_purpose="Home renovation",
            credit_score=700,
            has_existing_loans=True,
            existing_loan_amount=10000.0
        )
        print(f"Valid loan application: {loan_application}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Invalid loan application (missing existing loan amount)
    try:
        loan_application = LoanApplication(
            applicant_name="John Doe",
            email="john@example.com",
            phone="1234567890",
            date_of_birth=date(1990, 1, 1),
            address=Address(
                street="123 Main St",
                city="Anytown",
                state="CA",
                zip_code="12345",
                country="USA"
            ),
            education_level=EducationLevel.BACHELOR,
            employment_info=EmploymentInfo(
                status=EmploymentStatus.EMPLOYED,
                employer="Acme Inc",
                position="Software Engineer",
                years_at_job=3,
                annual_income=80000.0
            ),
            loan_amount=30000.0,
            loan_purpose="Home renovation",
            credit_score=700,
            has_existing_loans=True,
            # Missing existing_loan_amount
        )
        print(f"Valid loan application: {loan_application}")
    except ValueError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    print("=== Advanced Validation Patterns Demo ===")

    demo_cross_field_validation()
    demo_conditional_validation()
    demo_content_based_validation()
    demo_context_dependent_validation()
    demo_dynamic_validation()
    demo_advanced_form_validation()

    print("\n=== Demo Complete ===")
