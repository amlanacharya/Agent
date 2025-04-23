"""
Tests for Advanced Validation Patterns
-----------------------------------
This module contains tests for the validation_patterns module.
"""

import unittest
from datetime import date, datetime, timedelta
from pydantic import ValidationError

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


class TestCrossFieldValidation(unittest.TestCase):
    """Test cases for cross-field validation."""

    def test_date_range_valid(self):
        """Test valid date range."""
        date_range = DateRange(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 15)
        )
        self.assertEqual(date_range.start_date, date(2023, 1, 1))
        self.assertEqual(date_range.end_date, date(2023, 1, 15))

    def test_date_range_invalid(self):
        """Test invalid date range."""
        with self.assertRaises(ValidationError) as context:
            DateRange(
                start_date=date(2023, 1, 15),
                end_date=date(2023, 1, 1)
            )
        self.assertIn("End date must be after start date", str(context.exception))

    def test_password_reset_valid(self):
        """Test valid password reset."""
        password_reset = PasswordReset(
            password="StrongP@ss123",
            password_confirm="StrongP@ss123"
        )
        self.assertEqual(password_reset.password, "StrongP@ss123")
        self.assertEqual(password_reset.password_confirm, "StrongP@ss123")

    def test_password_reset_mismatch(self):
        """Test password mismatch."""
        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="StrongP@ss123",
                password_confirm="DifferentP@ss123"
            )
        self.assertIn("Passwords do not match", str(context.exception))

    def test_password_reset_weak_password(self):
        """Test weak password."""
        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="weak",
                password_confirm="weak"
            )
        self.assertIn("Password must be at least 8 characters long", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="weakpassword",
                password_confirm="weakpassword"
            )
        self.assertIn("Password must contain at least one uppercase letter", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="WEAKPASSWORD",
                password_confirm="WEAKPASSWORD"
            )
        self.assertIn("Password must contain at least one lowercase letter", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="WeakPassword",
                password_confirm="WeakPassword"
            )
        self.assertIn("Password must contain at least one digit", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PasswordReset(
                password="WeakPassword123",
                password_confirm="WeakPassword123"
            )
        self.assertIn("Password must contain at least one special character", str(context.exception))


class TestConditionalValidation(unittest.TestCase):
    """Test cases for conditional validation."""

    def test_credit_card_info_valid(self):
        """Test valid credit card info."""
        credit_card_info = CreditCardInfo(
            card_number="4111 1111 1111 1111",
            expiration_date="12/25",
            cvv="123"
        )
        self.assertEqual(credit_card_info.card_number, "4111111111111111")
        self.assertEqual(credit_card_info.expiration_date, "12/25")
        self.assertEqual(credit_card_info.cvv, "123")

    def test_credit_card_info_invalid_number(self):
        """Test invalid credit card number."""
        with self.assertRaises(ValidationError) as context:
            CreditCardInfo(
                card_number="4111 1111 1111 1112",  # Invalid Luhn check
                expiration_date="12/25",
                cvv="123"
            )
        self.assertIn("Invalid card number", str(context.exception))

    def test_credit_card_info_invalid_expiration(self):
        """Test invalid expiration date."""
        # Create a date in the past
        past_year = datetime.now().year - 1
        past_month = datetime.now().month
        past_date = f"{past_month:02d}/{past_year % 100:02d}"

        with self.assertRaises(ValidationError) as context:
            CreditCardInfo(
                card_number="4111 1111 1111 1111",
                expiration_date=past_date,
                cvv="123"
            )
        self.assertIn("Card has expired", str(context.exception))

    def test_payment_valid(self):
        """Test valid payment."""
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
        self.assertEqual(payment.amount, 100.0)
        self.assertEqual(payment.currency, "USD")
        self.assertEqual(payment.payment_method, PaymentMethod.CREDIT_CARD)
        self.assertIsNotNone(payment.credit_card_info)

    def test_payment_missing_info(self):
        """Test payment with missing info."""
        with self.assertRaises(ValidationError) as context:
            Payment(
                amount=100.0,
                currency="USD",
                payment_method=PaymentMethod.CREDIT_CARD,
                # Missing credit_card_info
            )
        self.assertIn("Credit card information is required", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Payment(
                amount=100.0,
                currency="USD",
                payment_method=PaymentMethod.PAYPAL,
                # Missing paypal_info
            )
        self.assertIn("PayPal information is required", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Payment(
                amount=100.0,
                currency="USD",
                payment_method=PaymentMethod.BANK_TRANSFER,
                # Missing bank_transfer_info
            )
        self.assertIn("Bank transfer information is required", str(context.exception))


class TestContentBasedValidation(unittest.TestCase):
    """Test cases for content-based validation."""

    def test_content_valid(self):
        """Test valid content."""
        text_content = Content(
            content_type=ContentType.TEXT,
            title="My Text",
            text_content="This is some text content."
        )
        self.assertEqual(text_content.content_type, ContentType.TEXT)
        self.assertEqual(text_content.title, "My Text")
        self.assertEqual(text_content.text_content, "This is some text content.")

        image_content = Content(
            content_type=ContentType.IMAGE,
            title="My Image",
            file_path="image.jpg"
        )
        self.assertEqual(image_content.content_type, ContentType.IMAGE)
        self.assertEqual(image_content.title, "My Image")
        self.assertEqual(image_content.file_path, "image.jpg")

    def test_content_missing_required_fields(self):
        """Test content with missing required fields."""
        with self.assertRaises(ValidationError) as context:
            Content(
                content_type=ContentType.TEXT,
                title="My Text",
                # Missing text_content
            )
        self.assertIn("Text content is required", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Content(
                content_type=ContentType.IMAGE,
                title="My Image",
                # Missing file_path
            )
        self.assertIn("File path is required", str(context.exception))

    def test_content_invalid_file_extension(self):
        """Test content with invalid file extension."""
        with self.assertRaises(ValidationError) as context:
            Content(
                content_type=ContentType.IMAGE,
                title="My Image",
                file_path="image.txt"  # Invalid extension for image
            )
        self.assertIn("Invalid image file extension", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Content(
                content_type=ContentType.VIDEO,
                title="My Video",
                file_path="video.jpg"  # Invalid extension for video
            )
        self.assertIn("Invalid video file extension", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Content(
                content_type=ContentType.DOCUMENT,
                title="My Document",
                file_path="document.mp4"  # Invalid extension for document
            )
        self.assertIn("Invalid document file extension", str(context.exception))


class TestContextDependentValidation(unittest.TestCase):
    """Test cases for context-dependent validation."""

    def test_booking_request_valid(self):
        """Test valid booking request."""
        booking = BookingRequest(
            room_id=101,
            check_in_date=date(2023, 6, 20),
            check_out_date=date(2023, 6, 25),
            guest_count=2
        )
        self.assertEqual(booking.room_id, 101)
        self.assertEqual(booking.check_in_date, date(2023, 6, 20))
        self.assertEqual(booking.check_out_date, date(2023, 6, 25))
        self.assertEqual(booking.guest_count, 2)

    def test_booking_request_invalid_dates(self):
        """Test booking request with invalid dates."""
        with self.assertRaises(ValidationError) as context:
            BookingRequest(
                room_id=101,
                check_in_date=date(2023, 6, 25),
                check_out_date=date(2023, 6, 20),
                guest_count=2
            )
        self.assertIn("Check-out must be after check-in", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            BookingRequest(
                room_id=101,
                check_in_date=date(2023, 6, 1),
                check_out_date=date(2023, 6, 20),  # 19 days, exceeds maximum
                guest_count=2
            )
        self.assertIn("Maximum stay is 14 days", str(context.exception))

    def test_booking_system_validation(self):
        """Test booking system validation."""
        booking_system = BookingSystem()

        # Test non-existent room
        booking = BookingRequest(
            room_id=999,  # Non-existent room
            check_in_date=date(2023, 6, 20),
            check_out_date=date(2023, 6, 25),
            guest_count=2
        )
        errors = booking_system.validate_booking(booking)
        self.assertIn("Room 999 does not exist", errors)

        # Test exceeding room capacity
        booking = BookingRequest(
            room_id=101,
            check_in_date=date(2023, 6, 20),
            check_out_date=date(2023, 6, 25),
            guest_count=3  # Room 101 capacity is 2
        )
        errors = booking_system.validate_booking(booking)
        self.assertIn("Room 101 can only accommodate 2 guests", errors)

        # Test room availability
        booking = BookingRequest(
            room_id=101,
            check_in_date=date(2023, 6, 3),  # Conflicts with existing booking
            check_out_date=date(2023, 6, 7),
            guest_count=2
        )
        errors = booking_system.validate_booking(booking)
        self.assertIn("Room 101 is not available for the selected dates", errors)

        # Test valid booking
        booking = BookingRequest(
            room_id=101,
            check_in_date=date(2023, 6, 6),
            check_out_date=date(2023, 6, 9),
            guest_count=2
        )
        errors = booking_system.validate_booking(booking)
        self.assertEqual(errors, [])


class TestDynamicValidation(unittest.TestCase):
    """Test cases for dynamic validation."""

    def test_dynamic_validation(self):
        """Test dynamic validation."""
        # Add validators dynamically
        setattr(DynamicModel, 'validate_value', field_validator('value')(create_range_validator(0, 100)))
        setattr(DynamicModel, 'validate_items', field_validator('items')(create_list_length_validator(1, 5)))

        # Test valid model
        model = DynamicModel(value=50, items=["item1", "item2"])
        self.assertEqual(model.value, 50)
        self.assertEqual(model.items, ["item1", "item2"])

        # Test invalid value
        with self.assertRaises(ValidationError) as context:
            DynamicModel(value=150, items=["item1"])
        self.assertIn("Value must be between 0 and 100", str(context.exception))

        # Test invalid items length
        with self.assertRaises(ValidationError) as context:
            DynamicModel(value=50, items=[])
        self.assertIn("List must have at least 1 items", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            DynamicModel(value=50, items=["item1", "item2", "item3", "item4", "item5", "item6"])
        self.assertIn("List must have at most 5 items", str(context.exception))


class TestAdvancedFormValidation(unittest.TestCase):
    """Test cases for advanced form validation."""

    def test_employment_info_valid(self):
        """Test valid employment info."""
        employed_info = EmploymentInfo(
            status=EmploymentStatus.EMPLOYED,
            employer="Acme Inc",
            position="Software Engineer",
            years_at_job=3,
            annual_income=80000.0
        )
        self.assertEqual(employed_info.status, EmploymentStatus.EMPLOYED)
        self.assertEqual(employed_info.employer, "Acme Inc")
        self.assertEqual(employed_info.position, "Software Engineer")
        self.assertEqual(employed_info.years_at_job, 3)
        self.assertEqual(employed_info.annual_income, 80000.0)

        unemployed_info = EmploymentInfo(
            status=EmploymentStatus.UNEMPLOYED
        )
        self.assertEqual(unemployed_info.status, EmploymentStatus.UNEMPLOYED)
        self.assertIsNone(unemployed_info.employer)
        self.assertIsNone(unemployed_info.position)
        self.assertIsNone(unemployed_info.years_at_job)
        self.assertIsNone(unemployed_info.annual_income)

    def test_employment_info_missing_required_fields(self):
        """Test employment info with missing required fields."""
        with self.assertRaises(ValidationError) as context:
            EmploymentInfo(
                status=EmploymentStatus.EMPLOYED,
                # Missing required fields for employed status
            )
        self.assertIn("Employer is required for employed status", str(context.exception))

    def test_address_valid(self):
        """Test valid address."""
        us_address = Address(
            street="123 Main St",
            city="Anytown",
            state="CA",
            zip_code="12345",
            country="USA"
        )
        self.assertEqual(us_address.street, "123 Main St")
        self.assertEqual(us_address.city, "Anytown")
        self.assertEqual(us_address.state, "CA")
        self.assertEqual(us_address.zip_code, "12345")
        self.assertEqual(us_address.country, "USA")

        canadian_address = Address(
            street="123 Maple St",
            city="Toronto",
            state="ON",
            zip_code="M5V 2N4",
            country="Canada"
        )
        self.assertEqual(canadian_address.zip_code, "M5V 2N4")

    def test_address_invalid_zip_code(self):
        """Test address with invalid zip code."""
        with self.assertRaises(ValidationError) as context:
            Address(
                street="123 Main St",
                city="Anytown",
                state="CA",
                zip_code="invalid",
                country="USA"
            )
        self.assertIn("Invalid US ZIP code format", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Address(
                street="123 Maple St",
                city="Toronto",
                state="ON",
                zip_code="invalid",
                country="Canada"
            )
        self.assertIn("Invalid Canadian postal code format", str(context.exception))

    def test_loan_application_valid(self):
        """Test valid loan application."""
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
        self.assertEqual(loan_application.applicant_name, "John Doe")
        self.assertEqual(loan_application.email, "john@example.com")
        self.assertEqual(loan_application.phone, "(123) 456-7890")  # Formatted
        self.assertEqual(loan_application.date_of_birth, date(1990, 1, 1))
        self.assertEqual(loan_application.loan_amount, 30000.0)
        self.assertEqual(loan_application.has_existing_loans, True)
        self.assertEqual(loan_application.existing_loan_amount, 10000.0)

    def test_loan_application_invalid_age(self):
        """Test loan application with invalid age."""
        with self.assertRaises(ValidationError) as context:
            LoanApplication(
                applicant_name="John Doe",
                email="john@example.com",
                phone="1234567890",
                date_of_birth=date.today() - timedelta(days=365 * 17),  # 17 years old
                address=Address(
                    street="123 Main St",
                    city="Anytown",
                    state="CA",
                    zip_code="12345",
                    country="USA"
                ),
                education_level=EducationLevel.HIGH_SCHOOL,
                employment_info=EmploymentInfo(
                    status=EmploymentStatus.STUDENT
                ),
                loan_amount=5000.0,
                loan_purpose="Education"
            )
        self.assertIn("Applicant must be at least 18 years old", str(context.exception))

    def test_loan_application_existing_loans(self):
        """Test loan application with existing loans validation."""
        with self.assertRaises(ValidationError) as context:
            LoanApplication(
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
                has_existing_loans=True,
                # Missing existing_loan_amount
            )
        self.assertIn("Existing loan amount is required", str(context.exception))

    def test_loan_application_income_based_validation(self):
        """Test loan application with income-based validation."""
        with self.assertRaises(ValidationError) as context:
            LoanApplication(
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
                credit_score=700
            )
        self.assertIn("Loan amount cannot exceed 50% of annual income", str(context.exception))

    def test_loan_application_credit_score_based_validation(self):
        """Test loan application with credit score-based validation."""
        with self.assertRaises(ValidationError) as context:
            LoanApplication(
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
                loan_amount=10000.0,
                loan_purpose="Home renovation",
                credit_score=550  # Below 600
            )
        self.assertIn("Maximum loan amount for credit score below 600 is $5,000", str(context.exception))


if __name__ == "__main__":
    unittest.main()
