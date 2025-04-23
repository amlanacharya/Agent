"""
Tests for Lesson 4.1 Exercise Solutions
-----------------------------------
This module contains tests for the lesson4.1_exercises module.
"""

import unittest
from datetime import date, datetime, timedelta
from pydantic import ValidationError

from lesson4_1_exercises import (
    # Exercise 1
    PaymentMethodType,
    CreditCardDetails,
    PayPalDetails,
    BankTransferDetails,
    PaymentSystem,

    # Exercise 2
    TravelType,
    PassengerType,
    PassengerInfo,
    TravelBooking,

    # Exercise 3
    ProductCategory,
    ProductInventory,

    # Exercise 4
    QuestionType,
    Question,
    Answer,
    SurveyForm,
    SurveyResponse
)


class TestPaymentSystem(unittest.TestCase):
    """Test cases for PaymentSystem exercise."""

    def test_credit_card_payment_valid(self):
        """Test valid credit card payment."""
        payment = PaymentSystem(
            transaction_id="TX123456",
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethodType.CREDIT_CARD,
            credit_card_details=CreditCardDetails(
                card_number="4111 1111 1111 1111",
                cardholder_name="John Doe",
                expiration_month=12,
                expiration_year=datetime.now().year + 2,
                cvv="123"
            )
        )
        self.assertEqual(payment.transaction_id, "TX123456")
        self.assertEqual(payment.amount, 100.0)
        self.assertEqual(payment.currency, "USD")
        self.assertEqual(payment.payment_method, PaymentMethodType.CREDIT_CARD)
        self.assertIsNotNone(payment.credit_card_details)

    def test_paypal_payment_valid(self):
        """Test valid PayPal payment."""
        payment = PaymentSystem(
            transaction_id="TX123457",
            amount=50.0,
            currency="EUR",
            payment_method=PaymentMethodType.PAYPAL,
            paypal_details=PayPalDetails(
                email="john@example.com"
            )
        )
        self.assertEqual(payment.transaction_id, "TX123457")
        self.assertEqual(payment.amount, 50.0)
        self.assertEqual(payment.currency, "EUR")
        self.assertEqual(payment.payment_method, PaymentMethodType.PAYPAL)
        self.assertIsNotNone(payment.paypal_details)

    def test_bank_transfer_payment_valid(self):
        """Test valid bank transfer payment."""
        payment = PaymentSystem(
            transaction_id="TX123458",
            amount=200.0,
            currency="GBP",
            payment_method=PaymentMethodType.BANK_TRANSFER,
            bank_transfer_details=BankTransferDetails(
                account_holder="John Doe",
                account_number="12345678",
                routing_number="111000025",  # Valid ABA routing number
                bank_name="Example Bank"
            )
        )
        self.assertEqual(payment.transaction_id, "TX123458")
        self.assertEqual(payment.amount, 200.0)
        self.assertEqual(payment.currency, "GBP")
        self.assertEqual(payment.payment_method, PaymentMethodType.BANK_TRANSFER)
        self.assertIsNotNone(payment.bank_transfer_details)

    def test_payment_missing_details(self):
        """Test payment with missing details."""
        with self.assertRaises(ValidationError) as context:
            PaymentSystem(
                transaction_id="TX123456",
                amount=100.0,
                currency="USD",
                payment_method=PaymentMethodType.CREDIT_CARD,
                # Missing credit_card_details
            )
        self.assertIn("Credit card details are required", str(context.exception))

    def test_payment_invalid_amount(self):
        """Test payment with invalid amount."""
        with self.assertRaises(ValidationError) as context:
            PaymentSystem(
                transaction_id="TX123456",
                amount=0.0,  # Invalid amount
                currency="USD",
                payment_method=PaymentMethodType.CREDIT_CARD,
                credit_card_details=CreditCardDetails(
                    card_number="4111 1111 1111 1111",
                    cardholder_name="John Doe",
                    expiration_month=12,
                    expiration_year=datetime.now().year + 2,
                    cvv="123"
                )
            )
        self.assertIn("Payment amount must be greater than zero", str(context.exception))

    def test_credit_card_details_invalid(self):
        """Test invalid credit card details."""
        with self.assertRaises(ValidationError) as context:
            CreditCardDetails(
                card_number="4111 1111 1111 1112",  # Invalid Luhn check
                cardholder_name="John Doe",
                expiration_month=12,
                expiration_year=datetime.now().year + 2,
                cvv="123"
            )
        self.assertIn("Invalid card number", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            CreditCardDetails(
                card_number="4111 1111 1111 1111",
                cardholder_name="John Doe",
                expiration_month=13,  # Invalid month
                expiration_year=datetime.now().year + 2,
                cvv="123"
            )
        self.assertIn("Expiration month must be between 1 and 12", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            CreditCardDetails(
                card_number="4111 1111 1111 1111",
                cardholder_name="John Doe",
                expiration_month=12,
                expiration_year=datetime.now().year - 1,  # Expired
                cvv="123"
            )
        self.assertIn("Card has expired", str(context.exception))


class TestTravelBooking(unittest.TestCase):
    """Test cases for TravelBooking exercise."""

    def test_domestic_booking_valid(self):
        """Test valid domestic booking."""
        booking = TravelBooking(
            booking_reference="ABC123",
            travel_type=TravelType.DOMESTIC,
            origin="JFK",
            destination="LAX",
            departure_date=date.today() + timedelta(days=30),
            return_date=date.today() + timedelta(days=37),
            passengers=[
                PassengerInfo(
                    first_name="John",
                    last_name="Doe",
                    date_of_birth=date(1980, 1, 1),
                    passenger_type=PassengerType.ADULT
                )
            ],
            contact_email="john@example.com",
            contact_phone="+1234567890"
        )
        self.assertEqual(booking.booking_reference, "ABC123")
        self.assertEqual(booking.travel_type, TravelType.DOMESTIC)
        self.assertEqual(booking.origin, "JFK")
        self.assertEqual(booking.destination, "LAX")
        self.assertEqual(len(booking.passengers), 1)

    def test_international_booking_valid(self):
        """Test valid international booking."""
        booking = TravelBooking(
            booking_reference="ABC123",
            travel_type=TravelType.INTERNATIONAL,
            origin="JFK",
            destination="LHR",
            departure_date=date.today() + timedelta(days=30),
            return_date=date.today() + timedelta(days=37),
            passengers=[
                PassengerInfo(
                    first_name="John",
                    last_name="Doe",
                    date_of_birth=date(1980, 1, 1),
                    passenger_type=PassengerType.ADULT,
                    passport_number="AB1234567",
                    passport_expiry=date.today() + timedelta(days=365 * 5),
                    nationality="USA"
                )
            ],
            contact_email="john@example.com",
            contact_phone="+1234567890"
        )
        self.assertEqual(booking.booking_reference, "ABC123")
        self.assertEqual(booking.travel_type, TravelType.INTERNATIONAL)
        self.assertEqual(booking.origin, "JFK")
        self.assertEqual(booking.destination, "LHR")
        self.assertEqual(len(booking.passengers), 1)

    def test_international_booking_missing_passport(self):
        """Test international booking with missing passport."""
        with self.assertRaises(ValidationError) as context:
            TravelBooking(
                booking_reference="ABC123",
                travel_type=TravelType.INTERNATIONAL,
                origin="JFK",
                destination="LHR",
                departure_date=date.today() + timedelta(days=30),
                return_date=date.today() + timedelta(days=37),
                passengers=[
                    PassengerInfo(
                        first_name="John",
                        last_name="Doe",
                        date_of_birth=date(1980, 1, 1),
                        passenger_type=PassengerType.ADULT,
                        # Missing passport information
                    )
                ],
                contact_email="john@example.com",
                contact_phone="+1234567890"
            )
        self.assertIn("Passport number is required", str(context.exception))

    def test_passenger_type_validation(self):
        """Test passenger type validation based on age."""
        with self.assertRaises(ValidationError) as context:
            PassengerInfo(
                first_name="John",
                last_name="Doe",
                date_of_birth=date(1980, 1, 1),
                passenger_type=PassengerType.INFANT  # Invalid for age
            )
        self.assertIn("Infant passengers must be under 2 years old", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PassengerInfo(
                first_name="John",
                last_name="Doe",
                date_of_birth=date(1980, 1, 1),
                passenger_type=PassengerType.CHILD  # Invalid for age
            )
        self.assertIn("Child passengers must be between 2 and 11 years old", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PassengerInfo(
                first_name="John",
                last_name="Doe",
                date_of_birth=date(1950, 1, 1),
                passenger_type=PassengerType.ADULT  # Invalid for age
            )
        self.assertIn("Adult passengers must be between 12 and 64 years old", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            PassengerInfo(
                first_name="John",
                last_name="Doe",
                date_of_birth=date(1980, 1, 1),
                passenger_type=PassengerType.SENIOR  # Invalid for age
            )
        self.assertIn("Senior passengers must be 65 years or older", str(context.exception))


class TestProductInventory(unittest.TestCase):
    """Test cases for ProductInventory exercise."""

    def test_electronics_product_valid(self):
        """Test valid electronics product."""
        product = ProductInventory(
            product_id="EL123456",
            name="Smartphone",
            description="Latest smartphone model",
            category=ProductCategory.ELECTRONICS,
            price=999.99,
            quantity=10,
            attributes={
                "brand": "TechCo",
                "model": "X1",
                "warranty_months": 24,
                "voltage": 5
            }
        )
        self.assertEqual(product.product_id, "EL123456")
        self.assertEqual(product.name, "Smartphone")
        self.assertEqual(product.category, ProductCategory.ELECTRONICS)
        self.assertEqual(product.price, 999.99)
        self.assertEqual(product.quantity, 10)
        self.assertEqual(product.attributes["brand"], "TechCo")

    def test_clothing_product_valid(self):
        """Test valid clothing product."""
        product = ProductInventory(
            product_id="CL123456",
            name="T-Shirt",
            description="Cotton T-Shirt",
            category=ProductCategory.CLOTHING,
            price=19.99,
            quantity=100,
            attributes={
                "size": "M",
                "color": "Blue",
                "material": "Cotton"
            }
        )
        self.assertEqual(product.product_id, "CL123456")
        self.assertEqual(product.name, "T-Shirt")
        self.assertEqual(product.category, ProductCategory.CLOTHING)
        self.assertEqual(product.price, 19.99)
        self.assertEqual(product.quantity, 100)
        self.assertEqual(product.attributes["size"], "M")

    def test_product_missing_attributes(self):
        """Test product with missing attributes."""
        with self.assertRaises(ValidationError) as context:
            ProductInventory(
                product_id="EL123456",
                name="Smartphone",
                description="Latest smartphone model",
                category=ProductCategory.ELECTRONICS,
                price=999.99,
                quantity=10,
                attributes={
                    # Missing required attributes
                }
            )
        self.assertIn("'brand' is required for electronics products", str(context.exception))

    def test_product_invalid_attributes(self):
        """Test product with invalid attributes."""
        with self.assertRaises(ValidationError) as context:
            ProductInventory(
                product_id="EL123456",
                name="Smartphone",
                description="Latest smartphone model",
                category=ProductCategory.ELECTRONICS,
                price=999.99,
                quantity=10,
                attributes={
                    "brand": "TechCo",
                    "model": "X1",
                    "warranty_months": -1,  # Invalid warranty
                    "voltage": 5
                }
            )
        self.assertIn("warranty_months must be a non-negative integer", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            ProductInventory(
                product_id="CL123456",
                name="T-Shirt",
                description="Cotton T-Shirt",
                category=ProductCategory.CLOTHING,
                price=19.99,
                quantity=100,
                attributes={
                    "size": "XXS",  # Invalid size
                    "color": "Blue",
                    "material": "Cotton"
                }
            )
        self.assertIn("size must be one of", str(context.exception))


class TestSurveyForm(unittest.TestCase):
    """Test cases for SurveyForm exercise."""

    def test_survey_form_valid(self):
        """Test valid survey form."""
        survey = SurveyForm(
            id="SV001",
            title="Customer Satisfaction Survey",
            description="Please provide your feedback",
            questions=[
                Question(
                    id="Q1",
                    text="How satisfied are you with our service?",
                    type=QuestionType.RATING,
                    min_rating=1,
                    max_rating=5
                ),
                Question(
                    id="Q2",
                    text="Would you recommend our service to others?",
                    type=QuestionType.SINGLE_CHOICE,
                    options=["Yes", "No", "Maybe"]
                ),
                Question(
                    id="Q3",
                    text="Why not?",
                    type=QuestionType.TEXT,
                    depends_on="Q2",
                    depends_on_value="No"
                )
            ]
        )
        self.assertEqual(survey.id, "SV001")
        self.assertEqual(survey.title, "Customer Satisfaction Survey")
        self.assertEqual(len(survey.questions), 3)

    def test_question_validation(self):
        """Test question validation."""
        with self.assertRaises(ValidationError) as context:
            Question(
                id="Q1",
                text="Select your favorite colors",
                type=QuestionType.MULTIPLE_CHOICE,
                # Missing options
            )
        self.assertIn("Options are required for multiple_choice questions", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            Question(
                id="Q1",
                text="Rate our service",
                type=QuestionType.RATING,
                # Missing min_rating and max_rating
            )
        self.assertIn("min_rating and max_rating are required for rating questions", str(context.exception))

    def test_survey_response_validation(self):
        """Test survey response validation."""
        survey = SurveyForm(
            id="SV001",
            title="Customer Satisfaction Survey",
            description="Please provide your feedback",
            questions=[
                Question(
                    id="Q1",
                    text="How satisfied are you with our service?",
                    type=QuestionType.RATING,
                    min_rating=1,
                    max_rating=5
                ),
                Question(
                    id="Q2",
                    text="Would you recommend our service to others?",
                    type=QuestionType.SINGLE_CHOICE,
                    options=["Yes", "No", "Maybe"]
                ),
                Question(
                    id="Q3",
                    text="Why not?",
                    type=QuestionType.TEXT,
                    depends_on="Q2",
                    depends_on_value="No"
                )
            ]
        )

        # Valid response with dependency met
        response1 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R001",
            answers=[
                Answer(question_id="Q1", value=4),
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive")
            ]
        )
        errors1 = response1.validate_with_survey(survey)
        self.assertEqual(errors1, [])

        # Valid response with dependency not met (Q3 not required)
        response2 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R002",
            answers=[
                Answer(question_id="Q1", value=5),
                Answer(question_id="Q2", value="Yes")
                # Q3 not answered because Q2 is not "No"
            ]
        )
        errors2 = response2.validate_with_survey(survey)
        self.assertEqual(errors2, [])

        # Invalid response (missing required answer)
        response3 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R003",
            answers=[
                # Missing Q1
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive")
            ]
        )
        errors3 = response3.validate_with_survey(survey)
        self.assertIn("Required question not answered: Q1", errors3)

        # Invalid response (invalid rating)
        response4 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R004",
            answers=[
                Answer(question_id="Q1", value=6),  # Rating out of range
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive")
            ]
        )
        errors4 = response4.validate_with_survey(survey)
        self.assertIn("Rating for question Q1 must be between 1 and 5", errors4)


if __name__ == "__main__":
    unittest.main()
