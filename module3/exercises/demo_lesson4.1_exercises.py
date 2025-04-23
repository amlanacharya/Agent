"""
Demo for Lesson 4.1 Exercise Solutions
-----------------------------------
This script demonstrates the solutions for the exercises in Lesson 4.1.
"""

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
from datetime import datetime, date, timedelta
from pydantic import ValidationError


def demo_payment_system():
    """Demonstrate the PaymentSystem model."""
    print("\n=== Exercise 1: PaymentSystem ===\n")

    # Valid credit card payment
    try:
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
        print(f"Valid credit card payment: {payment}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Valid PayPal payment
    try:
        payment = PaymentSystem(
            transaction_id="TX123457",
            amount=50.0,
            currency="EUR",
            payment_method=PaymentMethodType.PAYPAL,
            paypal_details=PayPalDetails(
                email="john@example.com"
            )
        )
        print(f"Valid PayPal payment: {payment}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Valid bank transfer payment
    try:
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
        print(f"Valid bank transfer payment: {payment}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid payment (missing required details)
    try:
        payment = PaymentSystem(
            transaction_id="TX123456",
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethodType.CREDIT_CARD,
            # Missing credit_card_details
        )
        print(f"Valid payment: {payment}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid payment (providing wrong details)
    try:
        payment = PaymentSystem(
            transaction_id="TX123456",
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethodType.CREDIT_CARD,
            paypal_details=PayPalDetails(  # Wrong details for credit card payment
                email="john@example.com"
            )
        )
        print(f"Valid payment: {payment}")
    except ValidationError as e:
        print(f"Validation error: {e}")


def demo_travel_booking():
    """Demonstrate the TravelBooking model."""
    print("\n=== Exercise 2: TravelBooking ===\n")

    # Valid domestic booking
    try:
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
        print(f"Valid domestic booking: {booking}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Valid international booking
    try:
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
        print(f"Valid international booking: {booking}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid international booking (missing passport information)
    try:
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
                    # Missing passport information
                )
            ],
            contact_email="john@example.com",
            contact_phone="+1234567890"
        )
        print(f"Valid booking: {booking}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid booking (invalid passenger type for age)
    try:
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
                    passenger_type=PassengerType.CHILD  # Invalid for age
                )
            ],
            contact_email="john@example.com",
            contact_phone="+1234567890"
        )
        print(f"Valid booking: {booking}")
    except ValidationError as e:
        print(f"Validation error: {e}")


def demo_product_inventory():
    """Demonstrate the ProductInventory model."""
    print("\n=== Exercise 3: ProductInventory ===\n")

    # Valid electronics product
    try:
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
        print(f"Valid electronics product: {product}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Valid clothing product
    try:
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
        print(f"Valid clothing product: {product}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Valid food product
    try:
        product = ProductInventory(
            product_id="FO123456",
            name="Chocolate Bar",
            description="Dark chocolate",
            category=ProductCategory.FOOD,
            price=2.99,
            quantity=500,
            attributes={
                "expiration_date": (date.today() + timedelta(days=365)).strftime("%Y-%m-%d"),
                "allergens": ["milk", "soy"]
            }
        )
        print(f"Valid food product: {product}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid product (missing required attributes)
    try:
        product = ProductInventory(
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
        print(f"Valid product: {product}")
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid product (invalid attribute value)
    try:
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
                "warranty_months": -1,  # Invalid warranty
                "voltage": 5
            }
        )
        print(f"Valid product: {product}")
    except ValidationError as e:
        print(f"Validation error: {e}")


def demo_survey_form():
    """Demonstrate the SurveyForm and SurveyResponse models."""
    print("\n=== Exercise 4: SurveyForm ===\n")

    # Create a survey form
    try:
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
                ),
                Question(
                    id="Q4",
                    text="What features would you like to see improved?",
                    type=QuestionType.MULTIPLE_CHOICE,
                    options=["User Interface", "Performance", "Reliability", "Features", "Support"]
                )
            ]
        )
        print(f"Valid survey form: {survey}")

        # Valid response with dependency met
        response1 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R001",
            answers=[
                Answer(question_id="Q1", value=4),
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive"),
                Answer(question_id="Q4", value=["User Interface", "Performance"])
            ]
        )

        errors1 = response1.validate_with_survey(survey)
        if errors1:
            print(f"Response validation errors: {errors1}")
        else:
            print(f"Valid response with dependency met: {response1}")

        # Valid response with dependency not met (Q3 not required)
        response2 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R002",
            answers=[
                Answer(question_id="Q1", value=5),
                Answer(question_id="Q2", value="Yes"),
                # Q3 not answered because Q2 is not "No"
                Answer(question_id="Q4", value=["Features"])
            ]
        )

        errors2 = response2.validate_with_survey(survey)
        if errors2:
            print(f"Response validation errors: {errors2}")
        else:
            print(f"Valid response with dependency not met: {response2}")

        # Invalid response (missing required answer)
        response3 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R003",
            answers=[
                # Missing Q1
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive"),
                Answer(question_id="Q4", value=["Support"])
            ]
        )

        errors3 = response3.validate_with_survey(survey)
        if errors3:
            print(f"Response validation errors: {errors3}")
        else:
            print(f"Valid response: {response3}")

        # Invalid response (invalid rating)
        response4 = SurveyResponse(
            survey_id="SV001",
            respondent_id="R004",
            answers=[
                Answer(question_id="Q1", value=6),  # Rating out of range
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive"),
                Answer(question_id="Q4", value=["User Interface"])
            ]
        )

        errors4 = response4.validate_with_survey(survey)
        if errors4:
            print(f"Response validation errors: {errors4}")
        else:
            print(f"Valid response: {response4}")

    except ValidationError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    print("=== Lesson 4.1 Exercise Solutions Demo ===")

    demo_payment_system()
    demo_travel_booking()
    demo_product_inventory()
    demo_survey_form()

    print("\n=== Demo Complete ===")
