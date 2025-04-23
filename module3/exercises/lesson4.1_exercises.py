"""
Lesson 4.1 Exercise Solutions: Complex Validation Scenarios
--------------------------------------------------------
This module contains solutions for the exercises in Lesson 4.1.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, EmailStr
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, date
import re
from enum import Enum


# Exercise 1: Create a PaymentSystem model with conditional validation based on payment method
# ---------------------------------------------------------------------------------------

class PaymentMethodType(str, Enum):
    """Payment method type enum."""
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"


class CreditCardDetails(BaseModel):
    """Credit card payment details."""
    card_number: str
    cardholder_name: str
    expiration_month: int
    expiration_year: int
    cvv: str
    billing_address: Optional[str] = None
    
    @field_validator('card_number')
    @classmethod
    def validate_card_number(cls, v):
        """Validate credit card number using Luhn algorithm."""
        # Remove spaces and dashes
        v = v.replace(' ', '').replace('-', '')
        
        if not v.isdigit():
            raise ValueError("Card number must contain only digits")
        
        if not (13 <= len(v) <= 19):
            raise ValueError("Card number must be between 13 and 19 digits")
        
        # Luhn algorithm check
        digits = [int(d) for d in v]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        if sum(digits) % 10 != 0:
            raise ValueError("Invalid card number (failed Luhn check)")
        
        return v
    
    @field_validator('expiration_month')
    @classmethod
    def validate_expiration_month(cls, v):
        """Validate expiration month."""
        if not (1 <= v <= 12):
            raise ValueError("Expiration month must be between 1 and 12")
        return v
    
    @field_validator('expiration_year')
    @classmethod
    def validate_expiration_year(cls, v):
        """Validate expiration year."""
        current_year = datetime.now().year
        if v < current_year:
            raise ValueError("Card has expired")
        if v > current_year + 20:
            raise ValueError("Expiration year too far in the future")
        return v
    
    @model_validator(mode='after')
    def validate_expiration(self):
        """Validate that the card has not expired."""
        current_date = datetime.now()
        if (self.expiration_year == current_date.year and 
            self.expiration_month < current_date.month):
            raise ValueError("Card has expired")
        return self
    
    @field_validator('cvv')
    @classmethod
    def validate_cvv(cls, v):
        """Validate CVV."""
        if not v.isdigit():
            raise ValueError("CVV must contain only digits")
        
        if not (3 <= len(v) <= 4):
            raise ValueError("CVV must be 3 or 4 digits")
        
        return v


class PayPalDetails(BaseModel):
    """PayPal payment details."""
    email: EmailStr
    
    @field_validator('email')
    @classmethod
    def validate_paypal_email(cls, v):
        """Validate that the email is not from a temporary email provider."""
        temporary_domains = ['tempmail.com', 'throwaway.com', 'mailinator.com']
        domain = v.split('@')[-1]
        if domain in temporary_domains:
            raise ValueError(f"Email from {domain} is not allowed for PayPal payments")
        return v


class BankTransferDetails(BaseModel):
    """Bank transfer payment details."""
    account_holder: str
    account_number: str
    routing_number: str
    bank_name: str
    reference: Optional[str] = None
    
    @field_validator('account_number')
    @classmethod
    def validate_account_number(cls, v):
        """Validate account number."""
        if not v.isdigit():
            raise ValueError("Account number must contain only digits")
        
        if len(v) < 8:
            raise ValueError("Account number must be at least 8 digits")
        
        return v
    
    @field_validator('routing_number')
    @classmethod
    def validate_routing_number(cls, v):
        """Validate routing number."""
        if not v.isdigit():
            raise ValueError("Routing number must contain only digits")
        
        if len(v) != 9:
            raise ValueError("Routing number must be 9 digits")
        
        # Check if the routing number is valid using the checksum algorithm
        digits = [int(d) for d in v]
        checksum = (
            3 * (digits[0] + digits[3] + digits[6]) +
            7 * (digits[1] + digits[4] + digits[7]) +
            (digits[2] + digits[5] + digits[8])
        ) % 10
        
        if checksum != 0:
            raise ValueError("Invalid routing number (failed checksum)")
        
        return v


class PaymentSystem(BaseModel):
    """Payment system with conditional validation based on payment method."""
    transaction_id: str
    amount: float
    currency: str
    payment_method: PaymentMethodType
    credit_card_details: Optional[CreditCardDetails] = None
    paypal_details: Optional[PayPalDetails] = None
    bank_transfer_details: Optional[BankTransferDetails] = None
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate payment amount."""
        if v <= 0:
            raise ValueError("Payment amount must be greater than zero")
        return v
    
    @model_validator(mode='after')
    def validate_payment_details(self):
        """Validate that the appropriate payment details are provided."""
        if self.payment_method == PaymentMethodType.CREDIT_CARD and not self.credit_card_details:
            raise ValueError("Credit card details are required for credit card payments")
        
        if self.payment_method == PaymentMethodType.PAYPAL and not self.paypal_details:
            raise ValueError("PayPal details are required for PayPal payments")
        
        if self.payment_method == PaymentMethodType.BANK_TRANSFER and not self.bank_transfer_details:
            raise ValueError("Bank transfer details are required for bank transfer payments")
        
        # Validate that only the relevant payment details are provided
        if self.payment_method == PaymentMethodType.CREDIT_CARD:
            if self.paypal_details or self.bank_transfer_details:
                raise ValueError("Only credit card details should be provided for credit card payments")
        
        if self.payment_method == PaymentMethodType.PAYPAL:
            if self.credit_card_details or self.bank_transfer_details:
                raise ValueError("Only PayPal details should be provided for PayPal payments")
        
        if self.payment_method == PaymentMethodType.BANK_TRANSFER:
            if self.credit_card_details or self.paypal_details:
                raise ValueError("Only bank transfer details should be provided for bank transfer payments")
        
        return self


# Exercise 2: Implement a TravelBooking model with different validation rules
# -----------------------------------------------------------------------

class TravelType(str, Enum):
    """Travel type enum."""
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"


class PassengerType(str, Enum):
    """Passenger type enum."""
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"
    SENIOR = "senior"


class PassengerInfo(BaseModel):
    """Passenger information."""
    first_name: str
    last_name: str
    date_of_birth: date
    passenger_type: PassengerType
    passport_number: Optional[str] = None
    passport_expiry: Optional[date] = None
    nationality: Optional[str] = None
    
    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_name(cls, v):
        """Validate name."""
        if not v:
            raise ValueError("Name cannot be empty")
        
        if not re.match(r'^[A-Za-z\s\-\']+$', v):
            raise ValueError("Name can only contain letters, spaces, hyphens, and apostrophes")
        
        return v
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_date_of_birth(cls, v):
        """Validate date of birth."""
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        
        if age > 120:
            raise ValueError("Invalid date of birth")
        
        return v
    
    @model_validator(mode='after')
    def validate_passenger_type(self):
        """Validate passenger type based on age."""
        today = date.today()
        age = today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        
        if self.passenger_type == PassengerType.INFANT and age >= 2:
            raise ValueError("Infant passengers must be under 2 years old")
        
        if self.passenger_type == PassengerType.CHILD and (age < 2 or age >= 12):
            raise ValueError("Child passengers must be between 2 and 11 years old")
        
        if self.passenger_type == PassengerType.ADULT and (age < 12 or age >= 65):
            raise ValueError("Adult passengers must be between 12 and 64 years old")
        
        if self.passenger_type == PassengerType.SENIOR and age < 65:
            raise ValueError("Senior passengers must be 65 years or older")
        
        return self


class TravelBooking(BaseModel):
    """Travel booking with different validation rules based on travel type."""
    booking_reference: str
    travel_type: TravelType
    origin: str
    destination: str
    departure_date: date
    return_date: Optional[date] = None
    passengers: List[PassengerInfo]
    contact_email: EmailStr
    contact_phone: str
    
    @field_validator('booking_reference')
    @classmethod
    def validate_booking_reference(cls, v):
        """Validate booking reference."""
        if not re.match(r'^[A-Z0-9]{6}$', v):
            raise ValueError("Booking reference must be 6 uppercase letters or digits")
        return v
    
    @field_validator('origin', 'destination')
    @classmethod
    def validate_location(cls, v):
        """Validate location code."""
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError("Location must be a 3-letter IATA airport code")
        return v
    
    @model_validator(mode='after')
    def validate_travel_booking(self):
        """Validate travel booking based on travel type."""
        # Validate dates
        if self.departure_date < date.today():
            raise ValueError("Departure date cannot be in the past")
        
        if self.return_date and self.return_date < self.departure_date:
            raise ValueError("Return date must be after departure date")
        
        # Validate passengers
        if not self.passengers:
            raise ValueError("At least one passenger is required")
        
        if len(self.passengers) > 9:
            raise ValueError("Maximum 9 passengers allowed per booking")
        
        # Validate international travel requirements
        if self.travel_type == TravelType.INTERNATIONAL:
            # Check if all passengers have passport information
            for i, passenger in enumerate(self.passengers):
                if not passenger.passport_number:
                    raise ValueError(f"Passport number is required for passenger {i+1} for international travel")
                
                if not passenger.passport_expiry:
                    raise ValueError(f"Passport expiry date is required for passenger {i+1} for international travel")
                
                if not passenger.nationality:
                    raise ValueError(f"Nationality is required for passenger {i+1} for international travel")
                
                # Check passport validity
                if passenger.passport_expiry:
                    # Passport should be valid for at least 6 months after return date
                    min_validity_date = self.return_date or self.departure_date
                    min_validity_date = date(min_validity_date.year, min_validity_date.month, min_validity_date.day)
                    six_months_later = date(
                        min_validity_date.year + (1 if min_validity_date.month > 6 else 0),
                        (min_validity_date.month + 6) % 12 or 12,
                        min_validity_date.day
                    )
                    
                    if passenger.passport_expiry < six_months_later:
                        raise ValueError(
                            f"Passport for passenger {i+1} must be valid for at least 6 months "
                            f"after the return date"
                        )
        
        return self


# Exercise 3: Build a ProductInventory model with dynamic validation rules
# ---------------------------------------------------------------------

class ProductCategory(str, Enum):
    """Product category enum."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    TOYS = "toys"


class ProductInventory(BaseModel):
    """Product inventory with dynamic validation rules."""
    product_id: str
    name: str
    description: Optional[str] = None
    category: ProductCategory
    price: float
    quantity: int
    attributes: Dict[str, Any] = {}
    
    @field_validator('product_id')
    @classmethod
    def validate_product_id(cls, v):
        """Validate product ID."""
        if not re.match(r'^[A-Z]{2}\d{6}$', v):
            raise ValueError("Product ID must be 2 uppercase letters followed by 6 digits")
        return v
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        """Validate price."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity."""
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        return v
    
    @model_validator(mode='after')
    def validate_category_attributes(self):
        """Validate attributes based on product category."""
        # Define required attributes for each category
        category_validators = {
            ProductCategory.ELECTRONICS: self._validate_electronics,
            ProductCategory.CLOTHING: self._validate_clothing,
            ProductCategory.FOOD: self._validate_food,
            ProductCategory.BOOKS: self._validate_books,
            ProductCategory.TOYS: self._validate_toys
        }
        
        # Run the appropriate validator for the product category
        if self.category in category_validators:
            category_validators[self.category]()
        
        return self
    
    def _validate_electronics(self):
        """Validate electronics attributes."""
        required_attrs = ['brand', 'model', 'warranty_months']
        for attr in required_attrs:
            if attr not in self.attributes:
                raise ValueError(f"'{attr}' is required for electronics products")
        
        # Validate warranty_months
        warranty_months = self.attributes.get('warranty_months')
        if not isinstance(warranty_months, int) or warranty_months < 0:
            raise ValueError("warranty_months must be a non-negative integer")
        
        # Validate voltage if present
        if 'voltage' in self.attributes:
            voltage = self.attributes['voltage']
            if not isinstance(voltage, (int, float)) or voltage <= 0:
                raise ValueError("voltage must be a positive number")
    
    def _validate_clothing(self):
        """Validate clothing attributes."""
        required_attrs = ['size', 'color', 'material']
        for attr in required_attrs:
            if attr not in self.attributes:
                raise ValueError(f"'{attr}' is required for clothing products")
        
        # Validate size
        valid_sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        size = self.attributes.get('size')
        if isinstance(size, str) and size.upper() not in valid_sizes:
            raise ValueError(f"size must be one of {valid_sizes}")
    
    def _validate_food(self):
        """Validate food attributes."""
        required_attrs = ['expiration_date', 'allergens']
        for attr in required_attrs:
            if attr not in self.attributes:
                raise ValueError(f"'{attr}' is required for food products")
        
        # Validate expiration_date
        expiration_date = self.attributes.get('expiration_date')
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            if exp_date < date.today():
                raise ValueError("expiration_date cannot be in the past")
        except (ValueError, TypeError):
            raise ValueError("expiration_date must be in format YYYY-MM-DD")
        
        # Validate allergens
        allergens = self.attributes.get('allergens')
        if not isinstance(allergens, list):
            raise ValueError("allergens must be a list")
    
    def _validate_books(self):
        """Validate books attributes."""
        required_attrs = ['author', 'isbn', 'publisher']
        for attr in required_attrs:
            if attr not in self.attributes:
                raise ValueError(f"'{attr}' is required for book products")
        
        # Validate ISBN
        isbn = self.attributes.get('isbn')
        if not isinstance(isbn, str) or not re.match(r'^\d{10}(\d{3})?$', isbn.replace('-', '')):
            raise ValueError("isbn must be a valid 10 or 13 digit ISBN")
    
    def _validate_toys(self):
        """Validate toys attributes."""
        required_attrs = ['age_range', 'safety_certified']
        for attr in required_attrs:
            if attr not in self.attributes:
                raise ValueError(f"'{attr}' is required for toy products")
        
        # Validate age_range
        age_range = self.attributes.get('age_range')
        if not isinstance(age_range, str) or not re.match(r'^\d+\+$', age_range):
            raise ValueError("age_range must be in format '3+', '5+', etc.")
        
        # Validate safety_certified
        safety_certified = self.attributes.get('safety_certified')
        if not isinstance(safety_certified, bool):
            raise ValueError("safety_certified must be a boolean")


# Exercise 4: Create a survey form validation system with conditional rules
# ---------------------------------------------------------------------

class QuestionType(str, Enum):
    """Question type enum."""
    TEXT = "text"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    RATING = "rating"
    DATE = "date"


class Question(BaseModel):
    """Survey question with validation."""
    id: str
    text: str
    type: QuestionType
    required: bool = True
    options: Optional[List[str]] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    depends_on: Optional[str] = None
    depends_on_value: Optional[Any] = None
    
    @model_validator(mode='after')
    def validate_question(self):
        """Validate question based on type."""
        if self.type in [QuestionType.SINGLE_CHOICE, QuestionType.MULTIPLE_CHOICE] and not self.options:
            raise ValueError(f"Options are required for {self.type} questions")
        
        if self.type == QuestionType.RATING:
            if self.min_rating is None or self.max_rating is None:
                raise ValueError("min_rating and max_rating are required for rating questions")
            
            if self.min_rating >= self.max_rating:
                raise ValueError("min_rating must be less than max_rating")
        
        if (self.depends_on is not None) != (self.depends_on_value is not None):
            raise ValueError("Both depends_on and depends_on_value must be provided together")
        
        return self


class Answer(BaseModel):
    """Survey answer with validation."""
    question_id: str
    value: Any
    
    def validate_with_question(self, question: Question) -> List[str]:
        """
        Validate answer against question.
        
        Args:
            question: Question to validate against
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check if question is required and answer is empty
        if question.required and (self.value is None or self.value == "" or self.value == []):
            errors.append(f"Question {question.id} requires an answer")
            return errors
        
        # Skip further validation if answer is empty and question is not required
        if not question.required and (self.value is None or self.value == "" or self.value == []):
            return errors
        
        # Validate based on question type
        if question.type == QuestionType.TEXT:
            if not isinstance(self.value, str):
                errors.append(f"Answer for question {question.id} must be a string")
        
        elif question.type == QuestionType.SINGLE_CHOICE:
            if not isinstance(self.value, str) or self.value not in question.options:
                errors.append(f"Answer for question {question.id} must be one of the provided options")
        
        elif question.type == QuestionType.MULTIPLE_CHOICE:
            if not isinstance(self.value, list):
                errors.append(f"Answer for question {question.id} must be a list")
            else:
                for option in self.value:
                    if option not in question.options:
                        errors.append(f"Answer for question {question.id} contains an invalid option: {option}")
        
        elif question.type == QuestionType.RATING:
            try:
                rating = int(self.value)
                if rating < question.min_rating or rating > question.max_rating:
                    errors.append(
                        f"Rating for question {question.id} must be between "
                        f"{question.min_rating} and {question.max_rating}"
                    )
            except (ValueError, TypeError):
                errors.append(f"Rating for question {question.id} must be a number")
        
        elif question.type == QuestionType.DATE:
            try:
                datetime.strptime(self.value, '%Y-%m-%d')
            except (ValueError, TypeError):
                errors.append(f"Date for question {question.id} must be in format YYYY-MM-DD")
        
        return errors


class SurveyForm(BaseModel):
    """Survey form with conditional validation rules."""
    id: str
    title: str
    description: Optional[str] = None
    questions: List[Question]
    
    @model_validator(mode='after')
    def validate_questions(self):
        """Validate questions for consistency."""
        question_ids = set()
        dependent_questions = {}
        
        for question in self.questions:
            # Check for duplicate question IDs
            if question.id in question_ids:
                raise ValueError(f"Duplicate question ID: {question.id}")
            question_ids.add(question.id)
            
            # Track dependencies
            if question.depends_on:
                dependent_questions[question.id] = question.depends_on
        
        # Check for circular dependencies
        for question_id, depends_on in dependent_questions.items():
            visited = set()
            current = depends_on
            
            while current in dependent_questions:
                if current in visited:
                    raise ValueError(f"Circular dependency detected for question {question_id}")
                visited.add(current)
                current = dependent_questions[current]
            
            # Check if dependency exists
            if depends_on not in question_ids:
                raise ValueError(f"Question {question_id} depends on non-existent question {depends_on}")
        
        return self


class SurveyResponse(BaseModel):
    """Survey response with conditional validation."""
    survey_id: str
    respondent_id: str
    answers: List[Answer]
    
    def validate_with_survey(self, survey: SurveyForm) -> List[str]:
        """
        Validate response against survey.
        
        Args:
            survey: Survey to validate against
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Create maps for easier lookup
        question_map = {q.id: q for q in survey.questions}
        answer_map = {a.question_id: a for a in self.answers}
        
        # Check for answers to non-existent questions
        for answer in self.answers:
            if answer.question_id not in question_map:
                errors.append(f"Answer provided for non-existent question: {answer.question_id}")
        
        # Check if all required questions are answered, considering dependencies
        for question in survey.questions:
            # Skip validation if question depends on another question's answer
            if question.depends_on:
                # Get the answer to the dependency question
                dependency_answer = answer_map.get(question.depends_on)
                
                # Skip this question if the dependency is not met
                if not dependency_answer or dependency_answer.value != question.depends_on_value:
                    continue
            
            # Check if question is answered
            if question.id not in answer_map:
                if question.required:
                    errors.append(f"Required question not answered: {question.id}")
                continue
            
            # Validate answer against question
            answer = answer_map[question.id]
            answer_errors = answer.validate_with_question(question)
            errors.extend(answer_errors)
        
        return errors


# Example usage
if __name__ == "__main__":
    # Exercise 1: PaymentSystem
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
        print(f"Valid payment: {payment}")
    except ValueError as e:
        print(f"Payment validation error: {e}")
    
    # Exercise 2: TravelBooking
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
        print(f"Valid booking: {booking}")
    except ValueError as e:
        print(f"Booking validation error: {e}")
    
    # Exercise 3: ProductInventory
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
        print(f"Valid product: {product}")
    except ValueError as e:
        print(f"Product validation error: {e}")
    
    # Exercise 4: SurveyForm
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
                )
            ]
        )
        print(f"Valid survey: {survey}")
        
        response = SurveyResponse(
            survey_id="SV001",
            respondent_id="R001",
            answers=[
                Answer(question_id="Q1", value=4),
                Answer(question_id="Q2", value="No"),
                Answer(question_id="Q3", value="Service was too expensive")
            ]
        )
        
        errors = response.validate_with_survey(survey)
        if errors:
            print(f"Response validation errors: {errors}")
        else:
            print(f"Valid response: {response}")
    except ValueError as e:
        print(f"Survey validation error: {e}")
