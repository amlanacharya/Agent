"""
Advanced Validation Patterns
---------------------------
This module demonstrates advanced validation patterns for complex data scenarios.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, EmailStr
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, date
import re
from enum import Enum


# 1. Cross-Field Validation
# ------------------------

class DateRange(BaseModel):
    """Date range with cross-field validation."""
    start_date: date
    end_date: date
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate that end_date is after start_date."""
        if self.start_date > self.end_date:
            raise ValueError("End date must be after start date")
        return self


class PasswordReset(BaseModel):
    """Password reset form with cross-field validation."""
    password: str
    password_confirm: str
    
    @model_validator(mode='after')
    def validate_passwords_match(self):
        """Validate that passwords match."""
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        
        # Additional password strength validation
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', self.password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', self.password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not re.search(r'[0-9]', self.password):
            raise ValueError("Password must contain at least one digit")
        
        if not re.search(r'[^A-Za-z0-9]', self.password):
            raise ValueError("Password must contain at least one special character")
        
        return self


# 2. Conditional Validation
# ------------------------

class PaymentMethod(str, Enum):
    """Payment method enum."""
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"


class CreditCardInfo(BaseModel):
    """Credit card information."""
    card_number: str
    expiration_date: str
    cvv: str
    
    @field_validator('card_number')
    @classmethod
    def validate_card_number(cls, v):
        """Validate credit card number format."""
        # Remove spaces and dashes
        v = v.replace(' ', '').replace('-', '')
        
        # Check if it's all digits
        if not v.isdigit():
            raise ValueError("Card number must contain only digits")
        
        # Check length (most cards are 13-19 digits)
        if not (13 <= len(v) <= 19):
            raise ValueError("Card number must be between 13 and 19 digits")
        
        # Luhn algorithm check (simplified)
        digits = [int(d) for d in v]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        if sum(digits) % 10 != 0:
            raise ValueError("Invalid card number (failed Luhn check)")
        
        return v
    
    @field_validator('expiration_date')
    @classmethod
    def validate_expiration_date(cls, v):
        """Validate expiration date format (MM/YY)."""
        if not re.match(r'^(0[1-9]|1[0-2])/\d{2}$', v):
            raise ValueError("Expiration date must be in format MM/YY")
        
        # Check if card is expired
        month, year = v.split('/')
        current_year = datetime.now().year % 100  # Get last two digits
        current_month = datetime.now().month
        
        if (int(year) < current_year or 
            (int(year) == current_year and int(month) < current_month)):
            raise ValueError("Card has expired")
        
        return v
    
    @field_validator('cvv')
    @classmethod
    def validate_cvv(cls, v):
        """Validate CVV format."""
        if not v.isdigit():
            raise ValueError("CVV must contain only digits")
        
        if not (3 <= len(v) <= 4):
            raise ValueError("CVV must be 3 or 4 digits")
        
        return v


class PayPalInfo(BaseModel):
    """PayPal information."""
    email: EmailStr


class BankTransferInfo(BaseModel):
    """Bank transfer information."""
    account_number: str
    routing_number: str
    account_name: str
    
    @field_validator('account_number')
    @classmethod
    def validate_account_number(cls, v):
        """Validate account number format."""
        if not v.isdigit():
            raise ValueError("Account number must contain only digits")
        
        if not (8 <= len(v) <= 17):
            raise ValueError("Account number must be between 8 and 17 digits")
        
        return v
    
    @field_validator('routing_number')
    @classmethod
    def validate_routing_number(cls, v):
        """Validate routing number format."""
        if not v.isdigit():
            raise ValueError("Routing number must contain only digits")
        
        if len(v) != 9:
            raise ValueError("Routing number must be 9 digits")
        
        return v


class Payment(BaseModel):
    """Payment model with conditional validation based on payment method."""
    amount: float
    currency: str
    payment_method: PaymentMethod
    credit_card_info: Optional[CreditCardInfo] = None
    paypal_info: Optional[PayPalInfo] = None
    bank_transfer_info: Optional[BankTransferInfo] = None
    
    @model_validator(mode='after')
    def validate_payment_info(self):
        """Validate that the appropriate payment info is provided."""
        if self.payment_method == PaymentMethod.CREDIT_CARD and not self.credit_card_info:
            raise ValueError("Credit card information is required for credit card payments")
        
        if self.payment_method == PaymentMethod.PAYPAL and not self.paypal_info:
            raise ValueError("PayPal information is required for PayPal payments")
        
        if self.payment_method == PaymentMethod.BANK_TRANSFER and not self.bank_transfer_info:
            raise ValueError("Bank transfer information is required for bank transfer payments")
        
        return self


# 3. Content-Based Validation
# --------------------------

class ContentType(str, Enum):
    """Content type enum."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


class Content(BaseModel):
    """Content model with type-specific validation."""
    content_type: ContentType
    title: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @model_validator(mode='after')
    def validate_content(self):
        """Validate content based on content type."""
        if self.content_type == ContentType.TEXT and not self.text_content:
            raise ValueError("Text content is required for text content type")
        
        if self.content_type != ContentType.TEXT and not self.file_path:
            raise ValueError(f"File path is required for {self.content_type.value} content type")
        
        # Validate file extensions based on content type
        if self.file_path:
            if self.content_type == ContentType.IMAGE and not self.file_path.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                raise ValueError("Invalid image file extension")
            
            if self.content_type == ContentType.VIDEO and not self.file_path.lower().endswith(
                    ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv')):
                raise ValueError("Invalid video file extension")
            
            if self.content_type == ContentType.DOCUMENT and not self.file_path.lower().endswith(
                    ('.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt')):
                raise ValueError("Invalid document file extension")
        
        return self


# 4. Context-Dependent Validation
# -----------------------------

class BookingRequest(BaseModel):
    """Booking request with context-dependent validation."""
    room_id: int
    check_in_date: date
    check_out_date: date
    guest_count: int
    
    @model_validator(mode='after')
    def validate_booking(self):
        """Basic validation without context."""
        if self.check_in_date >= self.check_out_date:
            raise ValueError("Check-out must be after check-in")
        
        if (self.check_out_date - self.check_in_date).days > 14:
            raise ValueError("Maximum stay is 14 days")
        
        return self


class BookingSystem:
    """Booking system with context-dependent validation."""
    
    def __init__(self):
        """Initialize booking system with room capacities and bookings."""
        self.room_capacities = {
            101: 2,  # Room 101 can accommodate 2 guests
            102: 4,  # Room 102 can accommodate 4 guests
            103: 6,  # Room 103 can accommodate 6 guests
        }
        
        self.bookings = {
            # room_id: [(check_in_date, check_out_date), ...]
            101: [(date(2023, 6, 1), date(2023, 6, 5)), 
                  (date(2023, 6, 10), date(2023, 6, 15))],
            102: [(date(2023, 6, 5), date(2023, 6, 10))],
            103: [],
        }
    
    def validate_booking(self, booking: BookingRequest) -> List[str]:
        """
        Validate booking with context.
        
        Args:
            booking: BookingRequest instance
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check if room exists
        if booking.room_id not in self.room_capacities:
            errors.append(f"Room {booking.room_id} does not exist")
            return errors
        
        # Check room capacity
        if booking.guest_count > self.room_capacities[booking.room_id]:
            errors.append(f"Room {booking.room_id} can only accommodate "
                         f"{self.room_capacities[booking.room_id]} guests")
        
        # Check availability
        for existing_check_in, existing_check_out in self.bookings.get(booking.room_id, []):
            if (booking.check_in_date < existing_check_out and 
                booking.check_out_date > existing_check_in):
                errors.append(f"Room {booking.room_id} is not available for the selected dates")
                break
        
        return errors


# 5. Dynamic Validation
# -------------------

def create_range_validator(min_value, max_value):
    """
    Create a validator function for a range.
    
    Args:
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validator function
    """
    def validate_range(v):
        if v < min_value or v > max_value:
            raise ValueError(f"Value must be between {min_value} and {max_value}")
        return v
    
    return validate_range


def create_list_length_validator(min_length, max_length):
    """
    Create a validator function for list length.
    
    Args:
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Validator function
    """
    def validate_list_length(v):
        if len(v) < min_length:
            raise ValueError(f"List must have at least {min_length} items")
        if len(v) > max_length:
            raise ValueError(f"List must have at most {max_length} items")
        return v
    
    return validate_list_length


class DynamicModel(BaseModel):
    """Model with dynamically added validators."""
    value: int
    items: List[str] = []


# 6. Practical Example: Advanced Form Validation
# -------------------------------------------

class EducationLevel(str, Enum):
    """Education level enum."""
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"


class EmploymentStatus(str, Enum):
    """Employment status enum."""
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    STUDENT = "student"
    RETIRED = "retired"


class EmploymentInfo(BaseModel):
    """Employment information with conditional validation."""
    status: EmploymentStatus
    employer: Optional[str] = None
    position: Optional[str] = None
    years_at_job: Optional[int] = None
    annual_income: Optional[float] = None
    
    @model_validator(mode='after')
    def validate_employment_info(self):
        """Validate employment information based on status."""
        if self.status in [EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED]:
            if not self.employer:
                raise ValueError("Employer is required for employed status")
            
            if not self.position:
                raise ValueError("Position is required for employed status")
            
            if self.years_at_job is None:
                raise ValueError("Years at job is required for employed status")
            
            if self.annual_income is None:
                raise ValueError("Annual income is required for employed status")
        
        return self


class Address(BaseModel):
    """Address model with validation."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, v, info):
        """Validate zip code format based on country."""
        # Get values from other fields
        values = info.data
        country = values.get('country', '')
        
        if country.lower() == 'usa' or country.lower() == 'united states':
            # US ZIP code validation
            if not re.match(r'^\d{5}(-\d{4})?$', v):
                raise ValueError("Invalid US ZIP code format")
        elif country.lower() == 'canada':
            # Canadian postal code validation
            if not re.match(r'^[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d$', v):
                raise ValueError("Invalid Canadian postal code format")
        
        return v


class LoanApplication(BaseModel):
    """Loan application with complex validation rules."""
    applicant_name: str
    email: EmailStr
    phone: str
    date_of_birth: date
    address: Address
    education_level: EducationLevel
    employment_info: EmploymentInfo
    loan_amount: float
    loan_purpose: str
    credit_score: Optional[int] = None
    has_existing_loans: bool = False
    existing_loan_amount: Optional[float] = None
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        """Validate phone number format."""
        # Remove non-digit characters
        digits_only = re.sub(r'\D', '', v)
        
        # Check if it has a valid number of digits
        if not (10 <= len(digits_only) <= 15):
            raise ValueError("Phone number must have between 10 and 15 digits")
        
        # Format as (XXX) XXX-XXXX for US numbers
        if len(digits_only) == 10:
            return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
        
        return v
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_age(cls, v):
        """Validate that applicant is at least 18 years old."""
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        
        if age < 18:
            raise ValueError("Applicant must be at least 18 years old")
        
        if age > 100:
            raise ValueError("Invalid date of birth")
        
        return v
    
    @model_validator(mode='after')
    def validate_loan_application(self):
        """Validate loan application with complex business rules."""
        # Validate existing loan information
        if self.has_existing_loans and self.existing_loan_amount is None:
            raise ValueError("Existing loan amount is required if has_existing_loans is True")
        
        if not self.has_existing_loans and self.existing_loan_amount is not None:
            raise ValueError("Existing loan amount should be None if has_existing_loans is False")
        
        # Validate loan amount based on income
        if (self.employment_info.status in [EmploymentStatus.EMPLOYED, EmploymentStatus.SELF_EMPLOYED] 
                and self.employment_info.annual_income is not None):
            max_loan_amount = self.employment_info.annual_income * 0.5
            if self.loan_amount > max_loan_amount:
                raise ValueError(f"Loan amount cannot exceed 50% of annual income (${max_loan_amount:.2f})")
        
        # Validate loan amount based on employment status
        if self.employment_info.status == EmploymentStatus.UNEMPLOYED and self.loan_amount > 10000:
            raise ValueError("Maximum loan amount for unemployed applicants is $10,000")
        
        # Validate credit score if provided
        if self.credit_score is not None:
            if not (300 <= self.credit_score <= 850):
                raise ValueError("Credit score must be between 300 and 850")
            
            if self.credit_score < 600 and self.loan_amount > 5000:
                raise ValueError("Maximum loan amount for credit score below 600 is $5,000")
        
        return self


# Example usage
if __name__ == "__main__":
    # Cross-field validation
    try:
        date_range = DateRange(start_date=date(2023, 6, 1), end_date=date(2023, 5, 1))
    except ValueError as e:
        print(f"DateRange validation error: {e}")
    
    try:
        password_reset = PasswordReset(
            password="weakpw", 
            password_confirm="weakpw"
        )
    except ValueError as e:
        print(f"PasswordReset validation error: {e}")
    
    # Conditional validation
    try:
        payment = Payment(
            amount=100.0,
            currency="USD",
            payment_method=PaymentMethod.CREDIT_CARD,
            # Missing credit_card_info
        )
    except ValueError as e:
        print(f"Payment validation error: {e}")
    
    # Content-based validation
    try:
        content = Content(
            content_type=ContentType.IMAGE,
            title="My Image",
            # Missing file_path
        )
    except ValueError as e:
        print(f"Content validation error: {e}")
    
    # Context-dependent validation
    booking_system = BookingSystem()
    booking = BookingRequest(
        room_id=101,
        check_in_date=date(2023, 6, 2),
        check_out_date=date(2023, 6, 4),
        guest_count=3  # Exceeds capacity
    )
    errors = booking_system.validate_booking(booking)
    if errors:
        print(f"Booking validation errors: {errors}")
    
    # Dynamic validation
    # Add validators dynamically
    setattr(DynamicModel, 'validate_value', field_validator('value')(create_range_validator(0, 100)))
    setattr(DynamicModel, 'validate_items', field_validator('items')(create_list_length_validator(1, 5)))
    
    try:
        model = DynamicModel(value=150, items=[])  # Will raise validation errors
    except ValueError as e:
        print(f"DynamicModel validation error: {e}")
    
    # Advanced form validation
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
            # Missing existing_loan_amount
        )
    except ValueError as e:
        print(f"LoanApplication validation error: {e}")
