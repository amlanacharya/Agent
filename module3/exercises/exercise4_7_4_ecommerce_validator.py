"""
Exercise 4.7.4: E-commerce Domain Validation
-------------------------------------------
This module implements validation patterns specific to e-commerce systems, focusing on:
1. Product validation (pricing, inventory, categories)
2. Order validation (items, totals, shipping)
3. Customer validation (profiles, addresses, payment methods)
4. Promotion validation (discounts, coupons, eligibility)

These validation patterns help ensure that e-commerce systems maintain data integrity
and business rule compliance across the entire purchase lifecycle.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Literal, Set, Any, Union
from enum import Enum
from datetime import datetime, date
import re
import uuid


class ProductCategory(str, Enum):
    """Product categories in an e-commerce system."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME = "home"
    BEAUTY = "beauty"
    BOOKS = "books"
    TOYS = "toys"
    SPORTS = "sports"
    GROCERY = "grocery"
    OTHER = "other"


class ProductStatus(str, Enum):
    """Status of a product in the inventory."""
    ACTIVE = "active"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"
    DRAFT = "draft"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"


class ShippingMethod(str, Enum):
    """Available shipping methods."""
    STANDARD = "standard"
    EXPRESS = "express"
    OVERNIGHT = "overnight"
    STORE_PICKUP = "store_pickup"
    DIGITAL_DELIVERY = "digital_delivery"


class PaymentMethod(str, Enum):
    """Available payment methods."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "cryptocurrency"
    GIFT_CARD = "gift_card"
    CASH_ON_DELIVERY = "cash_on_delivery"


class OrderStatus(str, Enum):
    """Status of an order."""
    PENDING = "pending"
    PAID = "paid"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    ON_HOLD = "on_hold"


class Address(BaseModel):
    """Model for a physical address."""
    street_line1: str
    street_line2: Optional[str] = None
    city: str
    state: str
    postal_code: str
    country: str
    is_default: bool = False

    @field_validator('postal_code')
    @classmethod
    def validate_postal_code(cls, v, info):
        """Validate postal code format based on country."""
        country = info.data.get('country', '').upper()
        
        # Simple validation patterns for common countries
        if country == 'US' and not re.match(r'^\d{5}(-\d{4})?$', v):
            raise ValueError("Invalid US postal code format")
        elif country == 'CA' and not re.match(r'^[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d$', v):
            raise ValueError("Invalid Canadian postal code format")
        elif country == 'UK' and not re.match(r'^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$', v):
            raise ValueError("Invalid UK postal code format")
        
        return v


class Price(BaseModel):
    """Model for a price with currency."""
    amount: float = Field(ge=0.0)
    currency: Currency = Currency.USD

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate price amount."""
        if v < 0:
            raise ValueError("Price amount cannot be negative")
        return round(v, 2)  # Round to 2 decimal places


class Product(BaseModel):
    """Model for a product in an e-commerce system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    sku: str
    price: Price
    sale_price: Optional[Price] = None
    category: ProductCategory
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    tags: List[str] = []
    images: List[str] = []
    inventory_count: int = Field(ge=0)
    status: ProductStatus = ProductStatus.ACTIVE
    weight: Optional[float] = None  # in kg
    dimensions: Optional[Dict[str, float]] = None  # length, width, height in cm
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_digital: bool = False
    is_taxable: bool = True
    tax_code: Optional[str] = None

    @model_validator(mode='after')
    def validate_product(self):
        """Validate product data consistency."""
        # Sale price should be less than regular price
        if self.sale_price and self.sale_price.amount >= self.price.amount:
            raise ValueError("Sale price must be less than regular price")
        
        # Digital products shouldn't have weight or dimensions
        if self.is_digital and (self.weight is not None or self.dimensions is not None):
            raise ValueError("Digital products should not have weight or dimensions")
        
        # Physical products should have weight for shipping calculations
        if not self.is_digital and self.weight is None:
            print(f"Warning: Physical product '{self.name}' is missing weight information")
        
        # Check inventory status consistency
        if self.inventory_count == 0 and self.status == ProductStatus.ACTIVE:
            print(f"Warning: Product '{self.name}' has zero inventory but is marked as active")
        
        # Check for missing images
        if not self.images:
            print(f"Warning: Product '{self.name}' has no images")
        
        return self


class Customer(BaseModel):
    """Model for a customer in an e-commerce system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    addresses: List[Address] = []
    default_payment_method: Optional[PaymentMethod] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    marketing_preferences: Dict[str, bool] = {"email": False, "sms": False}
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Validate email format."""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email format")
        return v.lower()  # Normalize email to lowercase
    
    @model_validator(mode='after')
    def validate_customer(self):
        """Validate customer data consistency."""
        # Check for at least one address
        if not self.addresses:
            print(f"Warning: Customer {self.first_name} {self.last_name} has no addresses")
        
        # Check for default address
        if self.addresses and not any(addr.is_default for addr in self.addresses):
            print(f"Warning: Customer {self.first_name} {self.last_name} has no default address")
        
        return self


class OrderItem(BaseModel):
    """Model for an item in an order."""
    product_id: str
    product_name: str
    quantity: int = Field(ge=1)
    unit_price: Price
    subtotal: Price
    is_gift: bool = False
    gift_message: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_order_item(self):
        """Validate order item calculations."""
        # Check subtotal calculation
        expected_subtotal = round(self.unit_price.amount * self.quantity, 2)
        if abs(self.subtotal.amount - expected_subtotal) > 0.01:  # Allow for small rounding differences
            raise ValueError(f"Subtotal {self.subtotal.amount} doesn't match unit price × quantity ({expected_subtotal})")
        
        # Gift message should only be present for gift items
        if not self.is_gift and self.gift_message:
            raise ValueError("Gift message provided for non-gift item")
        
        return self


class Order(BaseModel):
    """Model for an order in an e-commerce system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    items: List[OrderItem]
    subtotal: Price
    tax: Price
    shipping_cost: Price
    total: Price
    shipping_address: Address
    billing_address: Optional[Address] = None
    payment_method: PaymentMethod
    shipping_method: ShippingMethod
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_order(self):
        """Validate order data consistency."""
        # Check if order has items
        if not self.items:
            raise ValueError("Order must have at least one item")
        
        # Calculate expected subtotal
        expected_subtotal = sum(item.subtotal.amount for item in self.items)
        if abs(self.subtotal.amount - expected_subtotal) > 0.01:
            raise ValueError(f"Order subtotal {self.subtotal.amount} doesn't match sum of item subtotals ({expected_subtotal})")
        
        # Calculate expected total
        expected_total = self.subtotal.amount + self.tax.amount + self.shipping_cost.amount
        if abs(self.total.amount - expected_total) > 0.01:
            raise ValueError(f"Order total {self.total.amount} doesn't match subtotal + tax + shipping ({expected_total})")
        
        # Use billing address as shipping address if not provided
        if self.billing_address is None:
            self.billing_address = self.shipping_address
        
        # Digital delivery shouldn't have shipping cost
        if self.shipping_method == ShippingMethod.DIGITAL_DELIVERY and self.shipping_cost.amount > 0:
            raise ValueError("Digital delivery should not have shipping costs")
        
        return self


class Promotion(BaseModel):
    """Model for a promotion or discount."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    discount_type: Literal["percentage", "fixed_amount", "free_shipping", "buy_x_get_y"]
    discount_value: float = Field(ge=0)
    code: Optional[str] = None
    start_date: datetime
    end_date: datetime
    minimum_order_value: Optional[float] = None
    maximum_discount: Optional[float] = None
    eligible_products: List[str] = []  # List of product IDs
    eligible_categories: List[ProductCategory] = []
    usage_limit: Optional[int] = None
    is_active: bool = True
    
    @model_validator(mode='after')
    def validate_promotion(self):
        """Validate promotion data consistency."""
        # Check date range
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        
        # Check if promotion is current
        now = datetime.now()
        if now < self.start_date or now > self.end_date:
            self.is_active = False
        
        # Validate discount value based on type
        if self.discount_type == "percentage" and (self.discount_value <= 0 or self.discount_value > 100):
            raise ValueError("Percentage discount must be between 0 and 100")
        
        # Validate code format if provided
        if self.code and not re.match(r'^[A-Z0-9_-]{4,20}$', self.code):
            raise ValueError("Promotion code must be 4-20 uppercase alphanumeric characters, underscores, or hyphens")
        
        return self


class EcommerceValidator:
    """Validator for e-commerce entities and operations."""
    
    @staticmethod
    def validate_product_pricing(product: Product) -> List[str]:
        """
        Validate product pricing rules.
        
        Args:
            product: The product to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for reasonable price range based on category
        if product.category == ProductCategory.ELECTRONICS and product.price.amount < 5:
            errors.append(f"Electronics product '{product.name}' has suspiciously low price: {product.price.amount}")
        
        # Check for high-value items that might need additional verification
        if product.price.amount > 1000:
            errors.append(f"High-value product '{product.name}' may need additional verification")
        
        # Check for reasonable sale price discount
        if product.sale_price:
            discount_percentage = (product.price.amount - product.sale_price.amount) / product.price.amount * 100
            if discount_percentage > 80:
                errors.append(f"Product '{product.name}' has unusually high discount: {discount_percentage:.1f}%")
        
        return errors
    
    @staticmethod
    def validate_inventory_levels(product: Product) -> List[str]:
        """
        Validate product inventory levels.
        
        Args:
            product: The product to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for low inventory
        if 0 < product.inventory_count < 5:
            errors.append(f"Product '{product.name}' has low inventory: {product.inventory_count}")
        
        # Check for inventory status consistency
        if product.inventory_count == 0 and product.status != ProductStatus.OUT_OF_STOCK:
            errors.append(f"Product '{product.name}' has zero inventory but status is {product.status.value}")
        
        if product.inventory_count > 0 and product.status == ProductStatus.OUT_OF_STOCK:
            errors.append(f"Product '{product.name}' has inventory but status is out_of_stock")
        
        return errors
    
    @staticmethod
    def validate_order_for_fulfillment(order: Order, inventory: Dict[str, int]) -> List[str]:
        """
        Validate that an order can be fulfilled based on current inventory.
        
        Args:
            order: The order to validate
            inventory: Dictionary mapping product IDs to inventory counts
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check inventory for each item
        for item in order.items:
            if item.product_id not in inventory:
                errors.append(f"Product {item.product_id} ({item.product_name}) not found in inventory")
                continue
                
            available = inventory.get(item.product_id, 0)
            if item.quantity > available:
                errors.append(f"Insufficient inventory for {item.product_name}: requested {item.quantity}, available {available}")
        
        # Check shipping method validity
        has_physical_items = any(not item.product_id.startswith('DIGITAL-') for item in order.items)
        has_digital_items = any(item.product_id.startswith('DIGITAL-') for item in order.items)
        
        if not has_physical_items and order.shipping_method != ShippingMethod.DIGITAL_DELIVERY:
            errors.append("Order with only digital items should use digital delivery")
        
        if has_physical_items and order.shipping_method == ShippingMethod.DIGITAL_DELIVERY:
            errors.append("Order with physical items cannot use digital delivery")
        
        return errors
    
    @staticmethod
    def validate_promotion_applicability(promotion: Promotion, order: Order) -> List[str]:
        """
        Validate that a promotion can be applied to an order.
        
        Args:
            promotion: The promotion to validate
            order: The order to check against
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if promotion is active
        now = datetime.now()
        if now < promotion.start_date or now > promotion.end_date:
            errors.append(f"Promotion '{promotion.name}' is not active at this time")
            return errors
        
        # Check minimum order value
        if promotion.minimum_order_value and order.subtotal.amount < promotion.minimum_order_value:
            errors.append(f"Order subtotal ({order.subtotal.amount}) is below minimum required for promotion ({promotion.minimum_order_value})")
        
        # Check for eligible products
        if promotion.eligible_products:
            order_product_ids = [item.product_id for item in order.items]
            if not any(pid in promotion.eligible_products for pid in order_product_ids):
                errors.append("Order does not contain any eligible products for this promotion")
        
        # Check for eligible categories
        if promotion.eligible_categories:
            # This would require product category information which we don't have in OrderItem
            # In a real system, you would need to look up the products or include category in OrderItem
            pass
        
        return errors


# Example usage
if __name__ == "__main__":
    # Create a product
    try:
        product = Product(
            name="Smartphone X",
            description="Latest smartphone with amazing features",
            sku="PHONE-X-123",
            price=Price(amount=799.99, currency=Currency.USD),
            sale_price=Price(amount=749.99, currency=Currency.USD),
            category=ProductCategory.ELECTRONICS,
            brand="TechBrand",
            tags=["smartphone", "tech", "5G"],
            images=["image1.jpg", "image2.jpg"],
            inventory_count=100,
            weight=0.2,
            dimensions={"length": 15, "width": 7, "height": 0.8}
        )
        print("Valid product created:", product.name)
        
        # Validate product
        validator = EcommerceValidator()
        pricing_errors = validator.validate_product_pricing(product)
        inventory_errors = validator.validate_inventory_levels(product)
        
        if pricing_errors:
            print("Pricing validation issues:")
            for error in pricing_errors:
                print(f"  - {error}")
        else:
            print("Product pricing is valid")
            
        if inventory_errors:
            print("Inventory validation issues:")
            for error in inventory_errors:
                print(f"  - {error}")
        else:
            print("Product inventory is valid")
            
    except ValueError as e:
        print("Validation error:", e)
