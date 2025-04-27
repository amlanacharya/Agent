"""
Test Script for Exercise 4.7.4: E-commerce Domain Validation
----------------------------------------------------------
This script tests the validation patterns for e-commerce systems implemented in
exercise4_7_4_ecommerce_validator.py.
"""

import unittest
from datetime import datetime, timedelta

from module3.exercises.exercise4_7_4_ecommerce_validator import (
    Product, Customer, Order, OrderItem, Promotion, Address,
    Price, EcommerceValidator, ProductCategory, ProductStatus,
    Currency, ShippingMethod, PaymentMethod, OrderStatus
)


class TestProductValidation(unittest.TestCase):
    """Test cases for product validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = EcommerceValidator()
        
        # Create a valid product for testing
        self.valid_product = Product(
            name="Test Product",
            description="A product for testing",
            sku="TEST-123",
            price=Price(amount=99.99, currency=Currency.USD),
            category=ProductCategory.ELECTRONICS,
            inventory_count=100
        )

    def test_valid_product(self):
        """Test a valid product."""
        # This should not raise any exceptions
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
        
        self.assertEqual(product.name, "Smartphone X")
        self.assertEqual(product.price.amount, 799.99)
        self.assertEqual(product.sale_price.amount, 749.99)
        self.assertEqual(product.category, ProductCategory.ELECTRONICS)

    def test_invalid_sale_price(self):
        """Test a product with invalid sale price."""
        # Sale price higher than regular price should raise ValueError
        with self.assertRaises(ValueError):
            Product(
                name="Bad Deal",
                description="A product with invalid sale price",
                sku="BAD-DEAL-123",
                price=Price(amount=99.99, currency=Currency.USD),
                sale_price=Price(amount=129.99, currency=Currency.USD),  # Higher than regular price
                category=ProductCategory.ELECTRONICS,
                inventory_count=50
            )

    def test_digital_product_with_physical_attributes(self):
        """Test a digital product with physical attributes."""
        # Digital product with weight/dimensions should raise ValueError
        with self.assertRaises(ValueError):
            Product(
                name="E-book",
                description="Digital book",
                sku="EBOOK-123",
                price=Price(amount=19.99, currency=Currency.USD),
                category=ProductCategory.BOOKS,
                inventory_count=999,
                is_digital=True,
                weight=0.5  # Should not have weight
            )

    def test_inventory_validation(self):
        """Test inventory validation."""
        # Create a product with zero inventory but active status
        product = Product(
            name="Out of Stock Item",
            description="This should be marked as out of stock",
            sku="OUT-OF-STOCK-123",
            price=Price(amount=49.99, currency=Currency.USD),
            category=ProductCategory.HOME,
            inventory_count=0,
            status=ProductStatus.ACTIVE  # Inconsistent with inventory
        )
        
        # Validate inventory
        errors = self.validator.validate_inventory_levels(product)
        
        # Should have at least one error about inventory status inconsistency
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("zero inventory but status is active" in error.lower() for error in errors))

    def test_pricing_validation(self):
        """Test pricing validation."""
        # Create a product with suspiciously high discount
        product = Product(
            name="Extreme Discount Item",
            description="Item with very high discount",
            sku="DISCOUNT-123",
            price=Price(amount=1000.00, currency=Currency.USD),
            sale_price=Price(amount=100.00, currency=Currency.USD),  # 90% discount
            category=ProductCategory.ELECTRONICS,
            inventory_count=10
        )
        
        # Validate pricing
        errors = self.validator.validate_product_pricing(product)
        
        # Should have at least one error about high discount
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("high discount" in error.lower() for error in errors))


class TestCustomerValidation(unittest.TestCase):
    """Test cases for customer validation."""

    def test_valid_customer(self):
        """Test a valid customer."""
        # This should not raise any exceptions
        customer = Customer(
            email="john.doe@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1-555-123-4567",
            addresses=[
                Address(
                    street_line1="123 Main St",
                    city="Anytown",
                    state="CA",
                    postal_code="90210",
                    country="US",
                    is_default=True
                )
            ],
            default_payment_method=PaymentMethod.CREDIT_CARD
        )
        
        self.assertEqual(customer.email, "john.doe@example.com")
        self.assertEqual(customer.first_name, "John")
        self.assertEqual(len(customer.addresses), 1)
        self.assertTrue(customer.addresses[0].is_default)

    def test_invalid_email(self):
        """Test a customer with invalid email."""
        # Invalid email should raise ValueError
        with self.assertRaises(ValueError):
            Customer(
                email="not-an-email",  # Invalid email format
                first_name="Jane",
                last_name="Smith"
            )

    def test_postal_code_validation(self):
        """Test postal code validation based on country."""
        # Invalid US postal code should raise ValueError
        with self.assertRaises(ValueError):
            Address(
                street_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="ABC123",  # Invalid US postal code
                country="US"
            )
        
        # Valid US postal code should not raise exception
        address = Address(
            street_line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="90210",
            country="US"
        )
        self.assertEqual(address.postal_code, "90210")


class TestOrderValidation(unittest.TestCase):
    """Test cases for order validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = EcommerceValidator()
        
        # Create a valid shipping address
        self.shipping_address = Address(
            street_line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="90210",
            country="US"
        )
        
        # Create valid order items
        self.item1 = OrderItem(
            product_id="PROD-123",
            product_name="Smartphone X",
            quantity=1,
            unit_price=Price(amount=799.99, currency=Currency.USD),
            subtotal=Price(amount=799.99, currency=Currency.USD)
        )
        
        self.item2 = OrderItem(
            product_id="PROD-456",
            product_name="Phone Case",
            quantity=2,
            unit_price=Price(amount=19.99, currency=Currency.USD),
            subtotal=Price(amount=39.98, currency=Currency.USD)
        )

    def test_valid_order(self):
        """Test a valid order."""
        # This should not raise any exceptions
        order = Order(
            customer_id="CUST-123",
            items=[self.item1, self.item2],
            subtotal=Price(amount=839.97, currency=Currency.USD),
            tax=Price(amount=67.20, currency=Currency.USD),
            shipping_cost=Price(amount=9.99, currency=Currency.USD),
            total=Price(amount=917.16, currency=Currency.USD),
            shipping_address=self.shipping_address,
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        self.assertEqual(order.customer_id, "CUST-123")
        self.assertEqual(len(order.items), 2)
        self.assertEqual(order.subtotal.amount, 839.97)
        self.assertEqual(order.total.amount, 917.16)

    def test_order_total_calculation(self):
        """Test order total calculation validation."""
        # Incorrect total should raise ValueError
        with self.assertRaises(ValueError):
            Order(
                customer_id="CUST-123",
                items=[self.item1],
                subtotal=Price(amount=799.99, currency=Currency.USD),
                tax=Price(amount=64.00, currency=Currency.USD),
                shipping_cost=Price(amount=9.99, currency=Currency.USD),
                total=Price(amount=899.99, currency=Currency.USD),  # Should be 873.98
                shipping_address=self.shipping_address,
                payment_method=PaymentMethod.CREDIT_CARD,
                shipping_method=ShippingMethod.STANDARD
            )

    def test_digital_delivery_validation(self):
        """Test digital delivery validation."""
        # Create a digital item
        digital_item = OrderItem(
            product_id="DIGITAL-123",
            product_name="E-book",
            quantity=1,
            unit_price=Price(amount=29.99, currency=Currency.USD),
            subtotal=Price(amount=29.99, currency=Currency.USD)
        )
        
        # Digital item with shipping cost should raise ValueError
        with self.assertRaises(ValueError):
            Order(
                customer_id="CUST-123",
                items=[digital_item],
                subtotal=Price(amount=29.99, currency=Currency.USD),
                tax=Price(amount=2.40, currency=Currency.USD),
                shipping_cost=Price(amount=4.99, currency=Currency.USD),  # Should be 0 for digital
                total=Price(amount=37.38, currency=Currency.USD),
                shipping_address=self.shipping_address,
                payment_method=PaymentMethod.CREDIT_CARD,
                shipping_method=ShippingMethod.DIGITAL_DELIVERY
            )

    def test_order_fulfillment_validation(self):
        """Test order fulfillment validation."""
        # Create a valid order
        order = Order(
            customer_id="CUST-123",
            items=[self.item1, self.item2],
            subtotal=Price(amount=839.97, currency=Currency.USD),
            tax=Price(amount=67.20, currency=Currency.USD),
            shipping_cost=Price(amount=9.99, currency=Currency.USD),
            total=Price(amount=917.16, currency=Currency.USD),
            shipping_address=self.shipping_address,
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        # Test with sufficient inventory
        inventory = {
            "PROD-123": 50,  # 50 smartphones in stock
            "PROD-456": 200  # 200 phone cases in stock
        }
        
        errors = self.validator.validate_order_for_fulfillment(order, inventory)
        self.assertEqual(len(errors), 0, "Should have no errors with sufficient inventory")
        
        # Test with insufficient inventory
        low_inventory = {
            "PROD-123": 0,  # Out of stock
            "PROD-456": 1   # Only 1 in stock, but order has 2
        }
        
        errors = self.validator.validate_order_for_fulfillment(order, low_inventory)
        self.assertGreater(len(errors), 0, "Should have errors with insufficient inventory")
        self.assertEqual(len(errors), 2, "Should have 2 errors (one for each product)")


class TestPromotionValidation(unittest.TestCase):
    """Test cases for promotion validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = EcommerceValidator()

    def test_valid_promotion(self):
        """Test a valid promotion."""
        # This should not raise any exceptions
        promotion = Promotion(
            name="Summer Sale",
            description="20% off all summer items",
            discount_type="percentage",
            discount_value=20.0,
            code="SUMMER20",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now() + timedelta(days=20),
            minimum_order_value=50.0,
            eligible_categories=[ProductCategory.CLOTHING, ProductCategory.SPORTS]
        )
        
        self.assertEqual(promotion.name, "Summer Sale")
        self.assertEqual(promotion.discount_type, "percentage")
        self.assertEqual(promotion.discount_value, 20.0)
        self.assertTrue(promotion.is_active)

    def test_invalid_date_range(self):
        """Test a promotion with invalid date range."""
        # End date before start date should raise ValueError
        with self.assertRaises(ValueError):
            Promotion(
                name="Invalid Promo",
                description="This promotion has an invalid date range",
                discount_type="percentage",
                discount_value=10.0,
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)  # End date before start date
            )

    def test_invalid_percentage_discount(self):
        """Test a promotion with invalid percentage discount."""
        # Percentage discount > 100% should raise ValueError
        with self.assertRaises(ValueError):
            Promotion(
                name="Invalid Discount",
                description="This promotion has an invalid discount percentage",
                discount_type="percentage",
                discount_value=120.0,  # More than 100%
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=30)
            )

    def test_promotion_applicability(self):
        """Test promotion applicability validation."""
        # Create a promotion
        promotion = Promotion(
            name="Electronics Discount",
            description="$50 off electronics orders over $500",
            discount_type="fixed_amount",
            discount_value=50.0,
            code="TECH50",
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now() + timedelta(days=25),
            minimum_order_value=500.0,
            eligible_products=["PROD-123", "PROD-456"]
        )
        
        # Create a valid shipping address
        shipping_address = Address(
            street_line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="90210",
            country="US"
        )
        
        # Create an order that meets requirements
        valid_order = Order(
            customer_id="CUST-123",
            items=[
                OrderItem(
                    product_id="PROD-123",
                    product_name="Smartphone X",
                    quantity=1,
                    unit_price=Price(amount=799.99, currency=Currency.USD),
                    subtotal=Price(amount=799.99, currency=Currency.USD)
                )
            ],
            subtotal=Price(amount=799.99, currency=Currency.USD),
            tax=Price(amount=64.00, currency=Currency.USD),
            shipping_cost=Price(amount=9.99, currency=Currency.USD),
            total=Price(amount=873.98, currency=Currency.USD),
            shipping_address=shipping_address,
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        # Create an order that doesn't meet requirements
        invalid_order = Order(
            customer_id="CUST-123",
            items=[
                OrderItem(
                    product_id="PROD-789",  # Not in eligible products
                    product_name="Headphones",
                    quantity=1,
                    unit_price=Price(amount=99.99, currency=Currency.USD),
                    subtotal=Price(amount=99.99, currency=Currency.USD)
                )
            ],
            subtotal=Price(amount=99.99, currency=Currency.USD),
            tax=Price(amount=8.00, currency=Currency.USD),
            shipping_cost=Price(amount=5.99, currency=Currency.USD),
            total=Price(amount=113.98, currency=Currency.USD),
            shipping_address=shipping_address,
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        # Valid order should have no applicability errors
        errors = self.validator.validate_promotion_applicability(promotion, valid_order)
        self.assertEqual(len(errors), 0, "Valid order should have no promotion applicability errors")
        
        # Invalid order should have applicability errors
        errors = self.validator.validate_promotion_applicability(promotion, invalid_order)
        self.assertGreater(len(errors), 0, "Invalid order should have promotion applicability errors")
        self.assertTrue(any("does not contain any eligible products" in error.lower() for error in errors))


if __name__ == "__main__":
    unittest.main()
