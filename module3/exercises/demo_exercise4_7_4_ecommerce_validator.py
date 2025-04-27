"""
Demo Script for Exercise 4.7.4: E-commerce Domain Validation
----------------------------------------------------------
This script demonstrates the validation patterns for e-commerce systems implemented in
exercise4_7_4_ecommerce_validator.py.
"""

from datetime import datetime, timedelta
import sys
# Adjust the import path based on how you're running the script
try:
    # When running from the module3/exercises directory
    from exercise4_7_4_ecommerce_validator import (
        Product, Customer, Order, OrderItem, Promotion, Address,
        Price, EcommerceValidator, ProductCategory, ProductStatus,
        Currency, ShippingMethod, PaymentMethod, OrderStatus
    )
except ImportError:
    # When running from the project root
    from module3.exercises.exercise4_7_4_ecommerce_validator import (
        Product, Customer, Order, OrderItem, Promotion, Address,
        Price, EcommerceValidator, ProductCategory, ProductStatus,
        Currency, ShippingMethod, PaymentMethod, OrderStatus
    )


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subheader."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)


def print_validation_errors(errors):
    """Print validation errors."""
    if not errors:
        print("✅ No validation errors")
    else:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")


def demo_product_validation():
    """Demonstrate product validation."""
    print_header("Product Validation Demo")
    validator = EcommerceValidator()

    print_subheader("Valid Product")
    
    try:
        valid_product = Product(
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
        
        print(f"Product: {valid_product.name}")
        print(f"  SKU: {valid_product.sku}")
        print(f"  Price: {valid_product.price.amount} {valid_product.price.currency.value}")
        print(f"  Sale Price: {valid_product.sale_price.amount} {valid_product.sale_price.currency.value}")
        print(f"  Category: {valid_product.category.value}")
        print(f"  Inventory: {valid_product.inventory_count}")
        print(f"  Status: {valid_product.status.value}")
        
        # Validate product
        pricing_errors = validator.validate_product_pricing(valid_product)
        inventory_errors = validator.validate_inventory_levels(valid_product)
        
        print("\nPricing validation:")
        print_validation_errors(pricing_errors)
        
        print("\nInventory validation:")
        print_validation_errors(inventory_errors)
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Invalid Product (Sale Price Higher Than Regular Price)")
    
    try:
        invalid_product = Product(
            name="Broken Deal Gadget",
            description="A gadget with pricing issues",
            sku="BROKEN-DEAL-123",
            price=Price(amount=99.99, currency=Currency.USD),
            sale_price=Price(amount=129.99, currency=Currency.USD),  # Higher than regular price
            category=ProductCategory.ELECTRONICS,
            inventory_count=50
        )
        
        print(f"Product: {invalid_product.name}")
        print(f"  Price: {invalid_product.price.amount} {invalid_product.price.currency.value}")
        print(f"  Sale Price: {invalid_product.sale_price.amount} {invalid_product.sale_price.currency.value}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Digital Product with Physical Attributes")
    
    try:
        digital_product = Product(
            name="E-book: Python Programming",
            description="Learn Python programming from scratch",
            sku="EBOOK-PYTHON-101",
            price=Price(amount=29.99, currency=Currency.USD),
            category=ProductCategory.BOOKS,
            inventory_count=999,
            is_digital=True,
            weight=0.5,  # Should not have weight
            dimensions={"length": 20, "width": 15, "height": 2}  # Should not have dimensions
        )
        
        print(f"Product: {digital_product.name}")
        print(f"  Digital: {digital_product.is_digital}")
        print(f"  Weight: {digital_product.weight}")
        print(f"  Dimensions: {digital_product.dimensions}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Product with Inventory Issues")
    
    try:
        out_of_stock_product = Product(
            name="Vintage Vinyl Record",
            description="Rare collector's edition vinyl",
            sku="VINYL-RARE-001",
            price=Price(amount=199.99, currency=Currency.USD),
            category=ProductCategory.OTHER,
            inventory_count=0,  # Zero inventory
            status=ProductStatus.ACTIVE  # But status is active
        )
        
        print(f"Product: {out_of_stock_product.name}")
        print(f"  Inventory: {out_of_stock_product.inventory_count}")
        print(f"  Status: {out_of_stock_product.status.value}")
        
        # Validate inventory
        inventory_errors = validator.validate_inventory_levels(out_of_stock_product)
        print("\nInventory validation:")
        print_validation_errors(inventory_errors)
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")


def demo_customer_validation():
    """Demonstrate customer validation."""
    print_header("Customer Validation Demo")

    print_subheader("Valid Customer")
    
    try:
        valid_address = Address(
            street_line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="90210",
            country="US",
            is_default=True
        )
        
        valid_customer = Customer(
            email="john.doe@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1-555-123-4567",
            addresses=[valid_address],
            default_payment_method=PaymentMethod.CREDIT_CARD,
            marketing_preferences={"email": True, "sms": False}
        )
        
        print(f"Customer: {valid_customer.first_name} {valid_customer.last_name}")
        print(f"  Email: {valid_customer.email}")
        print(f"  Phone: {valid_customer.phone}")
        print(f"  Addresses: {len(valid_customer.addresses)}")
        print(f"  Default Payment: {valid_customer.default_payment_method.value}")
        print(f"  Marketing Preferences: {valid_customer.marketing_preferences}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Invalid Customer (Bad Email)")
    
    try:
        invalid_customer = Customer(
            email="not-an-email",  # Invalid email format
            first_name="Jane",
            last_name="Smith",
            addresses=[
                Address(
                    street_line1="456 Oak Ave",
                    city="Somewhere",
                    state="NY",
                    postal_code="10001",
                    country="US",
                    is_default=True
                )
            ]
        )
        
        print(f"Customer: {invalid_customer.first_name} {invalid_customer.last_name}")
        print(f"  Email: {invalid_customer.email}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Customer with No Default Address")
    
    try:
        address1 = Address(
            street_line1="789 Pine St",
            city="Elsewhere",
            state="TX",
            postal_code="75001",
            country="US",
            is_default=False
        )
        
        address2 = Address(
            street_line1="101 Cedar Rd",
            city="Nowhere",
            state="WA",
            postal_code="98001",
            country="US",
            is_default=False
        )
        
        customer_no_default = Customer(
            email="bob.jones@example.com",
            first_name="Bob",
            last_name="Jones",
            addresses=[address1, address2]  # No default address
        )
        
        print(f"Customer: {customer_no_default.first_name} {customer_no_default.last_name}")
        print(f"  Email: {customer_no_default.email}")
        print(f"  Addresses: {len(customer_no_default.addresses)}")
        print(f"  Has Default: {any(a.is_default for a in customer_no_default.addresses)}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")


def demo_order_validation():
    """Demonstrate order validation."""
    print_header("Order Validation Demo")
    validator = EcommerceValidator()

    print_subheader("Valid Order")
    
    try:
        # Create order items
        item1 = OrderItem(
            product_id="PROD-123",
            product_name="Smartphone X",
            quantity=1,
            unit_price=Price(amount=799.99, currency=Currency.USD),
            subtotal=Price(amount=799.99, currency=Currency.USD)
        )
        
        item2 = OrderItem(
            product_id="PROD-456",
            product_name="Phone Case",
            quantity=2,
            unit_price=Price(amount=19.99, currency=Currency.USD),
            subtotal=Price(amount=39.98, currency=Currency.USD)
        )
        
        # Create shipping address
        shipping_address = Address(
            street_line1="123 Main St",
            city="Anytown",
            state="CA",
            postal_code="90210",
            country="US"
        )
        
        # Create order
        valid_order = Order(
            customer_id="CUST-123",
            items=[item1, item2],
            subtotal=Price(amount=839.97, currency=Currency.USD),
            tax=Price(amount=67.20, currency=Currency.USD),
            shipping_cost=Price(amount=9.99, currency=Currency.USD),
            total=Price(amount=917.16, currency=Currency.USD),
            shipping_address=shipping_address,
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        print(f"Order ID: {valid_order.id}")
        print(f"  Customer ID: {valid_order.customer_id}")
        print(f"  Items: {len(valid_order.items)}")
        print(f"  Subtotal: ${valid_order.subtotal.amount:.2f}")
        print(f"  Tax: ${valid_order.tax.amount:.2f}")
        print(f"  Shipping: ${valid_order.shipping_cost.amount:.2f}")
        print(f"  Total: ${valid_order.total.amount:.2f}")
        print(f"  Status: {valid_order.status.value}")
        
        # Check inventory for fulfillment
        inventory = {
            "PROD-123": 50,  # 50 smartphones in stock
            "PROD-456": 200  # 200 phone cases in stock
        }
        
        fulfillment_errors = validator.validate_order_for_fulfillment(valid_order, inventory)
        print("\nFulfillment validation:")
        print_validation_errors(fulfillment_errors)
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Invalid Order (Calculation Error)")
    
    try:
        # Create order with calculation error
        item = OrderItem(
            product_id="PROD-123",
            product_name="Smartphone X",
            quantity=1,
            unit_price=Price(amount=799.99, currency=Currency.USD),
            subtotal=Price(amount=799.99, currency=Currency.USD)
        )
        
        invalid_order = Order(
            customer_id="CUST-123",
            items=[item],
            subtotal=Price(amount=799.99, currency=Currency.USD),
            tax=Price(amount=64.00, currency=Currency.USD),
            shipping_cost=Price(amount=9.99, currency=Currency.USD),
            total=Price(amount=899.99, currency=Currency.USD),  # Should be 873.98
            shipping_address=Address(
                street_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="90210",
                country="US"
            ),
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        print(f"Order ID: {invalid_order.id}")
        print(f"  Subtotal: ${invalid_order.subtotal.amount:.2f}")
        print(f"  Tax: ${invalid_order.tax.amount:.2f}")
        print(f"  Shipping: ${invalid_order.shipping_cost.amount:.2f}")
        print(f"  Total: ${invalid_order.total.amount:.2f}")
        print(f"  Expected Total: ${invalid_order.subtotal.amount + invalid_order.tax.amount + invalid_order.shipping_cost.amount:.2f}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Digital Product with Physical Shipping")
    
    try:
        # Create digital product order with physical shipping
        digital_item = OrderItem(
            product_id="DIGITAL-123",
            product_name="E-book: Python Programming",
            quantity=1,
            unit_price=Price(amount=29.99, currency=Currency.USD),
            subtotal=Price(amount=29.99, currency=Currency.USD)
        )
        
        digital_order = Order(
            customer_id="CUST-123",
            items=[digital_item],
            subtotal=Price(amount=29.99, currency=Currency.USD),
            tax=Price(amount=2.40, currency=Currency.USD),
            shipping_cost=Price(amount=4.99, currency=Currency.USD),  # Should be 0 for digital
            total=Price(amount=37.38, currency=Currency.USD),
            shipping_address=Address(
                street_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="90210",
                country="US"
            ),
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD  # Should be DIGITAL_DELIVERY
        )
        
        print(f"Order ID: {digital_order.id}")
        print(f"  Product: {digital_order.items[0].product_name}")
        print(f"  Shipping Method: {digital_order.shipping_method.value}")
        print(f"  Shipping Cost: ${digital_order.shipping_cost.amount:.2f}")
        
        # Check fulfillment validation
        inventory = {"DIGITAL-123": 999}
        fulfillment_errors = validator.validate_order_for_fulfillment(digital_order, inventory)
        print("\nFulfillment validation:")
        print_validation_errors(fulfillment_errors)
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")


def demo_promotion_validation():
    """Demonstrate promotion validation."""
    print_header("Promotion Validation Demo")
    validator = EcommerceValidator()

    print_subheader("Valid Promotion")
    
    try:
        valid_promotion = Promotion(
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
        
        print(f"Promotion: {valid_promotion.name}")
        print(f"  Description: {valid_promotion.description}")
        print(f"  Discount: {valid_promotion.discount_type} - {valid_promotion.discount_value}")
        print(f"  Code: {valid_promotion.code}")
        print(f"  Valid Period: {valid_promotion.start_date.strftime('%Y-%m-%d')} to {valid_promotion.end_date.strftime('%Y-%m-%d')}")
        print(f"  Minimum Order: ${valid_promotion.minimum_order_value:.2f}")
        print(f"  Eligible Categories: {[cat.value for cat in valid_promotion.eligible_categories]}")
        print(f"  Active: {valid_promotion.is_active}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Invalid Promotion (Invalid Date Range)")
    
    try:
        invalid_promotion = Promotion(
            name="Backwards Promo",
            description="This promotion has an invalid date range",
            discount_type="percentage",
            discount_value=10.0,
            start_date=datetime.now() + timedelta(days=10),  # Future start
            end_date=datetime.now()  # End date before start date
        )
        
        print(f"Promotion: {invalid_promotion.name}")
        print(f"  Start Date: {invalid_promotion.start_date.strftime('%Y-%m-%d')}")
        print(f"  End Date: {invalid_promotion.end_date.strftime('%Y-%m-%d')}")
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Promotion Applicability Check")
    
    try:
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
            shipping_address=Address(
                street_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="90210",
                country="US"
            ),
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
            shipping_address=Address(
                street_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="90210",
                country="US"
            ),
            payment_method=PaymentMethod.CREDIT_CARD,
            shipping_method=ShippingMethod.STANDARD
        )
        
        print(f"Promotion: {promotion.name}")
        print(f"  Code: {promotion.code}")
        print(f"  Minimum Order: ${promotion.minimum_order_value:.2f}")
        print(f"  Eligible Products: {promotion.eligible_products}")
        
        print("\nOrder 1:")
        print(f"  Subtotal: ${valid_order.subtotal.amount:.2f}")
        print(f"  Products: {[item.product_id for item in valid_order.items]}")
        
        applicability_errors = validator.validate_promotion_applicability(promotion, valid_order)
        print("\nPromotion applicability for Order 1:")
        print_validation_errors(applicability_errors)
        
        print("\nOrder 2:")
        print(f"  Subtotal: ${invalid_order.subtotal.amount:.2f}")
        print(f"  Products: {[item.product_id for item in invalid_order.items]}")
        
        applicability_errors = validator.validate_promotion_applicability(promotion, invalid_order)
        print("\nPromotion applicability for Order 2:")
        print_validation_errors(applicability_errors)
        
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")


def main():
    """Main function to run the demo."""
    print_header("E-commerce Domain Validation Demo")

    print("""
This demo showcases validation patterns specific to e-commerce systems, including:

1. Product validation (pricing, inventory, categories)
2. Customer validation (profiles, addresses, payment methods)
3. Order validation (items, totals, shipping)
4. Promotion validation (discounts, coupons, eligibility)

These validation patterns help ensure that e-commerce systems maintain data integrity
and business rule compliance across the entire purchase lifecycle.
""")

    # Check if a specific demo was requested
    if len(sys.argv) > 1:
        demo = sys.argv[1].lower()
        if demo == "product":
            demo_product_validation()
        elif demo == "customer":
            demo_customer_validation()
        elif demo == "order":
            demo_order_validation()
        elif demo == "promotion":
            demo_promotion_validation()
        else:
            print(f"Unknown demo: {demo}")
            print("Available demos: product, customer, order, promotion")
    else:
        # Run all demos
        demo_product_validation()
        demo_customer_validation()
        demo_order_validation()
        demo_promotion_validation()


if __name__ == "__main__":
    main()
