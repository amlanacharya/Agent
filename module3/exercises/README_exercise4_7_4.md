# Exercise 4.7.4: E-commerce Domain Validation

## Overview

This exercise implements validation patterns specific to e-commerce systems, focusing on ensuring data integrity and business rule compliance across the entire purchase lifecycle. The validation system helps maintain high-quality e-commerce operations by enforcing domain-specific rules for products, customers, orders, and promotions.

## Key Features

### Core Models

- `Product`: Model for products with validation for pricing, inventory, and attributes
- `Customer`: Model for customer profiles with validation for contact information and addresses
- `Order`: Model for orders with validation for items, totals, and shipping
- `Promotion`: Model for promotions and discounts with validation for eligibility and applicability

### Support Classes

- `Address`: Model for physical addresses with country-specific postal code validation
- `Price`: Model for prices with currency
- `OrderItem`: Model for items in an order with quantity and pricing validation
- `EcommerceValidator`: Validator for e-commerce entities and operations

### Validation Patterns

1. **Product Validation**
   - Price consistency (sale price < regular price)
   - Digital vs. physical product attributes
   - Inventory status consistency
   - Reasonable pricing checks

2. **Customer Validation**
   - Email format validation
   - Address validation with country-specific rules
   - Default address requirements

3. **Order Validation**
   - Order total calculation verification
   - Item subtotal calculation verification
   - Shipping method consistency with item types
   - Inventory availability for fulfillment

4. **Promotion Validation**
   - Date range validation
   - Discount value validation
   - Promotion code format validation
   - Order eligibility validation

## Usage Example

```python
from exercise4_7_4_ecommerce_validator import (
    Product, Price, Currency, ProductCategory, EcommerceValidator
)

# Create a product
product = Product(
    name="Smartphone X",
    description="Latest smartphone with amazing features",
    sku="PHONE-X-123",
    price=Price(amount=799.99, currency=Currency.USD),
    sale_price=Price(amount=749.99, currency=Currency.USD),
    category=ProductCategory.ELECTRONICS,
    inventory_count=100
)

# Validate product
validator = EcommerceValidator()
pricing_errors = validator.validate_product_pricing(product)
inventory_errors = validator.validate_inventory_levels(product)

# Check for validation issues
if pricing_errors:
    print("Pricing validation issues:")
    for error in pricing_errors:
        print(f"  - {error}")
else:
    print("Product pricing is valid")
```

## Running the Demo

To run the full demo:

```bash
python -m module3.exercises.demo_exercise4_7_4_ecommerce_validator
```

To run a specific part of the demo:

```bash
python -m module3.exercises.demo_exercise4_7_4_ecommerce_validator product
python -m module3.exercises.demo_exercise4_7_4_ecommerce_validator customer
python -m module3.exercises.demo_exercise4_7_4_ecommerce_validator order
python -m module3.exercises.demo_exercise4_7_4_ecommerce_validator promotion
```

## Running the Tests

To run the tests:

```bash
python -m module3.exercises.test_exercise4_7_4_ecommerce_validator
```

## Key Concepts

1. **Domain-Specific Validation**: E-commerce systems require specialized validation rules that reflect business requirements and industry standards.

2. **Cross-Entity Validation**: Many validation rules in e-commerce span multiple entities (e.g., checking if an order can be fulfilled based on inventory).

3. **Calculation Verification**: E-commerce systems must ensure mathematical consistency in pricing, totals, taxes, and discounts.

4. **Business Rule Enforcement**: Validation helps enforce business rules like minimum order values, promotion eligibility, and shipping restrictions.

5. **Data Integrity**: Proper validation ensures that e-commerce data remains consistent and reliable throughout the purchase lifecycle.

## Extension Ideas

Here are some ways you can extend this exercise:

1. **Implement Tax Calculation Validation**: Add validation for tax calculations based on product categories and shipping locations.

2. **Add Inventory Management**: Implement validation for inventory adjustments, reservations, and backorders.

3. **Create User Role-Based Validation**: Add validation that restricts certain operations based on user roles (customer, admin, etc.).

4. **Implement Payment Processing Validation**: Add validation for payment methods, authorization, and fraud detection.

5. **Add Product Bundle Validation**: Implement validation for product bundles, kits, and configurable products.

6. **Create Return/Refund Validation**: Add validation for return eligibility, refund calculations, and restocking fees.

## Integration with Agent Systems

This validator can be integrated with agent systems to:

1. **Validate Customer Requests**: Ensure that customer requests for orders, returns, or account changes are valid.

2. **Validate Agent Responses**: Ensure that agent responses to customer inquiries about products, orders, or promotions are accurate.

3. **Validate Automated Actions**: Ensure that automated actions like order processing, inventory updates, or promotion applications follow business rules.

4. **Provide Helpful Error Messages**: Generate user-friendly error messages that help customers understand and resolve issues with their orders or account information.
