"""
Tests for Pydantic Basics
------------------------
This module contains tests for the pydantic_basics module.
"""

import unittest
from datetime import datetime, timedelta
from pydantic import ValidationError

from .pydantic_basics import (
    User, 
    Product, 
    UserWithConfig, 
    AdvancedUser, 
    SignupForm, 
    BaseItem, 
    ProductItem, 
    ServiceItem, 
    TaskInput, 
    process_task_creation
)


class TestPydanticBasics(unittest.TestCase):
    """Test cases for pydantic_basics module."""
    
    def test_basic_model(self):
        """Test basic User model."""
        # Valid user
        user = User(id=1, name="John Doe", email="john@example.com")
        self.assertEqual(user.id, 1)
        self.assertEqual(user.name, "John Doe")
        self.assertEqual(user.email, "john@example.com")
        self.assertIsNone(user.age)
        self.assertEqual(user.tags, [])
        
        # Test with optional fields
        user = User(id=2, name="Jane", email="jane@example.com", age=30, tags=["admin"])
        self.assertEqual(user.age, 30)
        self.assertEqual(user.tags, ["admin"])
        
        # Test type coercion
        user = User(id="42", name="Bob", email="bob@example.com")
        self.assertEqual(user.id, 42)
        self.assertIsInstance(user.id, int)
        
        # Test invalid data
        with self.assertRaises(ValidationError):
            User(id="not_an_int", name=123, email="invalid_email")
    
    def test_field_constraints(self):
        """Test field constraints in Product model."""
        # Valid product
        product = Product(id=1, name="Laptop", price=999.99)
        self.assertEqual(product.id, 1)
        self.assertEqual(product.name, "Laptop")
        self.assertEqual(product.price, 999.99)
        
        # Test with optional description
        product = Product(id=2, name="Phone", price=499.99, description="A smartphone")
        self.assertEqual(product.description, "A smartphone")
        
        # Test name too short
        with self.assertRaises(ValidationError):
            Product(id=3, name="PC", price=1299.99)  # Name too short
        
        # Test negative price
        with self.assertRaises(ValidationError):
            Product(id=4, name="Tablet", price=-199.99)  # Negative price
    
    def test_custom_validators(self):
        """Test custom validators in SignupForm model."""
        # Valid signup
        form = SignupForm(
            username="johndoe", 
            password="password123", 
            password_confirm="password123"
        )
        self.assertEqual(form.username, "johndoe")
        
        # Test non-alphanumeric username
        with self.assertRaises(ValidationError):
            SignupForm(
                username="john.doe", 
                password="password123", 
                password_confirm="password123"
            )
        
        # Test password mismatch
        with self.assertRaises(ValidationError):
            SignupForm(
                username="johndoe", 
                password="password123", 
                password_confirm="different"
            )
    
    def test_model_inheritance(self):
        """Test model inheritance with BaseItem."""
        # Create product
        product = ProductItem(id=1, name="Laptop", price=999.99, category="Electronics")
        self.assertEqual(product.id, 1)
        self.assertEqual(product.name, "Laptop")
        self.assertIsInstance(product.created_at, datetime)
        
        # Create service
        service = ServiceItem(
            id=2, 
            name="Consulting", 
            hourly_rate=150.0, 
            description="Expert advice"
        )
        self.assertEqual(service.id, 2)
        self.assertEqual(service.hourly_rate, 150.0)
    
    def test_task_input(self):
        """Test TaskInput model and process_task_creation function."""
        # Valid task
        future_date = datetime.now() + timedelta(days=7)
        valid_task = {
            "title": "Complete project",
            "description": "Finish the Pydantic module",
            "priority": "high",
            "due_date": future_date.isoformat(),
            "tags": ["work", "important"]
        }
        
        result = process_task_creation(valid_task)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task"]["title"], "Complete project")
        self.assertEqual(result["task"]["priority"], "high")
        
        # Invalid task - title too short
        invalid_task = {
            "title": "A",  # Too short
            "priority": "medium"
        }
        
        result = process_task_creation(invalid_task)
        self.assertEqual(result["status"], "error")
        self.assertIn("title", result["message"])
        
        # Invalid task - invalid priority
        invalid_task = {
            "title": "Valid Title",
            "priority": "critical"  # Not in enum
        }
        
        result = process_task_creation(invalid_task)
        self.assertEqual(result["status"], "error")
        self.assertIn("priority", result["message"])
        
        # Invalid task - past due date
        past_date = datetime.now() - timedelta(days=7)
        invalid_task = {
            "title": "Valid Title",
            "due_date": past_date.isoformat()
        }
        
        result = process_task_creation(invalid_task)
        self.assertEqual(result["status"], "error")
        self.assertIn("due_date", result["message"])


if __name__ == "__main__":
    unittest.main()
