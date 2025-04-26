"""
Tests for Exercise 4.3.3: Database Model Generator
------------------------------------------------
This module contains tests for the database model generator implementation.
"""

import unittest
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Union, get_type_hints

from exercise4.3.3_db_model_generator import (
    DBColumn,
    DBTable,
    DBModelGenerator,
    create_mock_db_schema
)


class TestDBModelGenerator(unittest.TestCase):
    """Test cases for the database model generator implementation."""
    
    def test_db_column(self):
        """Test the DBColumn class."""
        column = DBColumn(
            name="username",
            data_type="varchar",
            nullable=False,
            max_length=50,
            description="User's login name"
        )
        self.assertEqual(column.name, "username")
        self.assertEqual(column.data_type, "varchar")
        self.assertFalse(column.nullable)
        self.assertEqual(column.max_length, 50)
        self.assertEqual(column.description, "User's login name")
    
    def test_db_table(self):
        """Test the DBTable class."""
        columns = [
            DBColumn(name="id", data_type="integer", primary_key=True),
            DBColumn(name="name", data_type="varchar", max_length=100)
        ]
        table = DBTable(
            name="products",
            columns=columns,
            description="Product catalog"
        )
        self.assertEqual(table.name, "products")
        self.assertEqual(len(table.columns), 2)
        self.assertEqual(table.description, "Product catalog")
    
    def test_get_python_type(self):
        """Test conversion of database types to Python types."""
        # Integer types
        int_column = DBColumn(name="id", data_type="integer", nullable=False)
        self.assertEqual(DBModelGenerator.get_python_type(int_column), int)
        
        # String types
        str_column = DBColumn(name="name", data_type="varchar(100)", nullable=False)
        self.assertEqual(DBModelGenerator.get_python_type(str_column), str)
        
        # Date/time types
        date_column = DBColumn(name="birth_date", data_type="date", nullable=True)
        self.assertEqual(DBModelGenerator.get_python_type(date_column), Optional[date])
        
        datetime_column = DBColumn(name="created_at", data_type="timestamp", nullable=False)
        self.assertEqual(DBModelGenerator.get_python_type(datetime_column), datetime)
        
        # Boolean types
        bool_column = DBColumn(name="is_active", data_type="boolean", nullable=False)
        self.assertEqual(DBModelGenerator.get_python_type(bool_column), bool)
        
        # JSON types
        json_column = DBColumn(name="metadata", data_type="json", nullable=True)
        self.assertEqual(DBModelGenerator.get_python_type(json_column), Optional[Dict[str, Any]])
        
        # Unknown type
        unknown_column = DBColumn(name="custom", data_type="custom_type", nullable=False)
        self.assertEqual(DBModelGenerator.get_python_type(unknown_column), Any)
    
    def test_get_field_constraints(self):
        """Test generation of field constraints from database columns."""
        # String length constraint
        varchar_column = DBColumn(
            name="username",
            data_type="varchar",
            max_length=50,
            description="Username"
        )
        constraints = DBModelGenerator.get_field_constraints(varchar_column)
        self.assertEqual(constraints["description"], "Username")
        self.assertEqual(constraints["max_length"], 50)
        
        # Primary key
        pk_column = DBColumn(
            name="id",
            data_type="integer",
            primary_key=True,
            description="Primary key"
        )
        constraints = DBModelGenerator.get_field_constraints(pk_column)
        self.assertEqual(constraints["title"], "Primary Key")
        self.assertEqual(constraints["description"], "Primary key")
        
        # Foreign key
        fk_column = DBColumn(
            name="user_id",
            data_type="integer",
            foreign_key="users.id",
            description="User reference"
        )
        constraints = DBModelGenerator.get_field_constraints(fk_column)
        self.assertIn("Foreign Key: users.id", constraints["description"])
    
    def test_create_model_from_table(self):
        """Test creation of a Pydantic model from a database table."""
        # Create a simple table
        table = DBTable(
            name="products",
            description="Product catalog",
            columns=[
                DBColumn(
                    name="id",
                    data_type="integer",
                    nullable=False,
                    primary_key=True
                ),
                DBColumn(
                    name="name",
                    data_type="varchar",
                    max_length=100,
                    nullable=False
                ),
                DBColumn(
                    name="price",
                    data_type="decimal",
                    nullable=False
                ),
                DBColumn(
                    name="description",
                    data_type="text",
                    nullable=True
                ),
                DBColumn(
                    name="created_at",
                    data_type="timestamp",
                    nullable=False,
                    default="CURRENT_TIMESTAMP"
                )
            ]
        )
        
        # Generate model
        model = DBModelGenerator.create_model_from_table(table)
        
        # Check model properties
        self.assertEqual(model.__name__, "Products")
        self.assertEqual(model.__doc__, "Product catalog")
        
        # Check fields
        self.assertIn("id", model.model_fields)
        self.assertIn("name", model.model_fields)
        self.assertIn("price", model.model_fields)
        self.assertIn("description", model.model_fields)
        self.assertIn("created_at", model.model_fields)
        
        # Check field types
        self.assertEqual(model.model_fields["id"].annotation, int)
        self.assertEqual(model.model_fields["name"].annotation, str)
        self.assertEqual(model.model_fields["price"].annotation, float)
        
        # Check optional fields
        self.assertTrue(hasattr(model.model_fields["description"].annotation, "__origin__"))
        self.assertEqual(model.model_fields["description"].annotation.__origin__, Union)
        
        # Create an instance
        product = model(
            id=1,
            name="Test Product",
            price=19.99,
            created_at=datetime.now()
        )
        self.assertEqual(product.id, 1)
        self.assertEqual(product.name, "Test Product")
        self.assertEqual(product.price, 19.99)
        self.assertIsNone(product.description)
    
    def test_create_models_from_tables(self):
        """Test creation of multiple Pydantic models from database tables."""
        # Use the mock schema
        tables = create_mock_db_schema()
        
        # Generate models
        models = DBModelGenerator.create_models_from_tables(tables)
        
        # Check that all tables were converted to models
        self.assertEqual(len(models), 3)
        self.assertIn("users", models)
        self.assertIn("posts", models)
        self.assertIn("comments", models)
        
        # Check relationships between models
        User = models["users"]
        Post = models["posts"]
        Comment = models["comments"]
        
        # Create instances with relationships
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True,
            created_at=datetime.now()
        )
        
        post = Post(
            id=1,
            user_id=user.id,  # Reference to user
            title="Test Post",
            content="This is a test post",
            published=True,
            created_at=datetime.now()
        )
        
        comment = Comment(
            id=1,
            post_id=post.id,  # Reference to post
            user_id=user.id,  # Reference to user
            content="Great post!",
            created_at=datetime.now()
        )
        
        # Verify relationships
        self.assertEqual(post.user_id, user.id)
        self.assertEqual(comment.post_id, post.id)
        self.assertEqual(comment.user_id, user.id)


if __name__ == "__main__":
    unittest.main()
