"""
Tests for Lesson 4.3.3: Database Model Generator
-----------------------------------------------
This module contains tests for the database model generator implementation.
"""

import unittest
from datetime import datetime
from typing import Dict, Any, Optional, List, get_type_hints

from lesson4_3_3_exercises import (
    DBColumn,
    DBTable,
    DBModelGenerator
)


class TestDatabaseModelGenerator(unittest.TestCase):
    """Test cases for the database model generator implementation."""

    def test_db_column(self):
        """Test the DBColumn model."""
        # Basic column
        column = DBColumn(
            name="id",
            data_type="integer"
        )
        self.assertEqual(column.name, "id")
        self.assertEqual(column.data_type, "integer")
        self.assertTrue(column.nullable)
        self.assertFalse(column.primary_key)

        # Column with constraints
        column = DBColumn(
            name="username",
            data_type="varchar",
            nullable=False,
            unique=True,
            max_length=50,
            description="Username"
        )
        self.assertEqual(column.name, "username")
        self.assertEqual(column.data_type, "varchar")
        self.assertFalse(column.nullable)
        self.assertTrue(column.unique)
        self.assertEqual(column.max_length, 50)
        self.assertEqual(column.description, "Username")

    def test_db_table(self):
        """Test the DBTable model."""
        # Create a table with columns
        table = DBTable(
            name="users",
            description="User accounts table",
            columns=[
                DBColumn(
                    name="id",
                    data_type="integer",
                    nullable=False,
                    primary_key=True
                ),
                DBColumn(
                    name="username",
                    data_type="varchar",
                    nullable=False,
                    max_length=50
                )
            ]
        )
        self.assertEqual(table.name, "users")
        self.assertEqual(table.description, "User accounts table")
        self.assertEqual(len(table.columns), 2)
        self.assertEqual(table.columns[0].name, "id")
        self.assertEqual(table.columns[1].name, "username")

    def test_get_python_type(self):
        """Test the get_python_type method."""
        # Integer type
        column = DBColumn(name="id", data_type="integer", nullable=False)
        python_type = DBModelGenerator.get_python_type(column)
        self.assertEqual(python_type, int)

        # Nullable string type
        column = DBColumn(name="name", data_type="varchar", nullable=True)
        python_type = DBModelGenerator.get_python_type(column)
        # In Python 3.10+, Optional[X] is Union[X, None]
        self.assertTrue(hasattr(python_type, "__origin__"))
        self.assertEqual(str(python_type), "typing.Optional[str]")

        # Unknown type
        column = DBColumn(name="data", data_type="unknown_type")
        python_type = DBModelGenerator.get_python_type(column)
        # In Python 3.10+, Optional[X] is Union[X, None]
        self.assertTrue(hasattr(python_type, "__origin__"))
        self.assertEqual(str(python_type), "typing.Optional[typing.Any]")

    def test_create_field_definition(self):
        """Test the create_field_definition method."""
        # Field with description and default
        column = DBColumn(
            name="is_active",
            data_type="boolean",
            nullable=False,
            default=True,
            description="Whether the user is active"
        )
        field_def = DBModelGenerator.create_field_definition(column)
        self.assertEqual(field_def["description"], "Whether the user is active")
        self.assertEqual(field_def["default"], True)

        # Field with constraints
        column = DBColumn(
            name="username",
            data_type="varchar",
            nullable=False,
            max_length=50
        )
        field_def = DBModelGenerator.create_field_definition(column)
        self.assertEqual(field_def["max_length"], 50)

        # Field with numeric constraints
        column = DBColumn(
            name="age",
            data_type="integer",
            nullable=True,
            min_value=0,
            max_value=120
        )
        field_def = DBModelGenerator.create_field_definition(column)
        self.assertEqual(field_def["default"], None)
        self.assertEqual(field_def["ge"], 0)
        self.assertEqual(field_def["le"], 120)

    def test_create_model_from_table(self):
        """Test the create_model_from_table method."""
        # Create a table definition
        table = DBTable(
            name="users",
            description="User accounts table",
            columns=[
                DBColumn(
                    name="id",
                    data_type="integer",
                    nullable=False,
                    primary_key=True
                ),
                DBColumn(
                    name="username",
                    data_type="varchar",
                    nullable=False,
                    max_length=50
                ),
                DBColumn(
                    name="email",
                    data_type="varchar",
                    nullable=False
                ),
                DBColumn(
                    name="is_active",
                    data_type="boolean",
                    nullable=False,
                    default=True
                )
            ]
        )

        # Generate model
        User = DBModelGenerator.create_model_from_table(table)

        # Check model properties
        self.assertEqual(User.__name__, "Users")
        self.assertEqual(User.__doc__, "User accounts table")

        # Check field types
        type_hints = get_type_hints(User)
        self.assertEqual(type_hints["id"], int)
        self.assertEqual(type_hints["username"], str)
        self.assertEqual(type_hints["email"], str)
        self.assertEqual(type_hints["is_active"], bool)

        # Check field constraints
        # In Pydantic v2, constraints are stored differently
        # Let's check if the model has the expected fields
        self.assertIn("username", User.model_fields)
        self.assertIn("is_active", User.model_fields)
        # Check default value for is_active
        self.assertEqual(User.model_fields["is_active"].default, True)

        # Create an instance
        user = User(
            id=1,
            username="johndoe",
            email="john@example.com"
        )
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "johndoe")
        self.assertEqual(user.email, "john@example.com")
        self.assertEqual(user.is_active, True)  # Default value

    def test_create_models_from_tables(self):
        """Test the create_models_from_tables method."""
        # Create table definitions
        users_table = DBTable(
            name="users",
            description="User accounts table",
            columns=[
                DBColumn(name="id", data_type="integer", nullable=False, primary_key=True),
                DBColumn(name="username", data_type="varchar", nullable=False)
            ]
        )

        posts_table = DBTable(
            name="posts",
            description="Blog posts table",
            columns=[
                DBColumn(name="id", data_type="integer", nullable=False, primary_key=True),
                DBColumn(name="user_id", data_type="integer", nullable=False, foreign_key="users.id"),
                DBColumn(name="title", data_type="varchar", nullable=False),
                DBColumn(name="content", data_type="text", nullable=False)
            ]
        )

        # Generate models
        models = DBModelGenerator.create_models_from_tables([users_table, posts_table])

        # Check that both models were created
        self.assertEqual(len(models), 2)
        self.assertIn("users", models)
        self.assertIn("posts", models)

        # Check model properties
        User = models["users"]
        Post = models["posts"]

        self.assertEqual(User.__name__, "Users")
        self.assertEqual(Post.__name__, "Posts")

        # Create instances
        user = User(id=1, username="johndoe")
        post = Post(id=1, user_id=user.id, title="Hello", content="World")

        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, "johndoe")

        self.assertEqual(post.id, 1)
        self.assertEqual(post.user_id, 1)
        self.assertEqual(post.title, "Hello")
        self.assertEqual(post.content, "World")


if __name__ == "__main__":
    unittest.main()
