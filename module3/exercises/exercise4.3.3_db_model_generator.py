"""
Exercise 4.3.3: Database Model Generator

This exercise implements a dynamic model generator that creates Pydantic models
from database table schemas.
"""

from pydantic import BaseModel, Field, create_model
from typing import Dict, Any, Type, Optional, List, Union, get_type_hints
from datetime import datetime, date
import re


# Define database column types and their Python equivalents
class DBColumn:
    """Database column definition."""
    def __init__(
        self,
        name: str,
        data_type: str,
        nullable: bool = True,
        primary_key: bool = False,
        default: Any = None,
        max_length: Optional[int] = None,
        foreign_key: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.name = name
        self.data_type = data_type.lower()
        self.nullable = nullable
        self.primary_key = primary_key
        self.default = default
        self.max_length = max_length
        self.foreign_key = foreign_key
        self.description = description


class DBTable:
    """Database table definition."""
    def __init__(
        self,
        name: str,
        columns: List[DBColumn],
        description: Optional[str] = None
    ):
        self.name = name
        self.columns = columns
        self.description = description


class DBModelGenerator:
    """Generator for Pydantic models from database schemas."""
    
    # Mapping of database types to Python types
    TYPE_MAPPING = {
        # SQLite types
        "integer": int,
        "real": float,
        "text": str,
        "blob": bytes,
        "boolean": bool,
        
        # PostgreSQL types
        "smallint": int,
        "integer": int,
        "bigint": int,
        "decimal": float,
        "numeric": float,
        "real": float,
        "double precision": float,
        "serial": int,
        "bigserial": int,
        "varchar": str,
        "char": str,
        "text": str,
        "bytea": bytes,
        "timestamp": datetime,
        "date": date,
        "time": str,
        "boolean": bool,
        "json": Dict[str, Any],
        "jsonb": Dict[str, Any],
        
        # MySQL types
        "tinyint": int,
        "smallint": int,
        "mediumint": int,
        "int": int,
        "bigint": int,
        "float": float,
        "double": float,
        "decimal": float,
        "char": str,
        "varchar": str,
        "tinytext": str,
        "text": str,
        "mediumtext": str,
        "longtext": str,
        "binary": bytes,
        "varbinary": bytes,
        "tinyblob": bytes,
        "blob": bytes,
        "mediumblob": bytes,
        "longblob": bytes,
        "date": date,
        "datetime": datetime,
        "timestamp": datetime,
        "time": str,
        "year": int,
        "boolean": bool,
        "bool": bool,
        "json": Dict[str, Any],
    }
    
    @classmethod
    def get_python_type(cls, db_column: DBColumn) -> Type:
        """Convert database column type to Python type."""
        # Extract base type from database type (e.g., varchar(255) -> varchar)
        base_type = re.sub(r'\(.*\)', '', db_column.data_type).strip()
        
        # Get Python type from mapping
        python_type = cls.TYPE_MAPPING.get(base_type, Any)
        
        # Make nullable if needed
        if db_column.nullable and python_type != Any:
            return Optional[python_type]
        
        return python_type
    
    @classmethod
    def get_field_constraints(cls, db_column: DBColumn) -> Dict[str, Any]:
        """Get Pydantic field constraints from database column."""
        constraints = {}
        
        # Add description if available
        if db_column.description:
            constraints["description"] = db_column.description
        
        # Add constraints based on data type
        base_type = re.sub(r'\(.*\)', '', db_column.data_type).strip()
        
        # String length constraints
        if base_type in ["varchar", "char"] and db_column.max_length:
            constraints["max_length"] = db_column.max_length
        
        # Primary key
        if db_column.primary_key:
            constraints["title"] = "Primary Key"
        
        # Foreign key
        if db_column.foreign_key:
            constraints["description"] = f"{db_column.description or ''} (Foreign Key: {db_column.foreign_key})"
        
        return constraints
    
    @classmethod
    def create_model_from_table(cls, table: DBTable) -> Type[BaseModel]:
        """Create a Pydantic model from a database table schema."""
        fields = {}
        
        for column in table.columns:
            # Get Python type
            python_type = cls.get_python_type(column)
            
            # Get field constraints
            constraints = cls.get_field_constraints(column)
            
            # Determine default value
            if column.primary_key and column.data_type.lower() == "integer":
                # Auto-incrementing primary key
                default = None if column.nullable else ...
            elif column.default is not None:
                default = column.default
            elif column.nullable:
                default = None
            else:
                default = ...
            
            # Create field definition
            if constraints:
                field_def = (python_type, Field(default, **constraints))
            else:
                field_def = (python_type, default)
            
            fields[column.name] = field_def
        
        # Create model class
        model_name = "".join(word.capitalize() for word in table.name.split("_"))
        model = create_model(
            model_name,
            __doc__=table.description,
            **fields
        )
        
        return model
    
    @classmethod
    def create_models_from_tables(cls, tables: List[DBTable]) -> Dict[str, Type[BaseModel]]:
        """Create Pydantic models from multiple database tables."""
        models = {}
        
        for table in tables:
            model = cls.create_model_from_table(table)
            models[table.name] = model
        
        return models


# Example usage with mock database schema
def create_mock_db_schema() -> List[DBTable]:
    """Create a mock database schema for demonstration."""
    # Users table
    users_table = DBTable(
        name="users",
        description="User accounts table",
        columns=[
            DBColumn(
                name="id",
                data_type="integer",
                nullable=False,
                primary_key=True,
                description="User ID"
            ),
            DBColumn(
                name="username",
                data_type="varchar",
                max_length=50,
                nullable=False,
                description="User's login name"
            ),
            DBColumn(
                name="email",
                data_type="varchar",
                max_length=100,
                nullable=False,
                description="User's email address"
            ),
            DBColumn(
                name="password_hash",
                data_type="varchar",
                max_length=255,
                nullable=False,
                description="Hashed password"
            ),
            DBColumn(
                name="is_active",
                data_type="boolean",
                nullable=False,
                default=True,
                description="Whether the user account is active"
            ),
            DBColumn(
                name="created_at",
                data_type="timestamp",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                description="Account creation timestamp"
            ),
            DBColumn(
                name="last_login",
                data_type="timestamp",
                nullable=True,
                description="Last login timestamp"
            )
        ]
    )
    
    # Posts table
    posts_table = DBTable(
        name="posts",
        description="Blog posts table",
        columns=[
            DBColumn(
                name="id",
                data_type="integer",
                nullable=False,
                primary_key=True,
                description="Post ID"
            ),
            DBColumn(
                name="user_id",
                data_type="integer",
                nullable=False,
                foreign_key="users.id",
                description="Author user ID"
            ),
            DBColumn(
                name="title",
                data_type="varchar",
                max_length=200,
                nullable=False,
                description="Post title"
            ),
            DBColumn(
                name="content",
                data_type="text",
                nullable=False,
                description="Post content"
            ),
            DBColumn(
                name="published",
                data_type="boolean",
                nullable=False,
                default=False,
                description="Whether the post is published"
            ),
            DBColumn(
                name="created_at",
                data_type="timestamp",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                description="Post creation timestamp"
            ),
            DBColumn(
                name="updated_at",
                data_type="timestamp",
                nullable=True,
                description="Last update timestamp"
            )
        ]
    )
    
    # Comments table
    comments_table = DBTable(
        name="comments",
        description="Post comments table",
        columns=[
            DBColumn(
                name="id",
                data_type="integer",
                nullable=False,
                primary_key=True,
                description="Comment ID"
            ),
            DBColumn(
                name="post_id",
                data_type="integer",
                nullable=False,
                foreign_key="posts.id",
                description="Post ID"
            ),
            DBColumn(
                name="user_id",
                data_type="integer",
                nullable=False,
                foreign_key="users.id",
                description="User ID"
            ),
            DBColumn(
                name="content",
                data_type="text",
                nullable=False,
                description="Comment content"
            ),
            DBColumn(
                name="created_at",
                data_type="timestamp",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                description="Comment creation timestamp"
            )
        ]
    )
    
    return [users_table, posts_table, comments_table]


def main():
    """Demonstrate the database model generator."""
    # Create mock database schema
    tables = create_mock_db_schema()
    
    # Generate models
    models = DBModelGenerator.create_models_from_tables(tables)
    
    # Display generated models
    print("Generated Pydantic Models:")
    for table_name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"Model for table '{table_name}':")
        print(f"Class name: {model.__name__}")
        print(f"Documentation: {model.__doc__}")
        print("Fields:")
        
        for field_name, field in model.model_fields.items():
            field_type = field.annotation
            field_info = []
            
            # Check if optional
            is_optional = False
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                args = field_type.__args__
                if type(None) in args:
                    is_optional = True
                    other_types = [arg for arg in args if arg is not type(None)]
                    if len(other_types) == 1:
                        field_type = other_types[0]
            
            # Get field type name
            type_name = getattr(field_type, "__name__", str(field_type))
            
            # Add optional indicator
            if is_optional:
                type_name = f"Optional[{type_name}]"
            
            field_info.append(f"Type: {type_name}")
            
            # Add default value
            if field.default is not None and field.default is not ...:
                field_info.append(f"Default: {field.default}")
            elif field.default_factory is not None:
                field_info.append("Has default factory")
            elif field.default is ...:
                field_info.append("Required")
            else:
                field_info.append("Optional")
            
            # Add constraints
            if field.description:
                field_info.append(f"Description: {field.description}")
            
            if hasattr(field, "max_length") and field.max_length:
                field_info.append(f"Max length: {field.max_length}")
            
            print(f"  - {field_name}: {', '.join(field_info)}")
    
    # Create example instances
    print("\n" + "=" * 60)
    print("Creating example instances:")
    
    # User instance
    User = models["users"]
    user = User(
        id=1,
        username="johndoe",
        email="john@example.com",
        password_hash="hashed_password",
        is_active=True,
        created_at=datetime.now()
    )
    print("\nUser instance:")
    print(user.model_dump_json(indent=2))
    
    # Post instance
    Post = models["posts"]
    post = Post(
        id=1,
        user_id=1,
        title="Hello World",
        content="This is my first post!",
        published=True,
        created_at=datetime.now()
    )
    print("\nPost instance:")
    print(post.model_dump_json(indent=2))
    
    # Comment instance
    Comment = models["comments"]
    comment = Comment(
        id=1,
        post_id=1,
        user_id=1,
        content="Great post!",
        created_at=datetime.now()
    )
    print("\nComment instance:")
    print(comment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
