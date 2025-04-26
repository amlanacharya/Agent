"""
Lesson 4.3.3: Database Model Generator

This exercise implements a dynamic model generator that creates Pydantic models
from database table schemas.
"""

from pydantic import BaseModel, Field, create_model
from typing import Dict, Any, Type, Optional, List, Union, get_type_hints
from datetime import datetime, date
import re


class DBColumn(BaseModel):
    """Database column definition."""
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None
    unique: bool = False
    default: Optional[Any] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: Optional[str] = None


class DBTable(BaseModel):
    """Database table definition."""
    name: str
    columns: List[DBColumn]
    description: Optional[str] = None


class DBModelGenerator:
    """Generator for Pydantic models from database schemas."""
    
    # Mapping from database types to Python types
    TYPE_MAPPING = {
        "integer": int,
        "bigint": int,
        "smallint": int,
        "decimal": float,
        "numeric": float,
        "real": float,
        "double": float,
        "varchar": str,
        "char": str,
        "text": str,
        "boolean": bool,
        "date": date,
        "timestamp": datetime,
        "json": Dict[str, Any],
        "jsonb": Dict[str, Any]
    }
    
    @classmethod
    def get_python_type(cls, column: DBColumn) -> Type:
        """Convert a database type to a Python type."""
        base_type = cls.TYPE_MAPPING.get(column.data_type.lower(), Any)
        
        # Handle nullable fields
        if column.nullable:
            return Optional[base_type]
        
        return base_type
    
    @classmethod
    def create_field_definition(cls, column: DBColumn) -> Dict[str, Any]:
        """Create a Pydantic field definition from a column definition."""
        field_def = {}
        
        # Add description if available
        if column.description:
            field_def["description"] = column.description
        
        # Add default value if specified
        if column.default is not None:
            field_def["default"] = column.default
        elif column.nullable:
            field_def["default"] = None
        
        # Add constraints based on column properties
        if column.max_length is not None and column.data_type.lower() in ["varchar", "char"]:
            field_def["max_length"] = column.max_length
        
        if column.min_value is not None:
            field_def["ge"] = column.min_value
        
        if column.max_value is not None:
            field_def["le"] = column.max_value
        
        return field_def
    
    @classmethod
    def create_model_from_table(cls, table: DBTable) -> Type[BaseModel]:
        """Create a Pydantic model from a database table definition."""
        fields = {}
        
        for column in table.columns:
            python_type = cls.get_python_type(column)
            field_def = cls.create_field_definition(column)
            
            # Create the field
            if field_def:
                fields[column.name] = (python_type, Field(**field_def))
            else:
                fields[column.name] = (python_type, ...)
        
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
                nullable=False,
                unique=True,
                max_length=50,
                description="Username"
            ),
            DBColumn(
                name="email",
                data_type="varchar",
                nullable=False,
                unique=True,
                max_length=100,
                description="Email address"
            ),
            DBColumn(
                name="password_hash",
                data_type="varchar",
                nullable=False,
                max_length=255,
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
                nullable=False,
                max_length=200,
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
                description="Post last update timestamp"
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
        
        # Get field information
        for field_name, field_info in model.model_fields.items():
            field_type = get_type_hints(model).get(field_name, Any)
            field_type_name = getattr(field_type, "__name__", str(field_type))
            
            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                if type(None) in field_type.__args__:
                    inner_type = [t for t in field_type.__args__ if t is not type(None)][0]
                    field_type_name = f"Optional[{getattr(inner_type, '__name__', str(inner_type))}]"
            
            print(f"  - {field_name}: {field_type_name}")
            
            # Display field constraints
            constraints = []
            if field_info.description:
                constraints.append(f"description='{field_info.description}'")
            if field_info.default is not None and field_info.default is not ...:
                constraints.append(f"default={field_info.default}")
            
            # Get other constraints from field info
            for constraint in ["max_length", "ge", "le"]:
                if hasattr(field_info, constraint) and getattr(field_info, constraint) is not None:
                    constraints.append(f"{constraint}={getattr(field_info, constraint)}")
            
            if constraints:
                print(f"    Constraints: {', '.join(constraints)}")
    
    # Create an instance of a generated model
    User = models["users"]
    user = User(
        id=1,
        username="johndoe",
        email="john@example.com",
        password_hash="hashed_password",
        is_active=True
    )
    print(f"\n{'-' * 40}")
    print("Example User instance:")
    print(user.model_dump_json(indent=2))
    
    # Create related instances
    Post = models["posts"]
    post = Post(
        id=1,
        user_id=user.id,
        title="Hello World",
        content="This is my first post!",
        published=True
    )
    print(f"\n{'-' * 40}")
    print("Example Post instance:")
    print(post.model_dump_json(indent=2))
    
    Comment = models["comments"]
    comment = Comment(
        id=1,
        post_id=post.id,
        user_id=user.id,
        content="Great post!"
    )
    print(f"\n{'-' * 40}")
    print("Example Comment instance:")
    print(comment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
