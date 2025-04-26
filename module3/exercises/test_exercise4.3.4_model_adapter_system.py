"""
Tests for Exercise 4.3.4: Model Adapter System
--------------------------------------------
This module contains tests for the model adapter system implementation.
"""

import unittest
from datetime import datetime
from typing import Dict, Any, Optional, List

from exercise4.3.4_model_adapter_system import (
    ModelAdapter,
    AdapterRegistry,
    CreateUserRequest,
    UpdateUserRequest,
    UserDB,
    UserProfileDB,
    UserResponse,
    UserDetailResponse,
    UserListResponse,
    hash_password
)


class TestModelAdapterSystem(unittest.TestCase):
    """Test cases for the model adapter system implementation."""
    
    def test_model_adapter_basic(self):
        """Test basic model adapter functionality."""
        # Create a simple adapter
        adapter = ModelAdapter(
            CreateUserRequest,
            UserDB,
            field_mapping={
                "password_hash": "password"  # Map password from request to password_hash in DB
            },
            transformers={
                "password_hash": hash_password  # Transform password to hashed password
            }
        )
        
        # Create a request
        request = CreateUserRequest(
            username="testuser",
            email="test@example.com",
            password="password123",
            full_name="Test User"
        )
        
        # Adapt to DB model
        user_db = adapter.adapt(request)
        
        # Check field mapping and transformation
        self.assertEqual(user_db.username, "testuser")
        self.assertEqual(user_db.email, "test@example.com")
        self.assertEqual(user_db.password_hash, "hashed_password123")
        self.assertEqual(user_db.full_name, "Test User")
        self.assertTrue(user_db.is_active)
        self.assertIsInstance(user_db.id, str)
        self.assertIsInstance(user_db.created_at, datetime)
    
    def test_model_adapter_with_extra_fields(self):
        """Test model adapter with extra fields."""
        adapter = ModelAdapter(
            UserDB,
            UserResponse,
            exclude_fields=["password_hash"]
        )
        
        # Create a DB model
        user_db = UserDB(
            id="user123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )
        
        # Adapt to response model with extra fields
        response = adapter.adapt(
            user_db,
            extra_field="This should be ignored"  # Not in target model
        )
        
        # Check fields
        self.assertEqual(response.id, "user123")
        self.assertEqual(response.username, "testuser")
        self.assertEqual(response.email, "test@example.com")
        self.assertEqual(response.full_name, "Test User")
        self.assertTrue(response.is_active)
        self.assertEqual(response.created_at, datetime(2023, 1, 1, 12, 0, 0))
        
        # Extra field should be ignored
        self.assertFalse(hasattr(response, "extra_field"))
        
        # Sensitive field should be excluded
        self.assertFalse(hasattr(response, "password_hash"))
    
    def test_model_adapter_many(self):
        """Test adapting multiple models."""
        adapter = ModelAdapter(
            UserDB,
            UserResponse,
            exclude_fields=["password_hash"]
        )
        
        # Create multiple DB models
        users_db = [
            UserDB(
                id=f"user{i}",
                username=f"user{i}",
                email=f"user{i}@example.com",
                password_hash=f"hashed_password{i}",
                created_at=datetime(2023, 1, i, 12, 0, 0)
            )
            for i in range(1, 4)
        ]
        
        # Adapt multiple models
        responses = adapter.adapt_many(users_db)
        
        # Check results
        self.assertEqual(len(responses), 3)
        for i, response in enumerate(responses, 1):
            self.assertEqual(response.id, f"user{i}")
            self.assertEqual(response.username, f"user{i}")
            self.assertEqual(response.email, f"user{i}@example.com")
            self.assertEqual(response.created_at, datetime(2023, 1, i, 12, 0, 0))
    
    def test_adapter_registry(self):
        """Test the adapter registry."""
        # Create registry
        registry = AdapterRegistry()
        
        # Create adapters
        create_adapter = ModelAdapter(
            CreateUserRequest,
            UserDB,
            field_mapping={"password_hash": "password"},
            transformers={"password_hash": hash_password}
        )
        
        response_adapter = ModelAdapter(
            UserDB,
            UserResponse,
            exclude_fields=["password_hash"]
        )
        
        # Register adapters
        registry.register(CreateUserRequest, UserDB, create_adapter)
        registry.register(UserDB, UserResponse, response_adapter)
        
        # Get adapters
        retrieved_create_adapter = registry.get_adapter(CreateUserRequest, UserDB)
        retrieved_response_adapter = registry.get_adapter(UserDB, UserResponse)
        
        # Check retrieval
        self.assertIs(retrieved_create_adapter, create_adapter)
        self.assertIs(retrieved_response_adapter, response_adapter)
        
        # Non-existent adapter should return None
        self.assertIsNone(registry.get_adapter(UserResponse, UserDB))
    
    def test_registry_adapt(self):
        """Test adapting models through the registry."""
        # Create registry
        registry = AdapterRegistry()
        
        # Register adapters
        registry.register(
            CreateUserRequest,
            UserDB,
            ModelAdapter(
                CreateUserRequest,
                UserDB,
                field_mapping={"password_hash": "password"},
                transformers={"password_hash": hash_password}
            )
        )
        
        registry.register(
            UserDB,
            UserResponse,
            ModelAdapter(
                UserDB,
                UserResponse,
                exclude_fields=["password_hash"]
            )
        )
        
        # Create request
        request = CreateUserRequest(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Adapt to DB model through registry
        user_db = registry.adapt(request, UserDB)
        
        # Check DB model
        self.assertEqual(user_db.username, "testuser")
        self.assertEqual(user_db.email, "test@example.com")
        self.assertEqual(user_db.password_hash, "hashed_password123")
        
        # Adapt to response model through registry
        response = registry.adapt(user_db, UserResponse)
        
        # Check response model
        self.assertEqual(response.username, "testuser")
        self.assertEqual(response.email, "test@example.com")
        self.assertFalse(hasattr(response, "password_hash"))
    
    def test_registry_adapt_many(self):
        """Test adapting multiple models through the registry."""
        # Create registry
        registry = AdapterRegistry()
        
        # Register adapter
        registry.register(
            UserDB,
            UserResponse,
            ModelAdapter(
                UserDB,
                UserResponse,
                exclude_fields=["password_hash"]
            )
        )
        
        # Create multiple DB models
        users_db = [
            UserDB(
                id=f"user{i}",
                username=f"user{i}",
                email=f"user{i}@example.com",
                password_hash=f"hashed_password{i}"
            )
            for i in range(1, 4)
        ]
        
        # Adapt multiple models through registry
        responses = registry.adapt_many(users_db, UserResponse)
        
        # Check results
        self.assertEqual(len(responses), 3)
        for i, response in enumerate(responses, 1):
            self.assertEqual(response.id, f"user{i}")
            self.assertEqual(response.username, f"user{i}")
            self.assertEqual(response.email, f"user{i}@example.com")
    
    def test_registry_adapt_error(self):
        """Test error handling when no adapter is registered."""
        registry = AdapterRegistry()
        
        # Create models
        request = CreateUserRequest(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Attempt to adapt with no registered adapter
        with self.assertRaises(ValueError):
            registry.adapt(request, UserDB)
    
    def test_complex_adaptation(self):
        """Test more complex adaptation scenarios."""
        # Create a user and profile
        user_db = UserDB(
            id="user123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User"
        )
        
        profile_db = UserProfileDB(
            user_id="user123",
            bio="Test bio",
            location="Test location",
            website="https://example.com",
            social_links={"twitter": "https://twitter.com/test"}
        )
        
        # Create an adapter that combines user and profile
        adapter = ModelAdapter(
            UserDB,
            UserDetailResponse,
            exclude_fields=["password_hash"]
        )
        
        # Adapt with profile fields
        detail_response = adapter.adapt(
            user_db,
            bio=profile_db.bio,
            location=profile_db.location,
            website=profile_db.website,
            social_links=profile_db.social_links
        )
        
        # Check combined fields
        self.assertEqual(detail_response.id, "user123")
        self.assertEqual(detail_response.username, "testuser")
        self.assertEqual(detail_response.email, "test@example.com")
        self.assertEqual(detail_response.bio, "Test bio")
        self.assertEqual(detail_response.location, "Test location")
        self.assertEqual(detail_response.website, "https://example.com")
        self.assertEqual(detail_response.social_links, {"twitter": "https://twitter.com/test"})


if __name__ == "__main__":
    unittest.main()
