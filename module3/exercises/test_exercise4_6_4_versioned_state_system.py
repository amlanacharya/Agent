"""
Tests for Exercise 4.6.4: Versioned State System
---------------------------------------------
This module contains tests for the versioned state system implementation.
"""

import unittest
import os
import shutil
import json
from datetime import datetime, timedelta
import tempfile

from exercise4_6_2_conversation_state_machine import ConversationState
from exercise4_6_4_versioned_state_system import (
    SchemaVersionError,
    MigrationError,
    VersionedBaseModel,
    ConversationContextV1,
    ConversationContextV2,
    ConversationContextV3,
    StateTransitionV1,
    StateTransitionV2,
    StateTransitionV3,
    SchemaRegistry,
    VersionedStateSerializer,
    VersionedStateManager
)


class TestVersionedBaseModel(unittest.TestCase):
    """Test cases for the VersionedBaseModel class."""

    def test_schema_version(self):
        """Test schema version handling in base model."""
        class TestModel(VersionedBaseModel):
            model_config = {"schema_version": "2.5"}

        model = TestModel()
        self.assertEqual(model.schema_version, "1.0")  # Default value
        self.assertEqual(TestModel.get_schema_version(), "2.5")  # From model_config


class TestSchemaRegistry(unittest.TestCase):
    """Test cases for the SchemaRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = SchemaRegistry()

        # Register test schemas
        self.registry.register_schema("test_type", "1.0", ConversationContextV1)
        self.registry.register_schema("test_type", "2.0", ConversationContextV2)
        self.registry.register_schema("test_type", "3.0", ConversationContextV3)

        # Register migrations
        self.registry.register_migration(
            "test_type", "1.0", "2.0", ConversationContextV2.from_v1
        )
        self.registry.register_migration(
            "test_type", "2.0", "3.0", ConversationContextV3.from_v2
        )

    def test_register_schema(self):
        """Test registering schemas."""
        # Check that schemas were registered
        self.assertIn("test_type", self.registry.schemas)
        self.assertIn("1.0", self.registry.schemas["test_type"])
        self.assertIn("2.0", self.registry.schemas["test_type"])
        self.assertIn("3.0", self.registry.schemas["test_type"])

        # Check schema classes
        self.assertEqual(self.registry.schemas["test_type"]["1.0"], ConversationContextV1)
        self.assertEqual(self.registry.schemas["test_type"]["2.0"], ConversationContextV2)
        self.assertEqual(self.registry.schemas["test_type"]["3.0"], ConversationContextV3)

    def test_register_migration(self):
        """Test registering migrations."""
        # Check that migrations were registered
        self.assertIn("test_type", self.registry.migrations)
        self.assertIn(("1.0", "2.0"), self.registry.migrations["test_type"])
        self.assertIn(("2.0", "3.0"), self.registry.migrations["test_type"])

        # Check migration functions
        self.assertEqual(
            self.registry.migrations["test_type"][("1.0", "2.0")],
            ConversationContextV2.from_v1
        )
        self.assertEqual(
            self.registry.migrations["test_type"][("2.0", "3.0")],
            ConversationContextV3.from_v2
        )

    def test_get_schema(self):
        """Test getting schema by type and version."""
        # Get existing schema
        schema = self.registry.get_schema("test_type", "2.0")
        self.assertEqual(schema, ConversationContextV2)

        # Try to get non-existent schema
        with self.assertRaises(SchemaVersionError):
            self.registry.get_schema("test_type", "4.0")

        with self.assertRaises(SchemaVersionError):
            self.registry.get_schema("non_existent_type", "1.0")

    def test_get_latest_version(self):
        """Test getting the latest version for a schema type."""
        # Get latest version
        latest = self.registry.get_latest_version("test_type")
        self.assertEqual(latest, "3.0")

        # Try to get latest version for non-existent type
        with self.assertRaises(SchemaVersionError):
            self.registry.get_latest_version("non_existent_type")

    def test_migrate_direct(self):
        """Test direct migration between versions."""
        # Create a V1 context
        v1_context = ConversationContextV1(
            conversation_id="test-migration",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Migrate from V1 to V2
        v1_data = v1_context.model_dump()
        v2_data = self.registry.migrate("test_type", v1_data, "1.0", "2.0")

        # Check migration result
        self.assertEqual(v2_data["schema_version"], "2.0")
        self.assertEqual(v2_data["conversation_id"], "test-migration")
        self.assertEqual(v2_data["state"], ConversationState.GREETING.value)
        self.assertIsInstance(v2_data["state_history"], list)
        self.assertIsInstance(v2_data["state_history"][0], dict)
        self.assertEqual(v2_data["state_history"][0]["to_state"], ConversationState.GREETING.value)

    def test_migrate_path(self):
        """Test migration through a path of versions."""
        # Create a V1 context
        v1_context = ConversationContextV1(
            conversation_id="test-migration-path",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Migrate from V1 to V3 (should find path V1->V2->V3)
        v1_data = v1_context.model_dump()
        v3_data = self.registry.migrate("test_type", v1_data, "1.0", "3.0")

        # Check migration result
        self.assertEqual(v3_data["schema_version"], "3.0")
        self.assertEqual(v3_data["conversation_id"], "test-migration-path")
        self.assertEqual(v3_data["state"], ConversationState.GREETING.value)
        self.assertIn("tags", v3_data)
        self.assertIn("created_at", v3_data)
        self.assertIn("session_duration", v3_data)

    def test_migrate_no_path(self):
        """Test migration with no available path."""
        # Register a new schema with no migration path
        self.registry.register_schema("test_type", "4.0", VersionedBaseModel)

        # Create a V1 context
        v1_context = ConversationContextV1(
            conversation_id="test-no-path",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Try to migrate from V1 to V4 (no path)
        v1_data = v1_context.model_dump()
        with self.assertRaises(MigrationError):
            self.registry.migrate("test_type", v1_data, "1.0", "4.0")


class TestVersionedStateSerializer(unittest.TestCase):
    """Test cases for the VersionedStateSerializer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = SchemaRegistry()

        # Register schemas and migrations
        self.registry.register_schema("conversation_context", "1.0", ConversationContextV1)
        self.registry.register_schema("conversation_context", "2.0", ConversationContextV2)
        self.registry.register_schema("conversation_context", "3.0", ConversationContextV3)

        self.registry.register_migration(
            "conversation_context", "1.0", "2.0", ConversationContextV2.from_v1
        )
        self.registry.register_migration(
            "conversation_context", "2.0", "3.0", ConversationContextV3.from_v2
        )

        self.serializer = VersionedStateSerializer(self.registry)

        # Create a temporary directory for file tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_serialize(self):
        """Test serializing a versioned object."""
        # Create a V1 context
        context = ConversationContextV1(
            conversation_id="test-serialize",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Serialize
        data = self.serializer.serialize(context)

        # Check serialization result
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["conversation_id"], "test-serialize")
        self.assertEqual(data["state"], ConversationState.GREETING.value)
        self.assertIsInstance(data["state_history"], list)
        self.assertEqual(data["state_history"][0], ConversationState.GREETING.value)

    def test_deserialize(self):
        """Test deserializing data to a versioned object."""
        # Create serialized data
        data = {
            "schema_version": "1.0",
            "conversation_id": "test-deserialize",
            "user_id": "test-user",
            "state": ConversationState.GREETING.value,
            "state_history": [ConversationState.GREETING.value],
            "context_data": {}
        }

        # Deserialize with specific version to avoid auto-migration
        context = self.serializer.deserialize("conversation_context", data, "1.0")

        # Check deserialization result
        self.assertIsInstance(context, ConversationContextV1)
        self.assertEqual(context.schema_version, "1.0")
        self.assertEqual(context.conversation_id, "test-deserialize")
        self.assertEqual(context.state, ConversationState.GREETING.value)
        self.assertEqual(context.state_history, [ConversationState.GREETING.value])

    def test_deserialize_with_migration(self):
        """Test deserializing with automatic migration."""
        # Create serialized data for V1
        data = {
            "schema_version": "1.0",
            "conversation_id": "test-migrate",
            "user_id": "test-user",
            "state": ConversationState.GREETING.value,
            "state_history": [ConversationState.GREETING.value],
            "context_data": {}
        }

        # Deserialize to V3
        context = self.serializer.deserialize("conversation_context", data, "3.0")

        # Check deserialization result
        self.assertIsInstance(context, ConversationContextV3)
        self.assertEqual(context.schema_version, "3.0")
        self.assertEqual(context.conversation_id, "test-migrate")
        self.assertEqual(context.state, ConversationState.GREETING.value)
        self.assertIsInstance(context.state_history, list)
        self.assertIsInstance(context.state_history[0], dict)
        self.assertEqual(context.state_history[0]["to_state"], ConversationState.GREETING.value)
        self.assertIsInstance(context.tags, list)
        self.assertIsInstance(context.created_at, datetime)

    def test_save_load_json(self):
        """Test saving and loading to/from JSON."""
        # Create a context
        context = ConversationContextV2(
            conversation_id="test-json",
            user_id="test-user",
            state=ConversationState.COLLECTING_INFO.value,
            state_history=[{
                "from_state": "init",
                "to_state": ConversationState.GREETING.value,
                "timestamp": datetime.now().isoformat()
            }, {
                "from_state": ConversationState.GREETING.value,
                "to_state": ConversationState.COLLECTING_INFO.value,
                "timestamp": datetime.now().isoformat()
            }],
            metadata={"test": True}
        )

        # Save to JSON
        file_path = os.path.join(self.temp_dir, "test_context.json")
        self.serializer.save_to_json(context, file_path)

        # Check that file exists
        self.assertTrue(os.path.exists(file_path))

        # Load from JSON with specific version to avoid auto-migration
        loaded_context = self.serializer.load_from_json("conversation_context", file_path, "2.0")

        # Check loaded context
        self.assertIsInstance(loaded_context, ConversationContextV2)
        self.assertEqual(loaded_context.schema_version, "2.0")
        self.assertEqual(loaded_context.conversation_id, "test-json")
        self.assertEqual(loaded_context.state, ConversationState.COLLECTING_INFO.value)
        self.assertEqual(len(loaded_context.state_history), 2)
        self.assertEqual(loaded_context.metadata["test"], True)


class TestVersionedStateManager(unittest.TestCase):
    """Test cases for the VersionedStateManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for state storage
        self.temp_dir = tempfile.mkdtemp()
        self.manager = VersionedStateManager(storage_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_create_context(self):
        """Test creating contexts with different versions."""
        # Create V1 context
        context_v1 = self.manager.create_context(
            version="1.0",
            conversation_id="test-v1",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Check V1 context
        self.assertIsInstance(context_v1, ConversationContextV1)
        self.assertEqual(context_v1.schema_version, "1.0")
        self.assertEqual(context_v1.conversation_id, "test-v1")

        # Create V2 context
        context_v2 = self.manager.create_context(
            version="2.0",
            conversation_id="test-v2",
            user_id="test-user",
            state=ConversationState.COLLECTING_INFO.value,
            state_history=[{
                "from_state": "init",
                "to_state": ConversationState.GREETING.value,
                "timestamp": datetime.now().isoformat()
            }],
            metadata={"test": True}
        )

        # Check V2 context
        self.assertIsInstance(context_v2, ConversationContextV2)
        self.assertEqual(context_v2.schema_version, "2.0")
        self.assertEqual(context_v2.conversation_id, "test-v2")
        self.assertEqual(context_v2.metadata["test"], True)

        # Create V3 context (default)
        context_v3 = self.manager.create_context(
            conversation_id="test-v3",
            user_id="test-user",
            state=ConversationState.PROCESSING.value
        )

        # Check V3 context
        self.assertIsInstance(context_v3, ConversationContextV3)
        self.assertEqual(context_v3.schema_version, "3.0")
        self.assertEqual(context_v3.conversation_id, "test-v3")
        self.assertIsInstance(context_v3.tags, list)
        self.assertIsInstance(context_v3.created_at, datetime)

    def test_save_load_context(self):
        """Test saving and loading contexts."""
        # Create a context
        context = self.manager.create_context(
            version="2.0",
            conversation_id="test-save-load",
            user_id="test-user",
            state=ConversationState.COLLECTING_INFO.value,
            state_history=[{
                "from_state": "init",
                "to_state": ConversationState.GREETING.value,
                "timestamp": datetime.now().isoformat()
            }],
            metadata={"test": True}
        )

        # Save context
        self.manager.save_context(context, "test-save-load")

        # Check that file exists
        file_path = os.path.join(self.temp_dir, "context_test-save-load.json")
        self.assertTrue(os.path.exists(file_path))

        # Load context
        loaded_context = self.manager.load_context("test-save-load")

        # Check loaded context
        self.assertEqual(loaded_context.schema_version, "3.0")  # Auto-migrated to latest
        self.assertEqual(loaded_context.conversation_id, "test-save-load")
        self.assertEqual(loaded_context.state, ConversationState.COLLECTING_INFO.value)

        # Load with specific version
        v2_context = self.manager.load_context("test-save-load", "2.0")

        # Check loaded context with specific version
        self.assertEqual(v2_context.schema_version, "2.0")
        self.assertEqual(v2_context.conversation_id, "test-save-load")
        self.assertEqual(v2_context.state, ConversationState.COLLECTING_INFO.value)

    def test_migrate_context(self):
        """Test migrating a context to a new version."""
        # Create a V1 context
        context_v1 = self.manager.create_context(
            version="1.0",
            conversation_id="test-migrate",
            user_id="test-user",
            state=ConversationState.GREETING.value,
            state_history=[ConversationState.GREETING.value]
        )

        # Save context
        self.manager.save_context(context_v1, "test-migrate")

        # Migrate to V3
        migrated_context = self.manager.migrate_context("test-migrate", "3.0")

        # Check migrated context
        self.assertEqual(migrated_context.schema_version, "3.0")
        self.assertEqual(migrated_context.conversation_id, "test-migrate")
        self.assertEqual(migrated_context.state, ConversationState.GREETING.value)
        self.assertIsInstance(migrated_context.tags, list)
        self.assertIsInstance(migrated_context.created_at, datetime)

        # Check that migrated context was saved
        file_path = os.path.join(self.temp_dir, "context_test-migrate.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["schema_version"], "3.0")

    def test_list_available_contexts(self):
        """Test listing available contexts."""
        # Create and save contexts
        context_v1 = self.manager.create_context(
            version="1.0",
            conversation_id="test-list-1",
            user_id="test-user",
            state=ConversationState.GREETING.value
        )
        self.manager.save_context(context_v1, "test-list-1")

        context_v2 = self.manager.create_context(
            version="2.0",
            conversation_id="test-list-2",
            user_id="test-user",
            state=ConversationState.COLLECTING_INFO.value
        )
        self.manager.save_context(context_v2, "test-list-2")

        # List contexts
        contexts = self.manager.list_available_contexts()

        # Check list
        self.assertEqual(len(contexts), 2)
        self.assertIn(("test-list-1", "1.0"), contexts)
        self.assertIn(("test-list-2", "2.0"), contexts)

    def test_get_schema_version_info(self):
        """Test getting schema version information."""
        # Get version info
        version_info = self.manager.get_schema_version_info()

        # Check info
        self.assertIn("conversation_context", version_info)
        self.assertIn("state_transition", version_info)
        self.assertEqual(len(version_info["conversation_context"]), 3)
        self.assertEqual(len(version_info["state_transition"]), 3)
        self.assertIn("1.0", version_info["conversation_context"])
        self.assertIn("2.0", version_info["conversation_context"])
        self.assertIn("3.0", version_info["conversation_context"])


if __name__ == "__main__":
    unittest.main()
