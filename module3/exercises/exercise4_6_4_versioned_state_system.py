"""
Exercise 4.6.4: Versioned State System
------------------------------------
This module implements a versioned state system that can handle state evolution
as the agent evolves. It provides mechanisms for:

1. Defining different versions of state schemas
2. Migrating between schema versions
3. Serializing and deserializing state with version awareness
4. Handling backward and forward compatibility
5. Managing state evolution over time
"""

import json
import os
import copy
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Type, ClassVar, Callable, Tuple, Set
from pydantic import BaseModel, Field, model_validator

# Import the conversation state machine from the previous exercise
from exercise4_6_2_conversation_state_machine import ConversationState


class SchemaVersionError(Exception):
    """Exception raised when there are issues with schema versions."""
    pass


class MigrationError(Exception):
    """Exception raised when state migration fails."""
    pass


class VersionedBaseModel(BaseModel):
    """Base model with versioning support."""
    schema_version: str = Field(default="1.0")
    
    @classmethod
    def get_schema_version(cls) -> str:
        """Get the schema version of this model."""
        return cls.model_config.get("schema_version", "1.0")
    
    model_config = {"schema_version": "1.0"}


class ConversationContextV1(VersionedBaseModel):
    """Version 1 of the conversation context schema."""
    schema_version: str = Field(default="1.0", frozen=True)
    conversation_id: str = Field(default="")
    user_id: Optional[str] = None
    state: str = Field(default=ConversationState.GREETING.value)
    state_history: List[str] = Field(default_factory=list)
    last_state_change: datetime = Field(default_factory=datetime.now)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"schema_version": "1.0"}
    
    @model_validator(mode='after')
    def validate_state(self):
        """Validate that the state is a valid ConversationState value."""
        valid_states = [state.value for state in ConversationState]
        if self.state not in valid_states:
            raise ValueError(f"Invalid state: {self.state}. Must be one of {valid_states}")
        return self


class StateTransitionV1(VersionedBaseModel):
    """Version 1 of the state transition schema."""
    schema_version: str = Field(default="1.0", frozen=True)
    from_state: str
    to_state: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"schema_version": "1.0"}


class ConversationContextV2(VersionedBaseModel):
    """Version 2 of the conversation context schema."""
    schema_version: str = Field(default="2.0", frozen=True)
    conversation_id: str = Field(default="")
    user_id: Optional[str] = None
    state: str = Field(default=ConversationState.GREETING.value)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)  # Changed from List[str] to List[Dict]
    last_state_change: datetime = Field(default_factory=datetime.now)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # New field
    
    model_config = {"schema_version": "2.0"}
    
    @model_validator(mode='after')
    def validate_state(self):
        """Validate that the state is a valid ConversationState value."""
        valid_states = [state.value for state in ConversationState]
        if self.state not in valid_states:
            raise ValueError(f"Invalid state: {self.state}. Must be one of {valid_states}")
        return self
    
    @classmethod
    def from_v1(cls, v1_context: ConversationContextV1) -> "ConversationContextV2":
        """Convert from V1 schema to V2 schema."""
        # Convert simple state history to structured state history
        structured_history = []
        for i, state in enumerate(v1_context.state_history):
            # For the first state, use "init" as from_state
            if i == 0:
                from_state = "init"
            else:
                from_state = v1_context.state_history[i-1]
                
            structured_history.append({
                "from_state": from_state,
                "to_state": state,
                "timestamp": (v1_context.last_state_change if i == len(v1_context.state_history) - 1 
                             else datetime.now().isoformat())
            })
        
        # Create V2 context
        return cls(
            conversation_id=v1_context.conversation_id,
            user_id=v1_context.user_id,
            state=v1_context.state,
            state_history=structured_history,
            last_state_change=v1_context.last_state_change,
            context_data=v1_context.context_data,
            metadata={"migrated_from": "1.0", "migration_time": datetime.now().isoformat()}
        )


class StateTransitionV2(VersionedBaseModel):
    """Version 2 of the state transition schema."""
    schema_version: str = Field(default="2.0", frozen=True)
    from_state: str
    to_state: str
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: Optional[str] = None  # New field
    metadata: Dict[str, Any] = Field(default_factory=dict)  # New field
    
    model_config = {"schema_version": "2.0"}
    
    @classmethod
    def from_v1(cls, v1_transition: StateTransitionV1) -> "StateTransitionV2":
        """Convert from V1 schema to V2 schema."""
        return cls(
            from_state=v1_transition.from_state,
            to_state=v1_transition.to_state,
            timestamp=v1_transition.timestamp,
            metadata={"migrated_from": "1.0", "migration_time": datetime.now().isoformat()}
        )


class ConversationContextV3(VersionedBaseModel):
    """Version 3 of the conversation context schema."""
    schema_version: str = Field(default="3.0", frozen=True)
    conversation_id: str = Field(default="")
    user_id: Optional[str] = None
    state: str = Field(default=ConversationState.GREETING.value)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_state_change: datetime = Field(default_factory=datetime.now)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)  # New field
    created_at: datetime = Field(default_factory=datetime.now)  # New field
    session_duration: Optional[float] = None  # New field
    
    model_config = {"schema_version": "3.0"}
    
    @model_validator(mode='after')
    def validate_state(self):
        """Validate that the state is a valid ConversationState value."""
        valid_states = [state.value for state in ConversationState]
        if self.state not in valid_states:
            raise ValueError(f"Invalid state: {self.state}. Must be one of {valid_states}")
        return self
    
    @classmethod
    def from_v2(cls, v2_context: ConversationContextV2) -> "ConversationContextV3":
        """Convert from V2 schema to V3 schema."""
        # Update metadata to include migration info
        metadata = copy.deepcopy(v2_context.metadata)
        metadata.update({
            "migrated_from": "2.0",
            "migration_time": datetime.now().isoformat()
        })
        
        # Create V3 context
        return cls(
            conversation_id=v2_context.conversation_id,
            user_id=v2_context.user_id,
            state=v2_context.state,
            state_history=v2_context.state_history,
            last_state_change=v2_context.last_state_change,
            context_data=v2_context.context_data,
            metadata=metadata,
            tags=[],  # Initialize with empty tags
            created_at=datetime.now(),  # Use current time as creation time
            session_duration=None  # Initialize with no duration
        )


class StateTransitionV3(VersionedBaseModel):
    """Version 3 of the state transition schema."""
    schema_version: str = Field(default="3.0", frozen=True)
    from_state: str
    to_state: str
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration: Optional[float] = None  # New field: time spent in from_state
    user_action: Optional[str] = None  # New field: action that triggered transition
    
    model_config = {"schema_version": "3.0"}
    
    @classmethod
    def from_v2(cls, v2_transition: StateTransitionV2) -> "StateTransitionV3":
        """Convert from V2 schema to V3 schema."""
        # Update metadata to include migration info
        metadata = copy.deepcopy(v2_transition.metadata)
        metadata.update({
            "migrated_from": "2.0",
            "migration_time": datetime.now().isoformat()
        })
        
        return cls(
            from_state=v2_transition.from_state,
            to_state=v2_transition.to_state,
            timestamp=v2_transition.timestamp,
            reason=v2_transition.reason,
            metadata=metadata,
            duration=None,  # Initialize with no duration
            user_action=None  # Initialize with no user action
        )


class SchemaRegistry:
    """Registry of schema versions and migration functions."""
    
    def __init__(self):
        """Initialize the schema registry."""
        self.schemas: Dict[str, Dict[str, Type[VersionedBaseModel]]] = {}
        self.migrations: Dict[str, Dict[Tuple[str, str], Callable]] = {}
    
    def register_schema(self, schema_type: str, version: str, schema_class: Type[VersionedBaseModel]):
        """
        Register a schema class for a specific type and version.
        
        Args:
            schema_type: Type of schema (e.g., "conversation_context")
            version: Schema version (e.g., "1.0")
            schema_class: Pydantic model class for the schema
        """
        if schema_type not in self.schemas:
            self.schemas[schema_type] = {}
        
        self.schemas[schema_type][version] = schema_class
    
    def register_migration(self, schema_type: str, from_version: str, to_version: str, 
                          migration_func: Callable):
        """
        Register a migration function between schema versions.
        
        Args:
            schema_type: Type of schema (e.g., "conversation_context")
            from_version: Source schema version
            to_version: Target schema version
            migration_func: Function to migrate from source to target version
        """
        if schema_type not in self.migrations:
            self.migrations[schema_type] = {}
        
        self.migrations[schema_type][(from_version, to_version)] = migration_func
    
    def get_schema(self, schema_type: str, version: str) -> Type[VersionedBaseModel]:
        """
        Get a schema class by type and version.
        
        Args:
            schema_type: Type of schema
            version: Schema version
            
        Returns:
            Schema class
            
        Raises:
            SchemaVersionError: If schema not found
        """
        if schema_type not in self.schemas or version not in self.schemas[schema_type]:
            raise SchemaVersionError(f"Schema not found: {schema_type} version {version}")
        
        return self.schemas[schema_type][version]
    
    def get_latest_version(self, schema_type: str) -> str:
        """
        Get the latest version for a schema type.
        
        Args:
            schema_type: Type of schema
            
        Returns:
            Latest version string
            
        Raises:
            SchemaVersionError: If no schemas found for type
        """
        if schema_type not in self.schemas or not self.schemas[schema_type]:
            raise SchemaVersionError(f"No schemas found for type: {schema_type}")
        
        # Sort versions and return the highest
        versions = list(self.schemas[schema_type].keys())
        return sorted(versions, key=lambda v: [int(x) for x in v.split('.')])[-1]
    
    def migrate(self, schema_type: str, data: Dict[str, Any], 
               from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Migrate data from one schema version to another.
        
        Args:
            schema_type: Type of schema
            data: Data to migrate
            from_version: Source schema version
            to_version: Target schema version
            
        Returns:
            Migrated data
            
        Raises:
            MigrationError: If migration path not found
        """
        if from_version == to_version:
            return data
        
        # Check if direct migration exists
        if (schema_type in self.migrations and 
            (from_version, to_version) in self.migrations[schema_type]):
            # Get source and target schema classes
            source_schema = self.get_schema(schema_type, from_version)
            
            # Parse data with source schema
            source_obj = source_schema(**data)
            
            # Apply migration function
            migration_func = self.migrations[schema_type][(from_version, to_version)]
            target_obj = migration_func(source_obj)
            
            # Return as dict
            return target_obj.model_dump()
        
        # Try to find a migration path
        path = self._find_migration_path(schema_type, from_version, to_version)
        if not path:
            raise MigrationError(
                f"No migration path found from {from_version} to {to_version} for {schema_type}"
            )
        
        # Apply migrations in sequence
        current_data = data
        for i in range(len(path) - 1):
            current_version = path[i]
            next_version = path[i + 1]
            current_data = self.migrate(schema_type, current_data, current_version, next_version)
        
        return current_data
    
    def _find_migration_path(self, schema_type: str, from_version: str, 
                            to_version: str) -> Optional[List[str]]:
        """
        Find a path of migrations from source to target version.
        
        Args:
            schema_type: Type of schema
            from_version: Source schema version
            to_version: Target schema version
            
        Returns:
            List of versions forming a path, or None if no path found
        """
        if schema_type not in self.migrations:
            return None
        
        # Build a graph of version transitions
        graph: Dict[str, Set[str]] = {}
        for (src, tgt) in self.migrations[schema_type].keys():
            if src not in graph:
                graph[src] = set()
            graph[src].add(tgt)
        
        # Breadth-first search for a path
        queue = [(from_version, [from_version])]
        visited = {from_version}
        
        while queue:
            (current, path) = queue.pop(0)
            
            if current == to_version:
                return path
            
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return None


class VersionedStateSerializer:
    """Serializes and deserializes versioned state objects."""
    
    def __init__(self, registry: SchemaRegistry):
        """
        Initialize the serializer with a schema registry.
        
        Args:
            registry: Schema registry to use for versioning
        """
        self.registry = registry
    
    def serialize(self, obj: VersionedBaseModel) -> Dict[str, Any]:
        """
        Serialize a versioned object to a dictionary.
        
        Args:
            obj: Versioned object to serialize
            
        Returns:
            Dictionary representation with version information
        """
        data = obj.model_dump()
        
        # Ensure schema_version is included
        if "schema_version" not in data:
            data["schema_version"] = obj.schema_version
        
        return data
    
    def deserialize(self, schema_type: str, data: Dict[str, Any], 
                   target_version: Optional[str] = None) -> VersionedBaseModel:
        """
        Deserialize data to a versioned object, migrating if necessary.
        
        Args:
            schema_type: Type of schema
            data: Data to deserialize
            target_version: Optional target version (defaults to latest)
            
        Returns:
            Deserialized object
            
        Raises:
            SchemaVersionError: If schema version not found
        """
        # Get version from data
        version = data.get("schema_version")
        if not version:
            raise SchemaVersionError("No schema_version found in data")
        
        # Determine target version
        if not target_version:
            target_version = self.registry.get_latest_version(schema_type)
        
        # Migrate if necessary
        if version != target_version:
            data = self.registry.migrate(schema_type, data, version, target_version)
        
        # Get schema class and instantiate
        schema_class = self.registry.get_schema(schema_type, target_version)
        return schema_class(**data)
    
    def save_to_json(self, obj: VersionedBaseModel, file_path: str):
        """
        Save a versioned object to a JSON file.
        
        Args:
            obj: Versioned object to save
            file_path: Path to save to
        """
        data = self.serialize(obj)
        
        # Convert datetime objects to ISO format
        data = self._convert_datetimes_to_iso(data)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, schema_type: str, file_path: str, 
                      target_version: Optional[str] = None) -> VersionedBaseModel:
        """
        Load a versioned object from a JSON file.
        
        Args:
            schema_type: Type of schema
            file_path: Path to load from
            target_version: Optional target version (defaults to latest)
            
        Returns:
            Deserialized object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return self.deserialize(schema_type, data, target_version)
    
    def _convert_datetimes_to_iso(self, data: Any) -> Any:
        """
        Recursively convert datetime objects to ISO format strings.
        
        Args:
            data: Data to convert
            
        Returns:
            Converted data
        """
        if isinstance(data, dict):
            return {k: self._convert_datetimes_to_iso(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_datetimes_to_iso(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data


class VersionedStateManager:
    """
    Manages versioned state objects with schema evolution support.
    
    This class provides a high-level interface for working with versioned state,
    including loading, saving, and migrating between versions.
    """
    
    def __init__(self, storage_dir: str = "state_storage"):
        """
        Initialize the versioned state manager.
        
        Args:
            storage_dir: Directory for state storage
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Create and configure schema registry
        self.registry = SchemaRegistry()
        self._register_schemas()
        
        # Create serializer
        self.serializer = VersionedStateSerializer(self.registry)
    
    def _register_schemas(self):
        """Register all schema versions and migrations."""
        # Register conversation context schemas
        self.registry.register_schema("conversation_context", "1.0", ConversationContextV1)
        self.registry.register_schema("conversation_context", "2.0", ConversationContextV2)
        self.registry.register_schema("conversation_context", "3.0", ConversationContextV3)
        
        # Register state transition schemas
        self.registry.register_schema("state_transition", "1.0", StateTransitionV1)
        self.registry.register_schema("state_transition", "2.0", StateTransitionV2)
        self.registry.register_schema("state_transition", "3.0", StateTransitionV3)
        
        # Register migrations
        self.registry.register_migration(
            "conversation_context", "1.0", "2.0", ConversationContextV2.from_v1
        )
        self.registry.register_migration(
            "conversation_context", "2.0", "3.0", ConversationContextV3.from_v2
        )
        self.registry.register_migration(
            "state_transition", "1.0", "2.0", StateTransitionV2.from_v1
        )
        self.registry.register_migration(
            "state_transition", "2.0", "3.0", StateTransitionV3.from_v2
        )
    
    def create_context(self, version: str = "3.0", **kwargs) -> VersionedBaseModel:
        """
        Create a new conversation context with the specified version.
        
        Args:
            version: Schema version to use
            **kwargs: Additional arguments for the context
            
        Returns:
            New conversation context object
        """
        schema_class = self.registry.get_schema("conversation_context", version)
        return schema_class(**kwargs)
    
    def save_context(self, context: VersionedBaseModel, conversation_id: str):
        """
        Save a conversation context to storage.
        
        Args:
            context: Context to save
            conversation_id: Conversation ID for filename
        """
        file_path = os.path.join(self.storage_dir, f"context_{conversation_id}.json")
        self.serializer.save_to_json(context, file_path)
    
    def load_context(self, conversation_id: str, 
                    target_version: Optional[str] = None) -> VersionedBaseModel:
        """
        Load a conversation context from storage.
        
        Args:
            conversation_id: Conversation ID to load
            target_version: Optional target version to migrate to
            
        Returns:
            Loaded context object
        """
        file_path = os.path.join(self.storage_dir, f"context_{conversation_id}.json")
        return self.serializer.load_from_json("conversation_context", file_path, target_version)
    
    def list_available_contexts(self) -> List[Tuple[str, str]]:
        """
        List available contexts in storage with their versions.
        
        Returns:
            List of (conversation_id, version) tuples
        """
        contexts = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("context_") and filename.endswith(".json"):
                conversation_id = filename[8:-5]  # Remove "context_" and ".json"
                
                try:
                    with open(os.path.join(self.storage_dir, filename), 'r') as f:
                        data = json.load(f)
                        version = data.get("schema_version", "unknown")
                        contexts.append((conversation_id, version))
                except (json.JSONDecodeError, KeyError):
                    # Skip invalid files
                    continue
        
        return contexts
    
    def migrate_context(self, conversation_id: str, target_version: str) -> VersionedBaseModel:
        """
        Migrate a context to a new version.
        
        Args:
            conversation_id: Conversation ID to migrate
            target_version: Target version to migrate to
            
        Returns:
            Migrated context object
        """
        # Load the context (this will not migrate it yet)
        file_path = os.path.join(self.storage_dir, f"context_{conversation_id}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get current version
        current_version = data.get("schema_version")
        if not current_version:
            raise SchemaVersionError("No schema_version found in context data")
        
        # Migrate the data
        migrated_data = self.registry.migrate(
            "conversation_context", data, current_version, target_version
        )
        
        # Create object from migrated data
        schema_class = self.registry.get_schema("conversation_context", target_version)
        context = schema_class(**migrated_data)
        
        # Save the migrated context
        self.save_context(context, conversation_id)
        
        return context
    
    def get_schema_version_info(self) -> Dict[str, List[str]]:
        """
        Get information about available schema versions.
        
        Returns:
            Dictionary mapping schema types to lists of available versions
        """
        info = {}
        
        for schema_type in self.registry.schemas:
            info[schema_type] = sorted(
                self.registry.schemas[schema_type].keys(),
                key=lambda v: [int(x) for x in v.split('.')]
            )
        
        return info


# Example usage
if __name__ == "__main__":
    # Create a versioned state manager
    manager = VersionedStateManager()
    
    # Print available schema versions
    version_info = manager.get_schema_version_info()
    print("Available schema versions:")
    for schema_type, versions in version_info.items():
        print(f"  {schema_type}: {', '.join(versions)}")
    
    # Create contexts with different versions
    context_v1 = manager.create_context(
        version="1.0",
        conversation_id="demo-v1",
        user_id="user123",
        state=ConversationState.GREETING.value,
        state_history=[ConversationState.GREETING.value]
    )
    
    context_v2 = manager.create_context(
        version="2.0",
        conversation_id="demo-v2",
        user_id="user456",
        state=ConversationState.COLLECTING_INFO.value,
        state_history=[{
            "from_state": "init",
            "to_state": ConversationState.GREETING.value,
            "timestamp": datetime.now().isoformat()
        }, {
            "from_state": ConversationState.GREETING.value,
            "to_state": ConversationState.COLLECTING_INFO.value,
            "timestamp": datetime.now().isoformat(),
            "reason": "User provided query"
        }],
        context_data={"query_type": "weather"},
        metadata={"source": "demo"}
    )
    
    context_v3 = manager.create_context(
        version="3.0",
        conversation_id="demo-v3",
        user_id="user789",
        state=ConversationState.PROCESSING.value,
        state_history=[{
            "from_state": "init",
            "to_state": ConversationState.GREETING.value,
            "timestamp": datetime.now().isoformat()
        }, {
            "from_state": ConversationState.GREETING.value,
            "to_state": ConversationState.COLLECTING_INFO.value,
            "timestamp": datetime.now().isoformat(),
            "reason": "User provided query"
        }, {
            "from_state": ConversationState.COLLECTING_INFO.value,
            "to_state": ConversationState.PROCESSING.value,
            "timestamp": datetime.now().isoformat(),
            "reason": "Collected all required information",
            "duration": 15.5,
            "user_action": "provided_location"
        }],
        context_data={"query_type": "weather", "location": "New York"},
        metadata={"source": "demo"},
        tags=["weather", "location"],
        session_duration=30.5
    )
    
    # Save contexts
    print("\nSaving contexts...")
    manager.save_context(context_v1, "demo-v1")
    manager.save_context(context_v2, "demo-v2")
    manager.save_context(context_v3, "demo-v3")
    
    # List available contexts
    contexts = manager.list_available_contexts()
    print("\nAvailable contexts:")
    for conversation_id, version in contexts:
        print(f"  {conversation_id} (version {version})")
    
    # Load and migrate a V1 context to V3
    print("\nMigrating V1 context to V3...")
    migrated_context = manager.migrate_context("demo-v1", "3.0")
    print(f"Migrated context version: {migrated_context.schema_version}")
    print(f"Migrated context state: {migrated_context.state}")
    print(f"Migrated context has tags: {migrated_context.tags}")
    print(f"Migrated context has session_duration: {migrated_context.session_duration}")
    
    # Load a context with automatic migration to latest version
    print("\nLoading context with automatic migration...")
    loaded_context = manager.load_context("demo-v2")  # Will migrate to V3
    print(f"Loaded context version: {loaded_context.schema_version}")
    print(f"Loaded context state: {loaded_context.state}")
    
    # Load a context with specific version target
    print("\nLoading context with specific version target...")
    v2_context = manager.load_context("demo-v3", "2.0")  # Will downgrade to V2 if possible
    print(f"Loaded context version: {v2_context.schema_version}")
    print(f"Loaded context state: {v2_context.state}")
