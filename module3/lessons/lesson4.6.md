# Lesson 4.6: State Validation in Agent Systems 🔄

<img src="https://github.com/user-attachments/assets/25117f1e-d4cf-40df-8103-2afb4c4ff69a" width="50%" height="50%"/>

## 📋 Overview

In this lesson, we'll explore state validation in agent systems. Unlike traditional applications with well-defined state transitions, agent systems often maintain complex, evolving states across multiple interactions. Proper state validation ensures data consistency, prevents invalid state transitions, and helps recover from errors.

## 🧩 The Challenge of Agent State

Agent state management presents several validation challenges:

1. **Persistence**: Maintaining consistent state across multiple interactions
2. **Transitions**: Ensuring state changes follow valid paths
3. **Concurrency**: Handling multiple users or parallel conversations
4. **Recovery**: Restoring valid state after errors or interruptions
5. **Versioning**: Managing state evolution as the agent evolves

![State Validation](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXo1ZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZWJtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKT3qDNYRxNvUmk/giphy.gif)

## 🛠️ Basic State Validation

Let's start with basic state validation using Pydantic:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid

class AgentState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    conversation_turns: int = 0
    current_context: Dict[str, Any] = {}
    
    @model_validator(mode='after')
    def validate_timestamps(self):
        """Validate that timestamps are logical."""
        if self.last_updated < self.created_at:
            raise ValueError("last_updated cannot be earlier than created_at")
        return self
    
    @model_validator(mode='after')
    def validate_conversation_turns(self):
        """Validate that conversation turns is non-negative."""
        if self.conversation_turns < 0:
            raise ValueError("conversation_turns cannot be negative")
        return self
    
    def update(self, **kwargs):
        """Update state with new values."""
        # Update fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update last_updated timestamp
        self.last_updated = datetime.now()
        
        # Increment conversation turns if not explicitly set
        if 'conversation_turns' not in kwargs:
            self.conversation_turns += 1
        
        return self

# Usage
state = AgentState(user_id="user123")
print("Initial state:")
print(f"Session ID: {state.session_id}")
print(f"User ID: {state.user_id}")
print(f"Conversation turns: {state.conversation_turns}")
print(f"Created at: {state.created_at}")
print(f"Last updated: {state.last_updated}")

# Update state
state.update(current_context={"topic": "weather"})
print("\nUpdated state:")
print(f"Conversation turns: {state.conversation_turns}")
print(f"Current context: {state.current_context}")
print(f"Last updated: {state.last_updated}")

# Try invalid update
try:
    state.update(conversation_turns=-1)
except ValueError as e:
    print(f"\nValidation error: {e}")
```

## 🔄 State Transition Validation

Ensuring state transitions follow valid paths:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Set
from enum import Enum
from datetime import datetime

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    PROVIDING_RESULTS = "providing_results"
    FOLLOW_UP = "follow_up"
    ENDING = "ending"

class StateTransitionValidator(BaseModel):
    allowed_transitions: Dict[str, Set[str]] = {
        ConversationState.GREETING.value: {
            ConversationState.COLLECTING_INFO.value,
            ConversationState.ENDING.value
        },
        ConversationState.COLLECTING_INFO.value: {
            ConversationState.COLLECTING_INFO.value,  # Can stay in this state
            ConversationState.PROCESSING.value
        },
        ConversationState.PROCESSING.value: {
            ConversationState.PROVIDING_RESULTS.value
        },
        ConversationState.PROVIDING_RESULTS.value: {
            ConversationState.FOLLOW_UP.value,
            ConversationState.ENDING.value
        },
        ConversationState.FOLLOW_UP.value: {
            ConversationState.COLLECTING_INFO.value,
            ConversationState.PROCESSING.value,
            ConversationState.ENDING.value
        },
        ConversationState.ENDING.value: {
            ConversationState.GREETING.value  # Can start a new conversation
        }
    }
    
    def validate_transition(self, current_state: str, new_state: str) -> bool:
        """Validate that a state transition is allowed."""
        if current_state not in self.allowed_transitions:
            raise ValueError(f"Unknown current state: {current_state}")
        
        if new_state not in self.allowed_transitions[current_state]:
            return False
        
        return True

class ConversationContext(BaseModel):
    state: str = ConversationState.GREETING.value
    state_history: List[str] = [ConversationState.GREETING.value]
    last_state_change: datetime = Field(default_factory=datetime.now)
    collected_info: Dict[str, Any] = {}
    missing_info: List[str] = []
    
    def transition_to(self, new_state: str, validator: StateTransitionValidator):
        """Transition to a new state if valid."""
        if not validator.validate_transition(self.state, new_state):
            raise ValueError(f"Invalid state transition from {self.state} to {new_state}")
        
        # Update state
        self.state = new_state
        self.state_history.append(new_state)
        self.last_state_change = datetime.now()
        
        return self

# Usage
validator = StateTransitionValidator()
context = ConversationContext()

print(f"Initial state: {context.state}")

# Valid transitions
try:
    context.transition_to(ConversationState.COLLECTING_INFO.value, validator)
    print(f"Transitioned to: {context.state}")
    
    context.transition_to(ConversationState.PROCESSING.value, validator)
    print(f"Transitioned to: {context.state}")
    
    context.transition_to(ConversationState.PROVIDING_RESULTS.value, validator)
    print(f"Transitioned to: {context.state}")
except ValueError as e:
    print(f"Transition error: {e}")

# Invalid transition
try:
    context.transition_to(ConversationState.COLLECTING_INFO.value, validator)
    print(f"Transitioned to: {context.state}")
except ValueError as e:
    print(f"Transition error: {e}")

print(f"State history: {context.state_history}")
```

## 🧠 Context-Aware State Validation

Validating state based on conversation context:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Set
from enum import Enum
from datetime import datetime

class RequiredInfoValidator(BaseModel):
    required_fields: Dict[str, List[str]] = {
        "weather_query": ["location"],
        "booking_query": ["service_type", "date"],
        "product_query": ["product_name"]
    }
    
    def validate_required_info(self, query_type: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required information is present for a query type."""
        result = {
            "is_complete": True,
            "missing_fields": []
        }
        
        if query_type not in self.required_fields:
            raise ValueError(f"Unknown query type: {query_type}")
        
        for field in self.required_fields[query_type]:
            if field not in collected_info or collected_info[field] is None:
                result["is_complete"] = False
                result["missing_fields"].append(field)
        
        return result

class ConversationState(BaseModel):
    query_type: Optional[str] = None
    collected_info: Dict[str, Any] = {}
    is_complete: bool = False
    missing_fields: List[str] = []
    
    def update_info(self, field: str, value: Any):
        """Update collected information."""
        self.collected_info[field] = value
        return self
    
    def validate_completeness(self, validator: RequiredInfoValidator):
        """Validate that all required information is collected."""
        if not self.query_type:
            raise ValueError("Query type not set")
        
        result = validator.validate_required_info(self.query_type, self.collected_info)
        self.is_complete = result["is_complete"]
        self.missing_fields = result["missing_fields"]
        
        return self

# Usage
validator = RequiredInfoValidator()
state = ConversationState(query_type="weather_query")

print(f"Initial state: {state.query_type}")
print(f"Is complete: {state.is_complete}")
print(f"Missing fields: {state.missing_fields}")

# Validate completeness
state.validate_completeness(validator)
print(f"\nAfter validation:")
print(f"Is complete: {state.is_complete}")
print(f"Missing fields: {state.missing_fields}")

# Add required information
state.update_info("location", "New York")
state.validate_completeness(validator)
print(f"\nAfter adding location:")
print(f"Is complete: {state.is_complete}")
print(f"Missing fields: {state.missing_fields}")

# Try a different query type
state = ConversationState(query_type="booking_query")
state.update_info("service_type", "haircut")
state.validate_completeness(validator)
print(f"\nBooking query with service_type:")
print(f"Is complete: {state.is_complete}")
print(f"Missing fields: {state.missing_fields}")
```

## 🔄 State Persistence and Recovery

Ensuring state can be saved, loaded, and recovered:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import json
import uuid

class PersistentState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    state_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = {}
    
    def save_to_json(self, file_path: str):
        """Save state to a JSON file."""
        # Update last_updated
        self.last_updated = datetime.now()
        
        # Convert to dict, handling datetime objects
        state_dict = self.model_dump()
        state_dict["created_at"] = state_dict["created_at"].isoformat()
        state_dict["last_updated"] = state_dict["last_updated"].isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, file_path: str):
        """Load state from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                state_dict = json.load(f)
            
            # Convert ISO format strings back to datetime
            state_dict["created_at"] = datetime.fromisoformat(state_dict["created_at"])
            state_dict["last_updated"] = datetime.fromisoformat(state_dict["last_updated"])
            
            return cls(**state_dict)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load state: {e}")
    
    @classmethod
    def recover_from_backup(cls, primary_path: str, backup_path: str):
        """Attempt to recover state from backup if primary is corrupted."""
        try:
            # Try to load from primary first
            return cls.load_from_json(primary_path)
        except ValueError:
            # If primary fails, try backup
            try:
                state = cls.load_from_json(backup_path)
                # Save recovered state to primary
                state.save_to_json(primary_path)
                return state
            except ValueError:
                # If both fail, create a new state
                return cls()

# Usage (simulated)
state = PersistentState(user_id="user123")
state.data = {"preferences": {"theme": "dark"}, "history": ["query1", "query2"]}

print("Original state:")
print(f"Session ID: {state.session_id}")
print(f"User ID: {state.user_id}")
print(f"Data: {state.data}")

# Simulate saving and loading
# state.save_to_json("state.json")
# loaded_state = PersistentState.load_from_json("state.json")

# Simulate recovery (in a real system, these would be actual file operations)
# recovered_state = PersistentState.recover_from_backup("corrupted.json", "backup.json")
```

## 🔄 Versioned State Management

Handling state evolution as your agent evolves:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union, Type
from datetime import datetime
import uuid

class StateV1(BaseModel):
    version: Literal["1.0"] = "1.0"
    user_id: str
    preferences: Dict[str, str] = {}
    history: List[str] = []

class StateV2(BaseModel):
    version: Literal["2.0"] = "2.0"
    user_id: str
    preferences: Dict[str, Any] = {}  # More flexible type
    history: List[Dict[str, Any]] = []  # Structured history
    last_active: Optional[datetime] = None

class StateManager:
    version_map: Dict[str, Type[BaseModel]] = {
        "1.0": StateV1,
        "2.0": StateV2
    }
    current_version: str = "2.0"
    
    @classmethod
    def migrate_v1_to_v2(cls, v1_state: StateV1) -> StateV2:
        """Migrate from v1 to v2 state format."""
        # Convert simple history to structured history
        structured_history = []
        for item in v1_state.history:
            structured_history.append({
                "query": item,
                "timestamp": datetime.now().isoformat()
            })
        
        # Create v2 state
        return StateV2(
            user_id=v1_state.user_id,
            preferences=v1_state.preferences,
            history=structured_history,
            last_active=datetime.now()
        )
    
    @classmethod
    def load_state(cls, state_data: Dict[str, Any]) -> Union[StateV1, StateV2]:
        """Load state from data, handling version differences."""
        version = state_data.get("version", "1.0")
        
        if version not in cls.version_map:
            raise ValueError(f"Unknown state version: {version}")
        
        # Create state object of appropriate version
        state_class = cls.version_map[version]
        state = state_class(**state_data)
        
        # If not current version, migrate
        if version != cls.current_version:
            if version == "1.0" and cls.current_version == "2.0":
                state = cls.migrate_v1_to_v2(state)
        
        return state

# Usage
v1_data = {
    "version": "1.0",
    "user_id": "user123",
    "preferences": {"theme": "light"},
    "history": ["weather in New York", "restaurants near me"]
}

state = StateManager.load_state(v1_data)
print(f"Loaded state version: {state.version}")
print(f"User ID: {state.user_id}")

if isinstance(state, StateV2):
    print(f"Last active: {state.last_active}")
    print(f"Structured history: {state.history}")
```

## 🧪 Exercises

1. Create a state management system for a multi-step form that validates each step before allowing progression.

2. Implement a conversation state machine that enforces valid transitions between different conversation phases.

3. Build a state recovery system that can detect and fix inconsistent states.

4. Create a versioned state system that can migrate between different schema versions.

5. Implement a state validation system that ensures all required information is collected before completing a task.

## 🔍 Key Takeaways

- State validation ensures consistency and prevents invalid transitions
- Pydantic models provide a strong foundation for state validation
- State transitions should follow well-defined paths
- Context-aware validation ensures all required information is collected
- State persistence and recovery mechanisms are essential for robustness
- Versioned state management helps handle evolution of your agent

## 📚 Additional Resources

- [Pydantic Validation Documentation](https://docs.pydantic.dev/latest/usage/validators/)
- [State Machine Patterns](https://refactoring.guru/design-patterns/state)
- [Data Persistence Strategies](https://martinfowler.com/eaaCatalog/repository.html)
- [Schema Migration Techniques](https://docs.sqlalchemy.org/en/14/core/metadata.html#alembic)

## 🚀 Next Steps

In the next lesson, we'll explore agent-specific validation patterns, focusing on domain-specific validation for different agent types and custom validators for agent-specific scenarios.
