"""
Exercise 4.4.5: Context-Aware Validation System

This exercise implements a validation system that uses conversation history
to resolve ambiguous references.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import re
from datetime import datetime
import uuid


class MessageRole(str, Enum):
    """Roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Model for a conversation message."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Entities extracted from the message")
    intent: Optional[Dict[str, Any]] = Field(None, description="Intent detected in the message")


class ConversationContext(BaseModel):
    """Model for conversation context."""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique context ID")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Accumulated entities from the conversation")
    intents: List[Dict[str, Any]] = Field(default_factory=list, description="Intents from the conversation")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session-specific data")
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        # Update accumulated entities with new entities from the message
        for entity_name, entity_value in message.entities.items():
            self.entities[entity_name] = entity_value
        
        # Add intent if present
        if message.intent:
            self.intents.append(message.intent)
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        Get the last n messages from the conversation.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of the last n messages
        """
        return self.messages[-n:] if n <= len(self.messages) else self.messages[:]
    
    def get_last_user_message(self) -> Optional[Message]:
        """
        Get the last message from the user.
        
        Returns:
            The last user message, or None if no user messages exist
        """
        for message in reversed(self.messages):
            if message.role == MessageRole.USER:
                return message
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """
        Get the last message from the assistant.
        
        Returns:
            The last assistant message, or None if no assistant messages exist
        """
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message
        return None
    
    def get_entity_history(self, entity_name: str) -> List[Any]:
        """
        Get the history of values for a specific entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of values for the entity across the conversation
        """
        values = []
        for message in self.messages:
            if entity_name in message.entities:
                values.append(message.entities[entity_name])
        return values
    
    def get_intent_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of intents in the conversation.
        
        Returns:
            List of intents
        """
        return [message.intent for message in self.messages if message.intent]
    
    def has_entity(self, entity_name: str) -> bool:
        """
        Check if an entity has been mentioned in the conversation.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            True if the entity exists in the conversation context
        """
        return entity_name in self.entities
    
    def get_entity(self, entity_name: str) -> Optional[Any]:
        """
        Get the current value of an entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Current value of the entity, or None if not found
        """
        return self.entities.get(entity_name)
    
    def set_user_preference(self, preference_name: str, preference_value: Any) -> None:
        """
        Set a user preference.
        
        Args:
            preference_name: Name of the preference
            preference_value: Value of the preference
        """
        self.user_preferences[preference_name] = preference_value
    
    def get_user_preference(self, preference_name: str) -> Optional[Any]:
        """
        Get a user preference.
        
        Args:
            preference_name: Name of the preference
            
        Returns:
            Value of the preference, or None if not found
        """
        return self.user_preferences.get(preference_name)


class ReferenceType(str, Enum):
    """Types of references in user input."""
    PRONOUN = "pronoun"
    DEMONSTRATIVE = "demonstrative"
    DEFINITE_ARTICLE = "definite_article"
    ELLIPSIS = "ellipsis"
    COMPARATIVE = "comparative"


class ResolvedReference(BaseModel):
    """Model for a resolved reference."""
    original_text: str = Field(..., description="Original reference text")
    reference_type: ReferenceType = Field(..., description="Type of reference")
    resolved_value: Any = Field(..., description="Resolved value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the resolution")
    source_message_id: Optional[str] = Field(None, description="ID of the message that contained the referent")


class ContextAwareValidator(BaseModel):
    """
    Validation system that uses conversation history to resolve ambiguous references.
    
    This system analyzes user input in the context of the conversation history
    to resolve pronouns, demonstratives, and other ambiguous references.
    """
    context: ConversationContext = Field(..., description="Conversation context")
    
    def resolve_references(self, text: str) -> Dict[str, ResolvedReference]:
        """
        Resolve references in text using conversation context.
        
        Args:
            text: Text containing references to resolve
            
        Returns:
            Dictionary mapping reference text to resolved references
        """
        resolved_references = {}
        
        # Detect and resolve pronouns
        pronoun_patterns = {
            r'\b(it|its)\b': self._resolve_it,
            r'\b(they|them|their)\b': self._resolve_they,
            r'\b(he|him|his)\b': self._resolve_he,
            r'\b(she|her|hers)\b': self._resolve_she,
            r'\b(this|that)\b': self._resolve_demonstrative,
            r'\b(these|those)\b': self._resolve_demonstrative_plural,
            r'\bthe (one|ones)\b': self._resolve_definite_article
        }
        
        for pattern, resolver in pronoun_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                reference_text = match.group(0)
                if reference_text not in resolved_references:
                    resolved = resolver(reference_text, text, match.start())
                    if resolved:
                        resolved_references[reference_text] = resolved
        
        return resolved_references
    
    def _resolve_it(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'it' or 'its' references.
        
        Args:
            reference_text: The reference text ('it' or 'its')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Look for the most recent entity in the conversation
        # Prioritize entities from the last assistant message
        last_assistant_msg = self.context.get_last_assistant_message()
        
        if last_assistant_msg and last_assistant_msg.entities:
            # Get the most likely entity (simple heuristic: first entity)
            entity_name, entity_value = next(iter(last_assistant_msg.entities.items()))
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.PRONOUN,
                resolved_value=entity_value,
                confidence=0.8,
                source_message_id=last_assistant_msg.message_id
            )
        
        # If no entity in the last assistant message, check the accumulated entities
        if self.context.entities:
            # Get the most recently mentioned entity
            last_user_msg = self.context.get_last_user_message()
            if last_user_msg and last_user_msg.entities:
                entity_name, entity_value = next(iter(last_user_msg.entities.items()))
                
                return ResolvedReference(
                    original_text=reference_text,
                    reference_type=ReferenceType.PRONOUN,
                    resolved_value=entity_value,
                    confidence=0.7,
                    source_message_id=last_user_msg.message_id
                )
            
            # Fallback to any entity in the context
            entity_name, entity_value = next(iter(self.context.entities.items()))
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.PRONOUN,
                resolved_value=entity_value,
                confidence=0.5,
                source_message_id=None
            )
        
        return None
    
    def _resolve_they(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'they', 'them', or 'their' references.
        
        Args:
            reference_text: The reference text ('they', 'them', or 'their')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Look for plural entities or lists in the conversation
        plural_entities = []
        
        # Check recent messages for plural entities
        recent_messages = self.context.get_last_n_messages(3)
        for message in reversed(recent_messages):
            for entity_name, entity_value in message.entities.items():
                if isinstance(entity_value, list) or (isinstance(entity_value, str) and ',' in entity_value):
                    plural_entities.append((entity_name, entity_value, message.message_id))
        
        if plural_entities:
            # Use the most recent plural entity
            entity_name, entity_value, message_id = plural_entities[0]
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.PRONOUN,
                resolved_value=entity_value,
                confidence=0.8,
                source_message_id=message_id
            )
        
        return None
    
    def _resolve_he(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'he', 'him', or 'his' references.
        
        Args:
            reference_text: The reference text ('he', 'him', or 'his')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Look for male person entities in the conversation
        person_entities = []
        
        # Check recent messages for person entities
        recent_messages = self.context.get_last_n_messages(3)
        for message in reversed(recent_messages):
            for entity_name, entity_value in message.entities.items():
                if entity_name == "person" or entity_name.endswith("_person"):
                    # Simple heuristic: assume male if name ends with common male name endings
                    if isinstance(entity_value, str) and any(entity_value.lower().endswith(suffix) for suffix in ["john", "david", "michael", "robert", "james"]):
                        person_entities.append((entity_name, entity_value, message.message_id))
        
        if person_entities:
            # Use the most recent male person entity
            entity_name, entity_value, message_id = person_entities[0]
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.PRONOUN,
                resolved_value=entity_value,
                confidence=0.7,
                source_message_id=message_id
            )
        
        return None
    
    def _resolve_she(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'she', 'her', or 'hers' references.
        
        Args:
            reference_text: The reference text ('she', 'her', or 'hers')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Look for female person entities in the conversation
        person_entities = []
        
        # Check recent messages for person entities
        recent_messages = self.context.get_last_n_messages(3)
        for message in reversed(recent_messages):
            for entity_name, entity_value in message.entities.items():
                if entity_name == "person" or entity_name.endswith("_person"):
                    # Simple heuristic: assume female if name ends with common female name endings
                    if isinstance(entity_value, str) and any(entity_value.lower().endswith(suffix) for suffix in ["mary", "jennifer", "linda", "patricia", "elizabeth"]):
                        person_entities.append((entity_name, entity_value, message.message_id))
        
        if person_entities:
            # Use the most recent female person entity
            entity_name, entity_value, message_id = person_entities[0]
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.PRONOUN,
                resolved_value=entity_value,
                confidence=0.7,
                source_message_id=message_id
            )
        
        return None
    
    def _resolve_demonstrative(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'this' or 'that' references.
        
        Args:
            reference_text: The reference text ('this' or 'that')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # For 'this', look for the most recent entity or concept
        # For 'that', look for the second most recent entity or concept
        
        recent_messages = self.context.get_last_n_messages(2)
        if not recent_messages:
            return None
        
        if reference_text.lower() == "this":
            # Look in the most recent message
            if recent_messages[-1].entities:
                entity_name, entity_value = next(iter(recent_messages[-1].entities.items()))
                
                return ResolvedReference(
                    original_text=reference_text,
                    reference_type=ReferenceType.DEMONSTRATIVE,
                    resolved_value=entity_value,
                    confidence=0.7,
                    source_message_id=recent_messages[-1].message_id
                )
        elif reference_text.lower() == "that" and len(recent_messages) > 1:
            # Look in the second most recent message
            if recent_messages[-2].entities:
                entity_name, entity_value = next(iter(recent_messages[-2].entities.items()))
                
                return ResolvedReference(
                    original_text=reference_text,
                    reference_type=ReferenceType.DEMONSTRATIVE,
                    resolved_value=entity_value,
                    confidence=0.6,
                    source_message_id=recent_messages[-2].message_id
                )
        
        # Fallback: look for any entity in the context
        if self.context.entities:
            entity_name, entity_value = next(iter(self.context.entities.items()))
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.DEMONSTRATIVE,
                resolved_value=entity_value,
                confidence=0.4,
                source_message_id=None
            )
        
        return None
    
    def _resolve_demonstrative_plural(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'these' or 'those' references.
        
        Args:
            reference_text: The reference text ('these' or 'those')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Similar to _resolve_demonstrative but looking for plural entities
        plural_entities = []
        
        # Check recent messages for plural entities
        recent_messages = self.context.get_last_n_messages(3)
        for message in reversed(recent_messages):
            for entity_name, entity_value in message.entities.items():
                if isinstance(entity_value, list) or (isinstance(entity_value, str) and ',' in entity_value):
                    plural_entities.append((entity_name, entity_value, message.message_id))
        
        if plural_entities:
            # Use the most recent plural entity
            entity_name, entity_value, message_id = plural_entities[0]
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.DEMONSTRATIVE,
                resolved_value=entity_value,
                confidence=0.7,
                source_message_id=message_id
            )
        
        return None
    
    def _resolve_definite_article(self, reference_text: str, full_text: str, position: int) -> Optional[ResolvedReference]:
        """
        Resolve 'the one' or 'the ones' references.
        
        Args:
            reference_text: The reference text ('the one' or 'the ones')
            full_text: The full text containing the reference
            position: Position of the reference in the text
            
        Returns:
            Resolved reference, or None if resolution failed
        """
        # Look for entities in the last assistant message
        last_assistant_msg = self.context.get_last_assistant_message()
        
        if last_assistant_msg and last_assistant_msg.entities:
            # Get the most likely entity
            entity_name, entity_value = next(iter(last_assistant_msg.entities.items()))
            
            return ResolvedReference(
                original_text=reference_text,
                reference_type=ReferenceType.DEFINITE_ARTICLE,
                resolved_value=entity_value,
                confidence=0.6,
                source_message_id=last_assistant_msg.message_id
            )
        
        return None
    
    def validate_with_context(self, text: str, required_entities: List[str]) -> Dict[str, Any]:
        """
        Validate user input using conversation context.
        
        Args:
            text: User input text
            required_entities: List of required entity names
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": True,
            "missing_entities": [],
            "resolved_entities": {},
            "ambiguous_references": [],
            "confidence": 1.0
        }
        
        # First, resolve any references in the text
        resolved_references = self.resolve_references(text)
        
        # Check for each required entity
        for entity_name in required_entities:
            # Check if the entity is directly mentioned in the text
            # (In a real system, this would use an entity extraction model)
            entity_directly_mentioned = False
            
            # If not directly mentioned, check if it can be resolved from context
            if not entity_directly_mentioned:
                if self.context.has_entity(entity_name):
                    # Entity exists in context
                    entity_value = self.context.get_entity(entity_name)
                    result["resolved_entities"][entity_name] = entity_value
                else:
                    # Entity is missing and can't be resolved from context
                    result["is_valid"] = False
                    result["missing_entities"].append(entity_name)
        
        # Adjust confidence based on the number of resolved references
        if resolved_references:
            # Average confidence of resolved references
            avg_confidence = sum(ref.confidence for ref in resolved_references.values()) / len(resolved_references)
            result["confidence"] = min(result["confidence"], avg_confidence)
        
        # If there are unresolved references, mark them as ambiguous
        for reference_text, reference in resolved_references.items():
            if reference.confidence < 0.5:
                result["ambiguous_references"].append(reference_text)
        
        return result
    
    def extract_entities_with_context(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using conversation context.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # In a real system, this would use a proper entity extraction model
        # Here we'll use simple pattern matching for demonstration
        
        # Extract location entities
        location_match = re.search(r'(?:in|at|to) ([A-Z][a-z]+ ?[A-Z]?[a-z]*)', text)
        if location_match:
            entities["location"] = location_match.group(1)
        
        # Extract date entities
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
        if date_match:
            entities["date"] = date_match.group(1)
        
        # Extract time entities
        time_match = re.search(r'(\d{1,2}:\d{2}(?: [AP]M)?)', text)
        if time_match:
            entities["time"] = time_match.group(1)
        
        # Resolve references to fill in missing entities
        resolved_references = self.resolve_references(text)
        
        # For each resolved reference, try to determine which entity it refers to
        for reference_text, reference in resolved_references.items():
            # Simple heuristic: if the reference is used in a context that suggests
            # a specific entity type, use that entity type
            
            # Location context
            if re.search(r'(?:in|at|to) ' + re.escape(reference_text), text, re.IGNORECASE):
                if "location" not in entities and isinstance(reference.resolved_value, str):
                    entities["location"] = reference.resolved_value
            
            # Date context
            elif re.search(r'(?:on|for) ' + re.escape(reference_text), text, re.IGNORECASE):
                if "date" not in entities:
                    entities["date"] = reference.resolved_value
            
            # Time context
            elif re.search(r'(?:at|by) ' + re.escape(reference_text), text, re.IGNORECASE):
                if "time" not in entities:
                    entities["time"] = reference.resolved_value
        
        return entities


# Example usage
if __name__ == "__main__":
    # Create a conversation context
    context = ConversationContext()
    
    # Add some messages to the conversation
    context.add_message(
        Message(
            role=MessageRole.USER,
            content="What's the weather like in New York?",
            entities={"location": "New York"},
            intent={"type": "weather", "confidence": 0.9}
        )
    )
    
    context.add_message(
        Message(
            role=MessageRole.ASSISTANT,
            content="It's currently sunny and 75°F in New York.",
            entities={"location": "New York", "weather": "sunny", "temperature": "75°F"}
        )
    )
    
    # Create a context-aware validator
    validator = ContextAwareValidator(context=context)
    
    # Example 1: Resolve a pronoun reference
    text1 = "Is it going to rain there tomorrow?"
    resolved1 = validator.resolve_references(text1)
    
    print("Example 1: Resolving 'there' in 'Is it going to rain there tomorrow?'")
    for reference_text, reference in resolved1.items():
        print(f"Reference: '{reference_text}'")
        print(f"Type: {reference.reference_type}")
        print(f"Resolved value: {reference.resolved_value}")
        print(f"Confidence: {reference.confidence}")
        print()
    
    # Example 2: Validate with required entities
    result2 = validator.validate_with_context(text1, required_entities=["location", "date"])
    
    print("Example 2: Validating 'Is it going to rain there tomorrow?' with required entities")
    print(f"Is valid: {result2['is_valid']}")
    print(f"Missing entities: {result2['missing_entities']}")
    print(f"Resolved entities: {result2['resolved_entities']}")
    print(f"Ambiguous references: {result2['ambiguous_references']}")
    print(f"Confidence: {result2['confidence']}")
    print()
    
    # Example 3: Extract entities with context
    entities3 = validator.extract_entities_with_context(text1)
    
    print("Example 3: Extracting entities from 'Is it going to rain there tomorrow?'")
    print(f"Extracted entities: {entities3}")
    print()
    
    # Add another message to the conversation
    context.add_message(
        Message(
            role=MessageRole.USER,
            content="Is it going to rain there tomorrow?",
            entities={"location": "New York", "date": "tomorrow"},
            intent={"type": "weather", "confidence": 0.9}
        )
    )
    
    context.add_message(
        Message(
            role=MessageRole.ASSISTANT,
            content="Yes, there's a 30% chance of rain in New York tomorrow.",
            entities={"location": "New York", "date": "tomorrow", "rain_chance": "30%"}
        )
    )
    
    # Example 4: Resolve a more complex reference
    text4 = "What about the day after that?"
    resolved4 = validator.resolve_references(text4)
    
    print("Example 4: Resolving 'that' in 'What about the day after that?'")
    for reference_text, reference in resolved4.items():
        print(f"Reference: '{reference_text}'")
        print(f"Type: {reference.reference_type}")
        print(f"Resolved value: {reference.resolved_value}")
        print(f"Confidence: {reference.confidence}")
        print()
