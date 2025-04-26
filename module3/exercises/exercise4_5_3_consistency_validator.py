"""
Exercise 4.5.3: Consistency Validator

This exercise implements a validator that ensures agent responses don't contradict
information provided in previous interactions.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import re
from datetime import datetime, timedelta


class FactType(str, Enum):
    """Types of facts that can be stored in conversation context."""
    ATTRIBUTE = "attribute"  # Properties of entities (color, size, etc.)
    RELATION = "relation"    # Relationships between entities
    STATE = "state"          # Current state of an entity
    LOCATION = "location"    # Where an entity is located
    TEMPORAL = "temporal"    # Time-related information
    NUMERICAL = "numerical"  # Numerical values
    PREFERENCE = "preference"  # User preferences
    CUSTOM = "custom"        # Custom fact type


class FactSource(str, Enum):
    """Sources of facts in the conversation."""
    USER = "user"            # Provided by the user
    AGENT = "agent"          # Provided by the agent
    SYSTEM = "system"        # Provided by the system
    EXTERNAL = "external"    # From external data source
    INFERRED = "inferred"    # Inferred from other facts


class InconsistencyType(str, Enum):
    """Types of inconsistencies that can be detected."""
    CONTRADICTION = "contradiction"  # Direct contradiction of a fact
    NUMERICAL = "numerical"          # Numerical inconsistency
    TEMPORAL = "temporal"            # Temporal inconsistency
    ENTITY = "entity"                # Entity reference inconsistency
    LOGICAL = "logical"              # Logical inconsistency
    ATTRIBUTE = "attribute"          # Attribute inconsistency


class ContextualFact(BaseModel):
    """A fact with metadata about its source, confidence, and timestamp."""
    key: str = Field(..., description="Unique identifier for the fact")
    value: Any = Field(..., description="Value of the fact")
    fact_type: FactType = Field(..., description="Type of fact")
    source: FactSource = Field(..., description="Source of the fact")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the fact was recorded")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the fact (0-1)")
    expiration: Optional[datetime] = Field(None, description="When the fact expires (if applicable)")
    related_entities: List[str] = Field(default_factory=list, description="Entities related to this fact")
    
    @property
    def is_expired(self) -> bool:
        """Check if the fact has expired."""
        if self.expiration is None:
            return False
        return datetime.now() > self.expiration
    
    @property
    def age(self) -> timedelta:
        """Get the age of the fact."""
        return datetime.now() - self.timestamp


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    text: str = Field(..., description="Message text")
    sender: str = Field(..., description="Message sender (user or agent)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")
    extracted_facts: List[ContextualFact] = Field(default_factory=list, description="Facts extracted from the message")


class ConversationContext(BaseModel):
    """Context for a conversation, including facts and entities."""
    facts: Dict[str, ContextualFact] = Field(default_factory=dict, description="Facts known in the conversation")
    entities: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Entities and their attributes")
    
    def add_fact(self, key: str, value: Any, fact_type: FactType, source: FactSource, 
                confidence: float = 1.0, expiration: Optional[datetime] = None,
                related_entities: Optional[List[str]] = None) -> ContextualFact:
        """
        Add a fact to the context.
        
        Args:
            key: Unique identifier for the fact
            value: Value of the fact
            fact_type: Type of fact
            source: Source of the fact
            confidence: Confidence in the fact (0-1)
            expiration: When the fact expires (if applicable)
            related_entities: Entities related to this fact
            
        Returns:
            The created ContextualFact
        """
        fact = ContextualFact(
            key=key,
            value=value,
            fact_type=fact_type,
            source=source,
            confidence=confidence,
            expiration=expiration,
            related_entities=related_entities or []
        )
        self.facts[key] = fact
        
        # If this fact is about an entity, update the entity record
        for entity in fact.related_entities:
            if entity not in self.entities:
                self.entities[entity] = {}
            
            # For attribute facts, store the attribute with the entity
            if fact_type == FactType.ATTRIBUTE:
                attribute_name = key.split('.')[-1] if '.' in key else key
                self.entities[entity][attribute_name] = value
        
        return fact
    
    def get_fact(self, key: str) -> Optional[ContextualFact]:
        """Get a fact from the context."""
        return self.facts.get(key)
    
    def get_facts_by_type(self, fact_type: FactType) -> List[ContextualFact]:
        """Get all facts of a specific type."""
        return [fact for fact in self.facts.values() if fact.fact_type == fact_type]
    
    def get_facts_by_entity(self, entity: str) -> List[ContextualFact]:
        """Get all facts related to a specific entity."""
        return [fact for fact in self.facts.values() if entity in fact.related_entities]
    
    def get_entity_attributes(self, entity: str) -> Dict[str, Any]:
        """Get all attributes of a specific entity."""
        return self.entities.get(entity, {})
    
    def get_contradicting_facts(self, fact: ContextualFact) -> List[ContextualFact]:
        """Find facts that contradict the given fact."""
        contradictions = []
        
        # For simple value contradictions
        if fact.key in self.facts and self.facts[fact.key].value != fact.value:
            contradictions.append(self.facts[fact.key])
        
        # For numerical facts, check for numerical inconsistencies
        if fact.fact_type == FactType.NUMERICAL:
            for existing_fact in self.facts.values():
                if existing_fact.fact_type == FactType.NUMERICAL and existing_fact.key != fact.key:
                    # Check if they refer to the same thing but with different values
                    if set(existing_fact.related_entities) == set(fact.related_entities) and existing_fact.value != fact.value:
                        contradictions.append(existing_fact)
        
        return contradictions


class ConversationHistory(BaseModel):
    """History of a conversation, including messages and context."""
    messages: List[ConversationMessage] = Field(default_factory=list, description="Messages in the conversation")
    context: ConversationContext = Field(default_factory=ConversationContext, description="Conversation context")
    
    def add_message(self, text: str, sender: str, extracted_facts: Optional[List[ContextualFact]] = None) -> ConversationMessage:
        """
        Add a message to the conversation history.
        
        Args:
            text: Message text
            sender: Message sender (user or agent)
            extracted_facts: Facts extracted from the message
            
        Returns:
            The created ConversationMessage
        """
        message = ConversationMessage(
            text=text,
            sender=sender,
            extracted_facts=extracted_facts or []
        )
        self.messages.append(message)
        
        # Add extracted facts to context
        for fact in message.extracted_facts:
            self.context.facts[fact.key] = fact
            
        return message
    
    def get_last_n_messages(self, n: int) -> List[ConversationMessage]:
        """Get the last n messages in the conversation."""
        return self.messages[-n:] if n > 0 else []
    
    def get_messages_by_sender(self, sender: str) -> List[ConversationMessage]:
        """Get all messages from a specific sender."""
        return [msg for msg in self.messages if msg.sender == sender]


class InconsistencyReport(BaseModel):
    """Report of an inconsistency detected in a response."""
    inconsistency_type: InconsistencyType = Field(..., description="Type of inconsistency")
    description: str = Field(..., description="Description of the inconsistency")
    conflicting_facts: List[ContextualFact] = Field(default_factory=list, description="Facts that conflict with the response")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the inconsistency detection")
    snippet: Optional[str] = Field(None, description="Snippet from the response containing the inconsistency")
    suggestion: Optional[str] = Field(None, description="Suggestion for resolving the inconsistency")


class ConsistencyValidator(BaseModel):
    """
    Validator that ensures agent responses don't contradict information
    provided in previous interactions.
    """
    response: str = Field(..., description="Agent response to validate")
    context: ConversationContext = Field(..., description="Conversation context")
    history: Optional[ConversationHistory] = Field(None, description="Conversation history")
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate that the response is consistent with known facts."""
        inconsistencies = self.check_for_inconsistencies()
        
        if inconsistencies:
            descriptions = [inc.description for inc in inconsistencies]
            raise ValueError(f"Response contains inconsistencies: {'; '.join(descriptions)}")
        
        return self
    
    def check_for_inconsistencies(self) -> List[InconsistencyReport]:
        """
        Check for inconsistencies in the response.
        
        Returns:
            List of inconsistency reports
        """
        inconsistencies = []
        
        # Check for basic fact contradictions
        inconsistencies.extend(self._check_fact_contradictions())
        
        # Check for temporal inconsistencies
        inconsistencies.extend(self._check_temporal_consistency())
        
        # Check for numerical inconsistencies
        inconsistencies.extend(self._check_numerical_consistency())
        
        # Check for entity reference inconsistencies
        inconsistencies.extend(self._check_entity_consistency())
        
        return inconsistencies
    
    def _check_fact_contradictions(self) -> List[InconsistencyReport]:
        """Check for direct contradictions of facts in the context."""
        inconsistencies = []
        
        # Check for location inconsistencies
        location_facts = self.context.get_facts_by_type(FactType.LOCATION)
        for fact in location_facts:
            entity = fact.related_entities[0] if fact.related_entities else None
            if entity:
                # Look for mentions of the entity in a different location
                entity_pattern = rf"\b{re.escape(entity)}\b.*?\bin\s+([A-Za-z ]+)(?!\s*{re.escape(str(fact.value))})"
                location_match = re.search(entity_pattern, self.response, re.IGNORECASE)
                if location_match and location_match.group(1).lower() != str(fact.value).lower():
                    inconsistencies.append(InconsistencyReport(
                        inconsistency_type=InconsistencyType.CONTRADICTION,
                        description=f"Response mentions {entity} in {location_match.group(1)} but it was previously established to be in {fact.value}",
                        conflicting_facts=[fact],
                        snippet=location_match.group(0)
                    ))
        
        # Check for attribute inconsistencies
        attribute_facts = self.context.get_facts_by_type(FactType.ATTRIBUTE)
        for fact in attribute_facts:
            entity = fact.related_entities[0] if fact.related_entities else None
            if entity:
                attribute_name = fact.key.split('.')[-1] if '.' in fact.key else fact.key
                # Look for mentions of the entity with a different attribute value
                attribute_pattern = rf"\b{re.escape(entity)}\b.*?\b{re.escape(attribute_name)}\b.*?\b([A-Za-z0-9 ]+)(?!\s*{re.escape(str(fact.value))})"
                attribute_match = re.search(attribute_pattern, self.response, re.IGNORECASE)
                if attribute_match and attribute_match.group(1).lower() != str(fact.value).lower():
                    inconsistencies.append(InconsistencyReport(
                        inconsistency_type=InconsistencyType.ATTRIBUTE,
                        description=f"Response mentions {entity} with {attribute_name} {attribute_match.group(1)} but it was previously established as {fact.value}",
                        conflicting_facts=[fact],
                        snippet=attribute_match.group(0)
                    ))
        
        # Check for state inconsistencies
        state_facts = self.context.get_facts_by_type(FactType.STATE)
        for fact in state_facts:
            entity = fact.related_entities[0] if fact.related_entities else None
            if entity:
                # Look for mentions of the entity in a different state
                state_pattern = rf"\b{re.escape(entity)}\b.*?\bis\s+([A-Za-z ]+)(?!\s*{re.escape(str(fact.value))})"
                state_match = re.search(state_pattern, self.response, re.IGNORECASE)
                if state_match and state_match.group(1).lower() != str(fact.value).lower():
                    inconsistencies.append(InconsistencyReport(
                        inconsistency_type=InconsistencyType.CONTRADICTION,
                        description=f"Response mentions {entity} is {state_match.group(1)} but it was previously established as {fact.value}",
                        conflicting_facts=[fact],
                        snippet=state_match.group(0)
                    ))
        
        return inconsistencies
    
    def _check_temporal_consistency(self) -> List[InconsistencyReport]:
        """Check for temporal inconsistencies in the response."""
        inconsistencies = []
        
        # Check for temporal facts
        temporal_facts = self.context.get_facts_by_type(FactType.TEMPORAL)
        for fact in temporal_facts:
            entity = fact.related_entities[0] if fact.related_entities else None
            if entity:
                # Look for mentions of the entity with a different time
                time_pattern = rf"\b{re.escape(entity)}\b.*?\b(yesterday|today|tomorrow|last week|next week|last month|next month)\b"
                time_match = re.search(time_pattern, self.response, re.IGNORECASE)
                if time_match and time_match.group(1).lower() != str(fact.value).lower():
                    inconsistencies.append(InconsistencyReport(
                        inconsistency_type=InconsistencyType.TEMPORAL,
                        description=f"Response mentions {entity} with temporal reference {time_match.group(1)} but it was previously established as {fact.value}",
                        conflicting_facts=[fact],
                        snippet=time_match.group(0)
                    ))
        
        return inconsistencies
    
    def _check_numerical_consistency(self) -> List[InconsistencyReport]:
        """Check for numerical inconsistencies in the response."""
        inconsistencies = []
        
        # Check for numerical facts
        numerical_facts = self.context.get_facts_by_type(FactType.NUMERICAL)
        for fact in numerical_facts:
            entity = fact.related_entities[0] if fact.related_entities else None
            if entity and isinstance(fact.value, (int, float)):
                # Look for mentions of the entity with a different numerical value
                number_pattern = rf"\b{re.escape(entity)}\b.*?\b(\d+(?:\.\d+)?)\b"
                number_matches = re.finditer(number_pattern, self.response, re.IGNORECASE)
                
                for match in number_matches:
                    try:
                        response_value = float(match.group(1))
                        # Allow for small differences in floating point values
                        if isinstance(fact.value, float) and abs(response_value - fact.value) < 0.001:
                            continue
                            
                        if response_value != fact.value:
                            inconsistencies.append(InconsistencyReport(
                                inconsistency_type=InconsistencyType.NUMERICAL,
                                description=f"Response mentions {entity} with value {response_value} but it was previously established as {fact.value}",
                                conflicting_facts=[fact],
                                snippet=match.group(0)
                            ))
                    except ValueError:
                        # Not a valid number
                        pass
        
        return inconsistencies
    
    def _check_entity_consistency(self) -> List[InconsistencyReport]:
        """Check for entity reference inconsistencies in the response."""
        inconsistencies = []
        
        # Check for entity references
        for entity, attributes in self.context.entities.items():
            if entity in self.response:
                # Check if any attributes are mentioned with inconsistent values
                for attr_name, attr_value in attributes.items():
                    attr_pattern = rf"\b{re.escape(entity)}\b.*?\b{re.escape(attr_name)}\b.*?\b([A-Za-z0-9 ]+)\b"
                    attr_match = re.search(attr_pattern, self.response, re.IGNORECASE)
                    if attr_match and attr_match.group(1).lower() != str(attr_value).lower():
                        # Find the original fact
                        fact = None
                        for f in self.context.facts.values():
                            if entity in f.related_entities and f.fact_type == FactType.ATTRIBUTE:
                                fact_attr_name = f.key.split('.')[-1] if '.' in f.key else f.key
                                if fact_attr_name == attr_name:
                                    fact = f
                                    break
                        
                        if fact:
                            inconsistencies.append(InconsistencyReport(
                                inconsistency_type=InconsistencyType.ENTITY,
                                description=f"Response mentions {entity} with {attr_name} {attr_match.group(1)} but it was previously established as {attr_value}",
                                conflicting_facts=[fact],
                                snippet=attr_match.group(0)
                            ))
        
        return inconsistencies
    
    def extract_potential_facts(self) -> List[ContextualFact]:
        """
        Extract potential facts from the response.
        
        Returns:
            List of potential facts extracted from the response
        """
        facts = []
        
        # Extract location facts
        location_pattern = r"\b([A-Za-z ]+)\s+is\s+in\s+([A-Za-z ]+)\b"
        for match in re.finditer(location_pattern, self.response, re.IGNORECASE):
            entity = match.group(1).strip()
            location = match.group(2).strip()
            
            facts.append(ContextualFact(
                key=f"location.{entity}",
                value=location,
                fact_type=FactType.LOCATION,
                source=FactSource.AGENT,
                related_entities=[entity]
            ))
        
        # Extract attribute facts
        attribute_pattern = r"\b([A-Za-z ]+)\s+is\s+([A-Za-z ]+)\b"
        for match in re.finditer(attribute_pattern, self.response, re.IGNORECASE):
            entity = match.group(1).strip()
            attribute = match.group(2).strip()
            
            # Skip common verbs and stop words
            stop_words = ["a", "an", "the", "this", "that", "these", "those", "it", "they", "there"]
            if attribute.lower() in stop_words:
                continue
                
            facts.append(ContextualFact(
                key=f"attribute.{entity}",
                value=attribute,
                fact_type=FactType.ATTRIBUTE,
                source=FactSource.AGENT,
                related_entities=[entity]
            ))
        
        # Extract numerical facts
        numerical_pattern = r"\b([A-Za-z ]+)\s+is\s+(\d+(?:\.\d+)?)\b"
        for match in re.finditer(numerical_pattern, self.response, re.IGNORECASE):
            entity = match.group(1).strip()
            try:
                value = float(match.group(2))
                # Convert to int if it's a whole number
                if value.is_integer():
                    value = int(value)
                    
                facts.append(ContextualFact(
                    key=f"numerical.{entity}",
                    value=value,
                    fact_type=FactType.NUMERICAL,
                    source=FactSource.AGENT,
                    related_entities=[entity]
                ))
            except ValueError:
                # Not a valid number
                pass
        
        return facts


# Example usage
if __name__ == "__main__":
    # Create a conversation context
    context = ConversationContext()
    
    # Add some facts to the context
    context.add_fact(
        key="location.weather",
        value="New York",
        fact_type=FactType.LOCATION,
        source=FactSource.USER,
        related_entities=["weather"]
    )
    
    context.add_fact(
        key="attribute.weather",
        value="sunny",
        fact_type=FactType.STATE,
        source=FactSource.SYSTEM,
        related_entities=["weather"]
    )
    
    context.add_fact(
        key="numerical.temperature",
        value=75,
        fact_type=FactType.NUMERICAL,
        source=FactSource.SYSTEM,
        related_entities=["temperature"]
    )
    
    context.add_fact(
        key="temporal.meeting",
        value="tomorrow",
        fact_type=FactType.TEMPORAL,
        source=FactSource.USER,
        related_entities=["meeting"]
    )
    
    # Example 1: Consistent response
    try:
        consistent_response = "The weather in New York is sunny with a temperature of 75 degrees. Don't forget your meeting tomorrow."
        validator = ConsistencyValidator(
            response=consistent_response,
            context=context
        )
        print("Example 1: Consistent response")
        print(f"Response: '{consistent_response}'")
        print("Result: Consistent with context")
        print()
    except ValueError as e:
        print(f"Consistency error: {e}")
        print()
    
    # Example 2: Inconsistent location
    try:
        inconsistent_location = "The weather in Boston is sunny with a temperature of 75 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_location,
            context=context
        )
        print("Example 2: Inconsistent location")
        print(f"Response: '{inconsistent_location}'")
        print("Result: Consistent with context")
        print()
    except ValueError as e:
        print(f"Consistency error: {e}")
        print()
    
    # Example 3: Inconsistent weather condition
    try:
        inconsistent_weather = "The weather in New York is rainy with a temperature of 75 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_weather,
            context=context
        )
        print("Example 3: Inconsistent weather condition")
        print(f"Response: '{inconsistent_weather}'")
        print("Result: Consistent with context")
        print()
    except ValueError as e:
        print(f"Consistency error: {e}")
        print()
    
    # Example 4: Inconsistent temperature
    try:
        inconsistent_temperature = "The weather in New York is sunny with a temperature of 60 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_temperature,
            context=context
        )
        print("Example 4: Inconsistent temperature")
        print(f"Response: '{inconsistent_temperature}'")
        print("Result: Consistent with context")
        print()
    except ValueError as e:
        print(f"Consistency error: {e}")
        print()
    
    # Example 5: Inconsistent temporal reference
    try:
        inconsistent_temporal = "The weather in New York is sunny with a temperature of 75 degrees. Don't forget your meeting today."
        validator = ConsistencyValidator(
            response=inconsistent_temporal,
            context=context
        )
        print("Example 5: Inconsistent temporal reference")
        print(f"Response: '{inconsistent_temporal}'")
        print("Result: Consistent with context")
        print()
    except ValueError as e:
        print(f"Consistency error: {e}")
        print()
    
    # Example 6: Extract potential facts
    response_with_facts = "The sky is blue. The grass is green. The temperature is 75 degrees. New York is in the United States."
    validator = ConsistencyValidator(
        response=response_with_facts,
        context=context
    )
    
    extracted_facts = validator.extract_potential_facts()
    
    print("Example 6: Extract potential facts")
    print(f"Response: '{response_with_facts}'")
    print("Extracted facts:")
    for fact in extracted_facts:
        print(f"  - Key: {fact.key}")
        print(f"    Value: {fact.value}")
        print(f"    Type: {fact.fact_type}")
        print(f"    Related entities: {fact.related_entities}")
        print()
