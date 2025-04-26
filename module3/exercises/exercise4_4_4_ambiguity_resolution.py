"""
Exercise 4.4.4: Ambiguity Resolution System

This exercise implements a system that generates appropriate clarification questions
based on the type of ambiguity in user requests.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import re
from datetime import datetime


class AmbiguityType(str, Enum):
    """Types of ambiguity in user requests."""
    MISSING_ENTITY = "missing_entity"
    MULTIPLE_INTENTS = "multiple_intents"
    VAGUE_REFERENCE = "vague_reference"
    UNDERSPECIFIED = "underspecified"
    CONFLICTING_INFO = "conflicting_info"
    HOMONYM = "homonym"


class Intent(BaseModel):
    """Model for a detected intent."""
    type: str = Field(..., description="Type of intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Entities associated with this intent")


class Entity(BaseModel):
    """Model for a detected entity."""
    type: str = Field(..., description="Type of entity")
    value: Any = Field(..., description="Value of the entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_pos: Optional[int] = Field(None, description="Start position in text")
    end_pos: Optional[int] = Field(None, description="End position in text")


class PossibleInterpretation(BaseModel):
    """Model for a possible interpretation of an ambiguous request."""
    intent: Intent = Field(..., description="The intent for this interpretation")
    missing_entities: List[str] = Field(default_factory=list, description="Entities missing for this interpretation")
    clarification_question: str = Field(..., description="Question to ask to confirm this interpretation")


class AmbiguityResolution(BaseModel):
    """
    System for detecting and resolving ambiguities in user requests.
    
    This system analyzes user input for ambiguities, determines the type of ambiguity,
    and generates appropriate clarification questions.
    """
    original_text: str = Field(..., description="Original user input text")
    is_ambiguous: bool = Field(..., description="Whether the input is ambiguous")
    ambiguity_type: Optional[AmbiguityType] = Field(None, description="Type of ambiguity if ambiguous")
    clarification_question: Optional[str] = Field(None, description="Question to ask to resolve the ambiguity")
    possible_interpretations: List[PossibleInterpretation] = Field(
        default_factory=list,
        description="Possible interpretations of the ambiguous input"
    )
    
    @model_validator(mode='after')
    def validate_ambiguity_fields(self):
        """Validate that ambiguity fields are consistent."""
        if self.is_ambiguous:
            if not self.ambiguity_type:
                raise ValueError("Ambiguity type must be provided for ambiguous inputs")
            if not self.clarification_question:
                raise ValueError("Clarification question must be provided for ambiguous inputs")
        else:
            if self.ambiguity_type:
                raise ValueError("Ambiguity type should not be provided for non-ambiguous inputs")
            if self.clarification_question:
                raise ValueError("Clarification question should not be provided for non-ambiguous inputs")
            if self.possible_interpretations:
                raise ValueError("Possible interpretations should not be provided for non-ambiguous inputs")
        
        return self
    
    @classmethod
    def analyze(cls, text: str, entities: Dict[str, Entity], intents: List[Intent]) -> 'AmbiguityResolution':
        """
        Analyze input for ambiguities.
        
        Args:
            text: Original user input text
            entities: Dictionary of detected entities
            intents: List of detected intents
            
        Returns:
            AmbiguityResolution instance with analysis results
        """
        is_ambiguous = False
        ambiguity_type = None
        clarification_question = None
        possible_interpretations = []
        
        # Check for multiple intents with similar confidence
        if len(intents) > 1:
            # If top two intents have similar confidence (within 0.2), consider it ambiguous
            confidence_diff = intents[0].confidence - intents[1].confidence
            if confidence_diff < 0.2:
                is_ambiguous = True
                ambiguity_type = AmbiguityType.MULTIPLE_INTENTS
                
                # Generate clarification question based on the top intents
                intent_types = [intent.type for intent in intents[:2]]
                clarification_question = f"I'm not sure if you want to {intent_types[0]} or {intent_types[1]}. Can you clarify?"
                
                # Add possible interpretations
                for intent in intents[:2]:
                    missing = []
                    question = ""
                    
                    if intent.type == "weather":
                        if "location" not in entities:
                            missing.append("location")
                        question = "Did you want to check the weather? If so, for which location?"
                    elif intent.type == "booking":
                        if "service_type" not in entities:
                            missing.append("service_type")
                        if "date" not in entities:
                            missing.append("date")
                        question = "Did you want to make a booking? If so, what type of booking and for when?"
                    
                    possible_interpretations.append(
                        PossibleInterpretation(
                            intent=intent,
                            missing_entities=missing,
                            clarification_question=question
                        )
                    )
        
        # Check for missing required entities for the top intent
        elif intents and not is_ambiguous:
            top_intent = intents[0]
            
            if top_intent.type == "weather" and "location" not in entities:
                is_ambiguous = True
                ambiguity_type = AmbiguityType.MISSING_ENTITY
                clarification_question = "Which location would you like the weather for?"
                
                possible_interpretations.append(
                    PossibleInterpretation(
                        intent=top_intent,
                        missing_entities=["location"],
                        clarification_question="Did you want to check the weather? If so, for which location?"
                    )
                )
            
            elif top_intent.type == "booking":
                missing_entities = []
                
                if "service_type" not in entities:
                    missing_entities.append("service_type")
                
                if "date" not in entities:
                    missing_entities.append("date")
                
                if missing_entities:
                    is_ambiguous = True
                    ambiguity_type = AmbiguityType.MISSING_ENTITY
                    
                    if "service_type" in missing_entities and "date" in missing_entities:
                        clarification_question = "What type of booking would you like to make and for when?"
                    elif "service_type" in missing_entities:
                        clarification_question = "What type of booking would you like to make?"
                    else:  # date is missing
                        clarification_question = "When would you like to make this booking?"
                    
                    possible_interpretations.append(
                        PossibleInterpretation(
                            intent=top_intent,
                            missing_entities=missing_entities,
                            clarification_question=f"Did you want to make a booking? I need more information about: {', '.join(missing_entities)}"
                        )
                    )
        
        # Check for vague references
        if not is_ambiguous and re.search(r'\b(it|this|that|they|them|those)\b', text.lower()):
            # Simple heuristic: if text contains pronouns without clear referents, it might be ambiguous
            is_ambiguous = True
            ambiguity_type = AmbiguityType.VAGUE_REFERENCE
            clarification_question = "Can you be more specific about what you're referring to?"
            
            # Add a generic interpretation
            if intents:
                possible_interpretations.append(
                    PossibleInterpretation(
                        intent=intents[0],
                        missing_entities=[],
                        clarification_question="Could you rephrase your request without using pronouns like 'it', 'this', or 'that'?"
                    )
                )
        
        # Check for homonyms (words with multiple meanings)
        homonyms = {
            r'\bbook\b': ["reserve", "reading material"],
            r'\bbank\b': ["financial institution", "river bank"],
            r'\baddress\b': ["location", "speech"],
            r'\brun\b': ["execute", "jog"]
        }
        
        if not is_ambiguous:
            for pattern, meanings in homonyms.items():
                if re.search(pattern, text.lower()):
                    is_ambiguous = True
                    ambiguity_type = AmbiguityType.HOMONYM
                    word = re.search(pattern, text.lower()).group(0)
                    clarification_question = f"When you say '{word}', do you mean {meanings[0]} or {meanings[1]}?"
                    
                    # Add possible interpretations for each meaning
                    for meaning in meanings:
                        possible_interpretations.append(
                            PossibleInterpretation(
                                intent=Intent(type=f"{word}_{meaning.replace(' ', '_')}", confidence=0.5),
                                missing_entities=[],
                                clarification_question=f"Did you mean '{word}' as in {meaning}?"
                            )
                        )
                    
                    break
        
        return cls(
            original_text=text,
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_type,
            clarification_question=clarification_question,
            possible_interpretations=possible_interpretations
        )
    
    def get_best_clarification_question(self) -> str:
        """
        Get the best clarification question to ask.
        
        Returns:
            The best clarification question, or a default question if none is available
        """
        if not self.is_ambiguous:
            return "I understand your request clearly."
        
        if self.clarification_question:
            return self.clarification_question
        
        # Fallback questions based on ambiguity type
        if self.ambiguity_type == AmbiguityType.MISSING_ENTITY:
            return "I need more information to process your request. Can you provide more details?"
        
        if self.ambiguity_type == AmbiguityType.MULTIPLE_INTENTS:
            return "I'm not sure what you're asking for. Could you clarify your request?"
        
        if self.ambiguity_type == AmbiguityType.VAGUE_REFERENCE:
            return "Could you be more specific about what you're referring to?"
        
        if self.ambiguity_type == AmbiguityType.HOMONYM:
            return "One of the words in your request has multiple meanings. Could you clarify?"
        
        # Default fallback
        return "I'm not sure I understand. Could you rephrase your request?"
    
    def resolve_with_clarification(self, clarification_text: str) -> Dict[str, Any]:
        """
        Attempt to resolve the ambiguity with user clarification.
        
        Args:
            clarification_text: User's response to the clarification question
            
        Returns:
            Dictionary with resolution results
        """
        result = {
            "resolved": False,
            "intent": None,
            "entities": {},
            "confidence": 0.0
        }
        
        # Simple keyword matching for demonstration purposes
        # In a real system, this would use more sophisticated NLP
        
        if self.ambiguity_type == AmbiguityType.MISSING_ENTITY:
            # Try to extract the missing entity from clarification
            if self.possible_interpretations:
                intent = self.possible_interpretations[0].intent
                missing_entities = self.possible_interpretations[0].missing_entities
                
                # Extract location for weather intent
                if intent.type == "weather" and "location" in missing_entities:
                    location_match = re.search(r'(?:in|for|at) ([A-Z][a-z]+ ?[A-Z]?[a-z]*)', clarification_text)
                    if location_match:
                        result["resolved"] = True
                        result["intent"] = intent.type
                        result["entities"] = intent.entities.copy()
                        result["entities"]["location"] = location_match.group(1)
                        result["confidence"] = 0.8
                
                # Extract service_type and date for booking intent
                elif intent.type == "booking":
                    # Try to extract service type
                    if "service_type" in missing_entities:
                        service_match = re.search(r'(?:a|an) ([a-z]+) (?:booking|reservation|appointment)', clarification_text.lower())
                        if service_match:
                            if "entities" not in result:
                                result["entities"] = intent.entities.copy()
                            result["entities"]["service_type"] = service_match.group(1)
                    
                    # Try to extract date
                    if "date" in missing_entities:
                        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', clarification_text)
                        if date_match:
                            if "entities" not in result:
                                result["entities"] = intent.entities.copy()
                            result["entities"]["date"] = date_match.group(1)
                    
                    # Check if we resolved all missing entities
                    if all(entity in result["entities"] for entity in missing_entities):
                        result["resolved"] = True
                        result["intent"] = intent.type
                        result["confidence"] = 0.8
        
        elif self.ambiguity_type == AmbiguityType.MULTIPLE_INTENTS:
            # Try to determine which intent the user confirmed
            for interp in self.possible_interpretations:
                intent_type = interp.intent.type
                
                # Check for keywords indicating this intent
                if intent_type == "weather" and any(kw in clarification_text.lower() for kw in ["weather", "temperature", "forecast"]):
                    result["resolved"] = True
                    result["intent"] = "weather"
                    result["entities"] = interp.intent.entities.copy()
                    result["confidence"] = 0.9
                    break
                
                elif intent_type == "booking" and any(kw in clarification_text.lower() for kw in ["book", "reservation", "appointment"]):
                    result["resolved"] = True
                    result["intent"] = "booking"
                    result["entities"] = interp.intent.entities.copy()
                    result["confidence"] = 0.9
                    break
        
        elif self.ambiguity_type == AmbiguityType.HOMONYM:
            # Try to determine which meaning the user confirmed
            for interp in self.possible_interpretations:
                # Extract the meaning from the intent type (e.g., "book_reserve" -> "reserve")
                parts = interp.intent.type.split('_')
                if len(parts) > 1:
                    word = parts[0]
                    meaning = '_'.join(parts[1:])
                    
                    # Check if clarification contains this meaning
                    if meaning.replace('_', ' ') in clarification_text.lower():
                        result["resolved"] = True
                        result["intent"] = interp.intent.type
                        result["entities"] = interp.intent.entities.copy()
                        result["confidence"] = 0.9
                        break
        
        return result


# Example usage
if __name__ == "__main__":
    # Example 1: Missing entity (location for weather)
    text1 = "What's the weather like?"
    entities1 = {}
    intents1 = [Intent(type="weather", confidence=0.9)]
    
    ambiguity1 = AmbiguityResolution.analyze(text1, entities1, intents1)
    print(f"Text: '{ambiguity1.original_text}'")
    print(f"Is ambiguous: {ambiguity1.is_ambiguous}")
    print(f"Ambiguity type: {ambiguity1.ambiguity_type}")
    print(f"Clarification question: {ambiguity1.clarification_question}")
    print()
    
    # Simulate user clarification
    clarification1 = "I want to know the weather in New York"
    resolution1 = ambiguity1.resolve_with_clarification(clarification1)
    print(f"Resolved: {resolution1['resolved']}")
    print(f"Intent: {resolution1['intent']}")
    print(f"Entities: {resolution1['entities']}")
    print(f"Confidence: {resolution1['confidence']}")
    print()
    
    # Example 2: Multiple intents
    text2 = "I need to make a reservation"
    entities2 = {}
    intents2 = [
        Intent(type="booking", confidence=0.6),
        Intent(type="information", confidence=0.5)
    ]
    
    ambiguity2 = AmbiguityResolution.analyze(text2, entities2, intents2)
    print(f"Text: '{ambiguity2.original_text}'")
    print(f"Is ambiguous: {ambiguity2.is_ambiguous}")
    print(f"Ambiguity type: {ambiguity2.ambiguity_type}")
    print(f"Clarification question: {ambiguity2.clarification_question}")
    print()
    
    # Simulate user clarification
    clarification2 = "I want to book a restaurant reservation"
    resolution2 = ambiguity2.resolve_with_clarification(clarification2)
    print(f"Resolved: {resolution2['resolved']}")
    print(f"Intent: {resolution2['intent']}")
    print(f"Entities: {resolution2['entities']}")
    print(f"Confidence: {resolution2['confidence']}")
    print()
    
    # Example 3: Homonym
    text3 = "I need to book a flight"
    entities3 = {}
    intents3 = [Intent(type="travel", confidence=0.7)]
    
    ambiguity3 = AmbiguityResolution.analyze(text3, entities3, intents3)
    print(f"Text: '{ambiguity3.original_text}'")
    print(f"Is ambiguous: {ambiguity3.is_ambiguous}")
    if ambiguity3.is_ambiguous:
        print(f"Ambiguity type: {ambiguity3.ambiguity_type}")
        print(f"Clarification question: {ambiguity3.clarification_question}")
    print()
