# 🤖 Module 3: Structured Data Validation - Lesson 4.4: Agent Input Validation Patterns 🔍

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧠 Understand the unique challenges of validating user inputs in agent systems
- 🔄 Implement preprocessing and normalization techniques for natural language inputs
- 🎯 Build intent recognition and entity extraction systems with validation
- 🧩 Create strategies for handling ambiguous or incomplete user requests
- 📊 Design validation feedback loops to improve agent performance over time

---

## 📚 Introduction to Agent Input Validation

In this lesson, we'll explore how to validate and process user inputs in agent systems. While previous lessons focused on general validation patterns, agent systems present unique challenges due to the unpredictable and often ambiguous nature of user inputs. Effective input validation is crucial for building robust, reliable agents that can handle real-world interactions.

## 🧩 The Challenge of Agent Inputs

Agent systems face several input validation challenges that traditional applications don't:

1. **Natural Language Ambiguity**: Users often provide ambiguous or incomplete instructions
2. **Intent Recognition**: Determining what the user actually wants to accomplish
3. **Entity Extraction**: Identifying key entities and parameters from unstructured text
4. **Context Dependency**: Interpreting inputs based on conversation history
5. **Multi-modal Inputs**: Handling text, voice, images, and other input types

> 💡 **Key Insight**: Unlike traditional form validation, agent input validation must handle the inherent ambiguity and variability of natural language while still extracting structured, actionable data.

---

## 🛠️ Input Preprocessing and Normalization

Before validation, agent inputs often need preprocessing to convert them into a more structured format:

### 📝 Text Normalization

```python
from pydantic import BaseModel, Field
from typing import Optional, List
import re

class NormalizedInput(BaseModel):
    raw_text: str
    normalized_text: str
    tokens: List[str]

    @classmethod
    def from_raw_input(cls, raw_input: str):
        """Normalize raw user input text."""
        # Convert to lowercase
        normalized = raw_input.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove special characters (keeping alphanumeric, spaces, and basic punctuation)
        normalized = re.sub(r'[^\w\s.,?!-]', '', normalized)

        # Tokenize (simple space-based tokenization)
        tokens = normalized.split()

        return cls(
            raw_text=raw_input,
            normalized_text=normalized,
            tokens=tokens
        )

# Usage
user_input = "  Hey there!   What's the weather like in New York City? "
normalized = NormalizedInput.from_raw_input(user_input)
print(f"Raw: '{normalized.raw_text}'")
print(f"Normalized: '{normalized.normalized_text}'")
print(f"Tokens: {normalized.tokens}")
```

### 🔍 Entity Extraction

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re

class Entity(BaseModel):
    type: str
    value: Any
    confidence: float = 1.0

class ExtractedEntities(BaseModel):
    text: str
    entities: Dict[str, Entity] = {}

    @classmethod
    def extract_from_text(cls, text: str):
        """Extract entities from text using pattern matching."""
        entities = {}

        # Extract dates (simple pattern)
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        date_matches = re.findall(date_pattern, text)
        if date_matches:
            entities["date"] = Entity(
                type="date",
                value=date_matches[0],
                confidence=0.8
            )

        # Extract locations (simple pattern for demo)
        location_pattern = r'in ([A-Z][a-z]+ ?[A-Z]?[a-z]*)'
        location_matches = re.findall(location_pattern, text)
        if location_matches:
            entities["location"] = Entity(
                type="location",
                value=location_matches[0],
                confidence=0.7
            )

        # Extract numbers
        number_pattern = r'(\d+(?:\.\d+)?)'
        number_matches = re.findall(number_pattern, text)
        if number_matches:
            entities["number"] = Entity(
                type="number",
                value=float(number_matches[0]),
                confidence=0.9
            )

        return cls(
            text=text,
            entities=entities
        )

# Usage
text = "I need to book a flight to New York on 12/25/2023"
extracted = ExtractedEntities.extract_from_text(text)
print(f"Text: '{extracted.text}'")
print("Extracted entities:")
for entity_type, entity in extracted.entities.items():
    print(f"- {entity_type}: {entity.value} (confidence: {entity.confidence})")
```

---

## 🔄 Intent Recognition and Validation

Identifying user intent is crucial for agent systems:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Literal
import re

class Intent(BaseModel):
    type: str
    confidence: float
    entities: Dict[str, Any] = {}

class WeatherIntent(Intent):
    type: Literal["weather"] = "weather"
    entities: Dict[str, Any] = {}

    @model_validator(mode='after')
    def validate_weather_intent(self):
        """Validate that we have the necessary entities for a weather intent."""
        if "location" not in self.entities:
            raise ValueError("Weather intent requires a location")
        return self

class BookingIntent(Intent):
    type: Literal["booking"] = "booking"
    entities: Dict[str, Any] = {}

    @model_validator(mode='after')
    def validate_booking_intent(self):
        """Validate that we have the necessary entities for a booking intent."""
        required_entities = ["service_type", "date"]
        missing = [entity for entity in required_entities if entity not in self.entities]

        if missing:
            raise ValueError(f"Booking intent requires: {', '.join(missing)}")
        return self

class IntentClassifier(BaseModel):
    text: str
    intents: List[Intent] = []

    @classmethod
    def classify(cls, text: str):
        """Classify text into intents."""
        intents = []

        # Check for weather intent
        weather_patterns = ["weather", "temperature", "forecast", "rain", "sunny"]
        if any(pattern in text.lower() for pattern in weather_patterns):
            # Extract location
            location_match = re.search(r'in ([A-Z][a-z]+ ?[A-Z]?[a-z]*)', text)
            location = location_match.group(1) if location_match else None

            try:
                weather_intent = WeatherIntent(
                    confidence=0.8,
                    entities={"location": location} if location else {}
                )
                intents.append(weather_intent)
            except ValueError:
                # Intent validation failed, but we still record it with lower confidence
                intents.append(Intent(
                    type="weather",
                    confidence=0.4,
                    entities={"location": location} if location else {}
                ))

        # Check for booking intent
        booking_patterns = ["book", "reserve", "appointment", "schedule"]
        if any(pattern in text.lower() for pattern in booking_patterns):
            # Extract service type
            service_match = re.search(r'book (?:a|an) ([a-z]+)', text.lower())
            service_type = service_match.group(1) if service_match else None

            # Extract date
            date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
            date = date_match.group(1) if date_match else None

            try:
                booking_intent = BookingIntent(
                    confidence=0.8,
                    entities={
                        "service_type": service_type,
                        "date": date
                    }
                )
                intents.append(booking_intent)
            except ValueError:
                # Intent validation failed, but we still record it with lower confidence
                intents.append(Intent(
                    type="booking",
                    confidence=0.4,
                    entities={
                        "service_type": service_type,
                        "date": date
                    }
                ))

        return cls(
            text=text,
            intents=sorted(intents, key=lambda x: x.confidence, reverse=True)
        )

# Usage
text1 = "What's the weather like in New York?"
classification1 = IntentClassifier.classify(text1)
print(f"Text: '{classification1.text}'")
print("Intents:")
for intent in classification1.intents:
    print(f"- {intent.type} (confidence: {intent.confidence})")
    print(f"  Entities: {intent.entities}")

text2 = "Book a flight on 12/25/2023"
classification2 = IntentClassifier.classify(text2)
print(f"\nText: '{classification2.text}'")
print("Intents:")
for intent in classification2.intents:
    print(f"- {intent.type} (confidence: {intent.confidence})")
    print(f"  Entities: {intent.entities}")
```

---

## 🧠 Handling Ambiguous Inputs

Agent systems must gracefully handle ambiguous or incomplete inputs:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import re

class AmbiguityResolution(BaseModel):
    original_text: str
    is_ambiguous: bool = False
    ambiguity_type: Optional[str] = None
    clarification_question: Optional[str] = None
    possible_interpretations: List[Dict[str, Any]] = []

    @classmethod
    def analyze(cls, text: str, entities: Dict[str, Any], intents: List[Dict[str, Any]]):
        """Analyze input for ambiguities."""
        is_ambiguous = False
        ambiguity_type = None
        clarification_question = None
        possible_interpretations = []

        # Check for missing required entities
        if intents and intents[0].get("type") == "weather":
            if "location" not in entities:
                is_ambiguous = True
                ambiguity_type = "missing_entity"
                clarification_question = "Which location would you like the weather for?"

        # Check for multiple possible intents with similar confidence
        if len(intents) > 1 and abs(intents[0].get("confidence", 0) - intents[1].get("confidence", 0)) < 0.3:
            is_ambiguous = True
            ambiguity_type = "multiple_intents"
            intent_types = [intent.get("type") for intent in intents[:2]]
            clarification_question = f"Are you trying to {intent_types[0]} or {intent_types[1]}?"
            possible_interpretations = intents[:2]

        # Check for vague or underspecified requests
        vague_terms = ["thing", "stuff", "it", "that"]
        if any(term in text.lower().split() for term in vague_terms):
            is_ambiguous = True
            ambiguity_type = "vague_reference"
            clarification_question = "Could you be more specific about what you're referring to?"

        return cls(
            original_text=text,
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_type,
            clarification_question=clarification_question,
            possible_interpretations=possible_interpretations
        )

# Usage
text = "What's the weather like?"
entities = {}
intents = [{"type": "weather", "confidence": 0.8}]

ambiguity = AmbiguityResolution.analyze(text, entities, intents)
print(f"Text: '{ambiguity.original_text}'")
print(f"Is ambiguous: {ambiguity.is_ambiguous}")
if ambiguity.is_ambiguous:
    print(f"Ambiguity type: {ambiguity.ambiguity_type}")
    print(f"Clarification question: {ambiguity.clarification_question}")
```

---

## 🔁 Validation Feedback Loops

Implement feedback loops to improve validation over time:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class ValidationFeedback(BaseModel):
    input_text: str
    validation_result: Literal["success", "failure", "ambiguous"]
    failure_reason: Optional[str] = None
    user_correction: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationLearningSystem(BaseModel):
    feedback_history: List[ValidationFeedback] = []

    def add_feedback(self, feedback: ValidationFeedback):
        """Add validation feedback to history."""
        self.feedback_history.append(feedback)

    def get_common_failures(self, limit: int = 5):
        """Get the most common validation failures."""
        failures = [f for f in self.feedback_history if f.validation_result == "failure"]

        # Group by failure reason
        failure_counts = {}
        for failure in failures:
            reason = failure.failure_reason or "unknown"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        # Sort by count
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_failures[:limit]

    def get_improvement_suggestions(self):
        """Generate suggestions for improving validation based on feedback."""
        common_failures = self.get_common_failures()

        suggestions = []
        for reason, count in common_failures:
            if "missing_entity" in reason:
                suggestions.append(f"Improve entity extraction for {reason.replace('missing_entity_', '')}")
            elif "ambiguous" in reason:
                suggestions.append("Add more clarification questions for ambiguous inputs")
            elif "intent" in reason:
                suggestions.append("Refine intent classification model")

        return suggestions

# Usage
learning_system = ValidationLearningSystem()

# Add some feedback
learning_system.add_feedback(ValidationFeedback(
    input_text="What's the weather like?",
    validation_result="failure",
    failure_reason="missing_entity_location"
))

learning_system.add_feedback(ValidationFeedback(
    input_text="Book something next week",
    validation_result="failure",
    failure_reason="missing_entity_service_type"
))

learning_system.add_feedback(ValidationFeedback(
    input_text="Is it going to rain?",
    validation_result="ambiguous",
    failure_reason="missing_entity_location"
))

# Get improvement suggestions
suggestions = learning_system.get_improvement_suggestions()
print("Validation improvement suggestions:")
for suggestion in suggestions:
    print(f"- {suggestion}")
```

---

## 🔄 Integrating with LLM-Based Agents

For LLM-based agents, we can use Pydantic to validate structured outputs:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
import json

class LLMAgentInput(BaseModel):
    user_message: str
    conversation_history: List[Dict[str, str]] = []

    def to_prompt(self):
        """Convert to a prompt for the LLM."""
        history_text = ""
        for message in self.conversation_history[-5:]:  # Last 5 messages
            role = message.get("role", "user")
            content = message.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"

        return f"""
        Conversation History:
        {history_text}

        User: {self.user_message}

        Extract the following information from the user's message:
        1. Intent (weather, booking, general_question, etc.)
        2. Entities (location, date, service_type, etc.)
        3. Is the request ambiguous? If so, what clarification is needed?

        Respond in JSON format with the following structure:
        {{
            "intent": {{
                "type": "intent_type",
                "confidence": 0.0 to 1.0
            }},
            "entities": {{
                "entity_name": "entity_value"
            }},
            "is_ambiguous": true/false,
            "clarification_needed": "question to ask if ambiguous"
        }}
        """

class LLMAgentOutput(BaseModel):
    intent: Dict[str, Any]
    entities: Dict[str, Any] = {}
    is_ambiguous: bool = False
    clarification_needed: Optional[str] = None

    @model_validator(mode='after')
    def validate_output(self):
        """Validate the LLM output structure."""
        # Ensure intent has required fields
        if "type" not in self.intent:
            raise ValueError("Intent must have a 'type' field")
        if "confidence" not in self.intent:
            self.intent["confidence"] = 0.7  # Default confidence

        # Validate confidence is between 0 and 1
        if not 0 <= self.intent["confidence"] <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        # If ambiguous, ensure clarification is provided
        if self.is_ambiguous and not self.clarification_needed:
            raise ValueError("If request is ambiguous, clarification_needed must be provided")

        return self

    @classmethod
    def from_llm_response(cls, llm_response: str):
        """Parse LLM response into structured output."""
        try:
            # Extract JSON from the response (in case there's additional text)
            json_match = re.search(r'({.*})', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_response

            # Parse JSON
            data = json.loads(json_str)
            return cls(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Handle parsing errors
            return cls(
                intent={"type": "unknown", "confidence": 0.0},
                is_ambiguous=True,
                clarification_needed="I couldn't understand your request. Could you rephrase it?"
            )

# Simulated LLM response
llm_response = """
{
    "intent": {
        "type": "weather",
        "confidence": 0.85
    },
    "entities": {
        "location": "New York"
    },
    "is_ambiguous": false,
    "clarification_needed": null
}
"""

# Parse and validate
try:
    parsed_output = LLMAgentOutput.from_llm_response(llm_response)
    print("Parsed LLM output:")
    print(f"Intent: {parsed_output.intent}")
    print(f"Entities: {parsed_output.entities}")
    print(f"Is ambiguous: {parsed_output.is_ambiguous}")
    if parsed_output.clarification_needed:
        print(f"Clarification: {parsed_output.clarification_needed}")
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## 💪 Practice Exercises

1. **Create a Command Validator**: Build a Pydantic model for validating a complex agent command with multiple parameters and options.

2. **Build an Entity Extractor**: Implement an entity extraction system that can identify dates, times, locations, and people from natural language text.

3. **Design a Feedback System**: Create a feedback loop system that tracks validation failures and suggests improvements to the validation logic.

4. **Implement Ambiguity Resolution**: Develop a system that generates appropriate clarification questions based on the type of ambiguity.

5. **Create Context-Aware Validation**: Build a validation system that uses conversation history to resolve ambiguous references.

---

## 🔍 Key Concepts to Remember

1. **Input Preprocessing**: Converting raw user inputs into normalized, structured formats
2. **Entity Extraction**: Identifying key information like dates, locations, and quantities
3. **Intent Recognition**: Determining the user's goal from natural language input
4. **Ambiguity Resolution**: Strategies for handling unclear or incomplete requests
5. **Validation Feedback Loops**: Using validation failures to improve future performance
6. **LLM Output Validation**: Ensuring LLM responses meet structural requirements

---

## 📚 Additional Resources

- [Rasa NLU Documentation](https://rasa.com/docs/rasa/nlu/about/)
- [Pydantic Validation Documentation](https://docs.pydantic.dev/latest/usage/validators/)
- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Intent Recognition Patterns](https://www.ibm.com/cloud/watson-assistant/docs/intent-detection)
- [spaCy Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)

---

## 🚀 Next Steps

In the next lesson, we'll explore agent output validation patterns, focusing on ensuring that agent responses are correct, consistent, and appropriate before presenting them to users.

---

> 💡 **Note on LLM Integration**: When working with LLMs in agent systems, input validation becomes even more critical. LLMs can be sensitive to the format and content of inputs, and proper validation helps ensure that the LLM receives well-structured prompts that lead to high-quality outputs. Similarly, validating LLM outputs helps maintain consistency and reliability in your agent's responses.

---

Happy coding! 🤖
