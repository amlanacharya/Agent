# 🔄 Module 3: Structured Data Validation - Lesson 4.5: Agent Output Validation 🧪

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand the importance of validating agent outputs
- 🛠️ Implement basic output validation using Pydantic
- 📊 Create format-specific validators for different response types
- 🔒 Build content safety validation systems
- 🧩 Ensure consistency and completeness in agent responses
- 🔄 Design validation pipelines for LLM-generated content

---

## 📚 Introduction to Agent Output Validation

In this lesson, we'll explore how to validate agent outputs before presenting them to users. While input validation ensures that agents receive proper instructions, output validation ensures that responses are correct, consistent, and appropriate. This is especially important for LLM-based agents, which may occasionally generate incorrect, inconsistent, or inappropriate content.

## 🧩 The Challenge of Agent Outputs

Agent outputs present several validation challenges:

1. **Factual Accuracy**: Ensuring responses contain correct information
2. **Consistency**: Maintaining consistency with previous responses
3. **Completeness**: Verifying that all user questions are addressed
4. **Safety**: Filtering out harmful or inappropriate content
5. **Format Adherence**: Ensuring outputs follow expected formats

> 💡 **Key Insight**: Unlike traditional application outputs, agent responses (especially from LLMs) can be unpredictable and require multiple layers of validation to ensure quality, safety, and accuracy.

---

## 🛠️ Basic Output Validation

Let's start with basic output validation using Pydantic:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
import re

class AgentResponse(BaseModel):
    message: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Optional[str] = None

    @field_validator('message')
    def validate_message_length(cls, v):
        """Validate that the message is not too short or too long."""
        if len(v) < 10:
            raise ValueError("Response is too short")
        if len(v) > 1000:
            raise ValueError("Response is too long")
        return v

    @field_validator('message')
    def validate_no_placeholders(cls, v):
        """Check for placeholder text that wasn't properly filled in."""
        placeholders = [
            "{placeholder}",
            "[insert",
            "[PLACEHOLDER]",
            "TODO",
            "FIXME"
        ]
        for placeholder in placeholders:
            if placeholder.lower() in v.lower():
                raise ValueError(f"Response contains placeholder text: {placeholder}")
        return v

# Usage
try:
    response = AgentResponse(
        message="Here's the information you requested about the weather in New York.",
        confidence=0.85,
        source="weather_api"
    )
    print("Valid response:", response)
except ValueError as e:
    print("Validation error:", e)

# Invalid example
try:
    response = AgentResponse(
        message="[INSERT WEATHER DATA]",
        confidence=0.85,
        source="weather_api"
    )
    print("Valid response:", response)
except ValueError as e:
    print("Validation error:", e)
```

---

## 🔄 Format-Specific Validation

Different response formats require different validation approaches:

### 📊 Structured Response Validation

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, date

class WeatherForecast(BaseModel):
    location: str
    date: date
    temperature: float
    condition: str
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

    @model_validator(mode='after')
    def validate_temperature_range(self):
        """Validate that temperature is within a reasonable range."""
        if not -100 <= self.temperature <= 150:
            raise ValueError(f"Temperature {self.temperature} is outside reasonable range")
        return self

    @model_validator(mode='after')
    def validate_condition(self):
        """Validate that condition is a known weather condition."""
        valid_conditions = [
            "sunny", "partly cloudy", "cloudy", "rainy", "stormy",
            "snowy", "foggy", "windy", "clear"
        ]
        if self.condition.lower() not in valid_conditions:
            raise ValueError(f"Unknown weather condition: {self.condition}")
        return self

class FlightInfo(BaseModel):
    flight_number: str
    departure: str
    arrival: str
    departure_time: datetime
    arrival_time: datetime
    status: Literal["on time", "delayed", "cancelled"]

    @model_validator(mode='after')
    def validate_times(self):
        """Validate that arrival time is after departure time."""
        if self.arrival_time <= self.departure_time:
            raise ValueError("Arrival time must be after departure time")
        return self

    @field_validator('flight_number')
    def validate_flight_number(cls, v):
        """Validate flight number format."""
        if not re.match(r'^[A-Z]{2}\d{3,4}$', v):
            raise ValueError("Invalid flight number format")
        return v

class AgentStructuredResponse(BaseModel):
    response_type: Literal["weather", "flight", "general"]
    text_response: str
    structured_data: Optional[Union[WeatherForecast, FlightInfo, Dict[str, Any]]] = None

    @model_validator(mode='after')
    def validate_structured_data_type(self):
        """Validate that structured data matches response type."""
        if self.response_type == "weather" and not isinstance(self.structured_data, WeatherForecast):
            raise ValueError("Weather response must include WeatherForecast data")
        elif self.response_type == "flight" and not isinstance(self.structured_data, FlightInfo):
            raise ValueError("Flight response must include FlightInfo data")
        return self

# Usage
try:
    # Valid weather response
    weather_response = AgentStructuredResponse(
        response_type="weather",
        text_response="Here's the weather forecast for New York tomorrow.",
        structured_data=WeatherForecast(
            location="New York",
            date=date(2023, 12, 25),
            temperature=45.5,
            condition="partly cloudy",
            humidity=65.0,
            wind_speed=10.2
        )
    )
    print("Valid weather response:", weather_response)
except ValueError as e:
    print("Validation error:", e)

try:
    # Invalid flight response
    flight_response = AgentStructuredResponse(
        response_type="flight",
        text_response="Here's your flight information.",
        structured_data=FlightInfo(
            flight_number="AA123",
            departure="JFK",
            arrival="LAX",
            departure_time=datetime(2023, 12, 25, 10, 0),
            arrival_time=datetime(2023, 12, 25, 9, 0),  # Invalid: arrival before departure
            status="on time"
        )
    )
    print("Valid flight response:", flight_response)
except ValueError as e:
    print("Validation error:", e)
```

### 💬 Natural Language Response Validation

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
import re

class NaturalLanguageResponse(BaseModel):
    text: str
    tone: Literal["formal", "informal", "technical", "friendly"] = "friendly"
    contains_question: bool = False

    @field_validator('text')
    def validate_text_quality(cls, v):
        """Validate text quality."""
        # Check for minimum length
        if len(v) < 20:
            raise ValueError("Response is too short")

        # Check for excessive repetition
        words = v.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check non-trivial words
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count > 3 and len(words) < 50:
                raise ValueError(f"Response contains excessive repetition of '{word}'")

        return v

    @model_validator(mode='after')
    def validate_tone_consistency(self):
        """Validate that the text matches the specified tone."""
        formal_indicators = ["would", "could", "please", "thank you", "sincerely"]
        informal_indicators = ["hey", "cool", "awesome", "yeah", "btw"]
        technical_indicators = ["specifically", "furthermore", "additionally", "consequently"]

        text_lower = self.text.lower()

        if self.tone == "formal" and any(indicator in text_lower for indicator in informal_indicators):
            raise ValueError("Formal tone contains informal language")

        if self.tone == "informal" and all(indicator not in text_lower for indicator in informal_indicators):
            if any(indicator in text_lower for indicator in formal_indicators):
                raise ValueError("Informal tone contains predominantly formal language")

        if self.tone == "technical" and all(indicator not in text_lower for indicator in technical_indicators):
            raise ValueError("Technical tone lacks technical language")

        return self

    @model_validator(mode='after')
    def validate_question_consistency(self):
        """Validate that contains_question flag matches content."""
        has_question_mark = "?" in self.text
        has_question_words = any(word in self.text.lower() for word in ["who", "what", "when", "where", "why", "how"])

        actual_contains_question = has_question_mark or (has_question_words and has_question_mark)

        if self.contains_question != actual_contains_question:
            raise ValueError(f"contains_question flag ({self.contains_question}) doesn't match content")

        return self

# Usage
try:
    response = NaturalLanguageResponse(
        text="Thank you for your inquiry. We would be pleased to assist you with your request. Could you please provide more details?",
        tone="formal",
        contains_question=True
    )
    print("Valid response:", response)
except ValueError as e:
    print("Validation error:", e)

try:
    # Tone mismatch
    response = NaturalLanguageResponse(
        text="Hey there! Yeah, we can totally help with that. It's awesome that you reached out!",
        tone="formal",
        contains_question=False
    )
    print("Valid response:", response)
except ValueError as e:
    print("Validation error:", e)
```

---

## 🔍 Content Safety Validation

Ensuring agent outputs are safe and appropriate:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
import re

class ContentSafetyCheck(BaseModel):
    text: str
    is_safe: bool = True
    safety_issues: List[str] = []

    @field_validator('text')
    def validate_content_safety(cls, v):
        """Validate that content is safe and appropriate."""
        issues = []

        # Check for potentially unsafe content (simplified example)
        unsafe_patterns = [
            (r'\b(password|credit card|ssn|social security)\b', "Contains sensitive data references"),
            (r'\b(hate|kill|violent|attack)\b', "Contains potentially violent language"),
            (r'\b(profanity1|profanity2|profanity3)\b', "Contains profanity")
        ]

        for pattern, issue in unsafe_patterns:
            if re.search(pattern, v.lower()):
                issues.append(issue)

        # Update model
        if issues:
            return v

        return v

    @classmethod
    def check(cls, text: str):
        """Check text for safety issues."""
        instance = cls(text=text)
        issues = []

        # Check for potentially unsafe content (simplified example)
        unsafe_patterns = [
            (r'\b(password|credit card|ssn|social security)\b', "Contains sensitive data references"),
            (r'\b(hate|kill|violent|attack)\b', "Contains potentially violent language"),
            (r'\b(profanity1|profanity2|profanity3)\b', "Contains profanity")
        ]

        for pattern, issue in unsafe_patterns:
            if re.search(pattern, text.lower()):
                issues.append(issue)

        # Update instance
        instance.safety_issues = issues
        instance.is_safe = len(issues) == 0

        return instance

# Usage
text1 = "Here's the weather forecast for tomorrow."
safety_check1 = ContentSafetyCheck.check(text1)
print(f"Text: '{safety_check1.text}'")
print(f"Is safe: {safety_check1.is_safe}")
if safety_check1.safety_issues:
    print("Safety issues:")
    for issue in safety_check1.safety_issues:
        print(f"- {issue}")

text2 = "I hate to tell you this, but your password might be compromised."
safety_check2 = ContentSafetyCheck.check(text2)
print(f"\nText: '{safety_check2.text}'")
print(f"Is safe: {safety_check2.is_safe}")
if safety_check2.safety_issues:
    print("Safety issues:")
    for issue in safety_check2.safety_issues:
        print(f"- {issue}")
```

---

## 🧠 Consistency Validation

Ensuring agent responses are consistent with previous information:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class ConversationContext(BaseModel):
    facts: Dict[str, Any] = {}
    last_updated: Dict[str, datetime] = {}

    def add_fact(self, key: str, value: Any):
        """Add a fact to the context."""
        self.facts[key] = value
        self.last_updated[key] = datetime.now()

    def get_fact(self, key: str) -> Optional[Any]:
        """Get a fact from the context."""
        return self.facts.get(key)

class ConsistencyValidator(BaseModel):
    response: str
    context: ConversationContext

    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate that the response is consistent with known facts."""
        inconsistencies = []

        # Check for weather inconsistencies
        if "weather" in self.context.facts:
            weather = self.context.facts["weather"]
            if weather == "sunny" and "rain" in self.response.lower():
                inconsistencies.append("Response mentions rain but weather was previously stated as sunny")

        # Check for location inconsistencies
        if "location" in self.context.facts:
            location = self.context.facts["location"]
            location_pattern = rf"in ([A-Za-z ]+)(?!\s*{re.escape(location)})"
            location_match = re.search(location_pattern, self.response)
            if location_match and location_match.group(1).lower() != location.lower():
                inconsistencies.append(f"Response mentions a different location ({location_match.group(1)}) than previously established ({location})")

        if inconsistencies:
            raise ValueError(f"Response contains inconsistencies: {'; '.join(inconsistencies)}")

        return self

# Usage
context = ConversationContext()
context.add_fact("weather", "sunny")
context.add_fact("location", "New York")
context.add_fact("temperature", 75)

try:
    # Consistent response
    validator = ConsistencyValidator(
        response="It's a beautiful sunny day in New York with a temperature of 75 degrees.",
        context=context
    )
    print("Response is consistent with context")
except ValueError as e:
    print(f"Consistency error: {e}")

try:
    # Inconsistent response
    validator = ConsistencyValidator(
        response="Expect rain showers in Boston today.",
        context=context
    )
    print("Response is consistent with context")
except ValueError as e:
    print(f"Consistency error: {e}")
```

---

## 🔄 Completeness Validation

Ensuring agent responses address all aspects of user queries:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
import re

class UserQuery(BaseModel):
    text: str
    extracted_questions: List[str] = []

    @model_validator(mode='after')
    def extract_questions(self):
        """Extract questions from the query text."""
        # Simple question extraction (in a real system, this would be more sophisticated)
        question_pattern = r'([^.!?]+\?)'
        matches = re.findall(question_pattern, self.text)
        self.extracted_questions = [q.strip() for q in matches]

        # If no question marks but has question words, treat the whole thing as a question
        if not self.extracted_questions and any(word in self.text.lower() for word in ["who", "what", "when", "where", "why", "how"]):
            self.extracted_questions = [self.text]

        return self

class CompletenessValidator(BaseModel):
    user_query: UserQuery
    agent_response: str

    @model_validator(mode='after')
    def validate_completeness(self):
        """Validate that the response addresses all questions in the query."""
        if not self.user_query.extracted_questions:
            return self  # No questions to address

        unanswered_questions = []

        for question in self.user_query.extracted_questions:
            # Extract key entities from the question
            entities = self._extract_entities(question)

            # Check if response addresses these entities
            if not all(entity.lower() in self.agent_response.lower() for entity in entities):
                unanswered_questions.append(question)

        if unanswered_questions:
            raise ValueError(f"Response does not address all questions: {unanswered_questions}")

        return self

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified version)."""
        # Remove common question words and stop words
        stop_words = ["who", "what", "when", "where", "why", "how", "is", "are", "the", "a", "an"]
        words = text.lower().replace("?", "").split()
        entities = [word for word in words if word not in stop_words and len(word) > 3]

        return entities[:2]  # Return top 2 entities for simplicity

# Usage
query = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
print(f"Extracted questions: {query.extracted_questions}")

try:
    # Complete response
    validator = CompletenessValidator(
        user_query=query,
        agent_response="The weather in New York is currently sunny and 75°F. The best time to visit New York is during spring (April to June) or fall (September to November) when the weather is mild."
    )
    print("Response addresses all questions")
except ValueError as e:
    print(f"Completeness error: {e}")

try:
    # Incomplete response
    validator = CompletenessValidator(
        user_query=query,
        agent_response="The weather in New York is currently sunny and 75°F."
    )
    print("Response addresses all questions")
except ValueError as e:
    print(f"Completeness error: {e}")
```

---

## 💪 Practice Exercises

1. **Create a Hybrid Response Validator**: Build a Pydantic model for validating agent responses that include both text and structured data components.

2. **Implement a Content Moderation System**: Develop a system that checks agent outputs for inappropriate content, bias, and factual accuracy.

3. **Build a Consistency Validator**: Create a validator that ensures agent responses don't contradict information provided in previous interactions.

4. **Design a Completeness Checker**: Implement a system that verifies all parts of a multi-part question are addressed in the response.

5. **Create a Quality Validator**: Build a response quality validator that checks for clarity, conciseness, and helpfulness.

---

## 🔍 Key Concepts to Remember

1. **Output Validation**: Ensures agent responses are correct, consistent, and appropriate
2. **Format-Specific Validation**: Different response types require different validation approaches
3. **Content Safety**: Filtering out harmful or inappropriate content
4. **Consistency Checking**: Ensuring responses don't contradict known information
5. **Completeness Validation**: Verifying that all user questions are addressed
6. **Quality Assessment**: Evaluating clarity, conciseness, and helpfulness of responses

---

## 📚 Additional Resources

- [Pydantic Validation Documentation](https://docs.pydantic.dev/latest/usage/validators/)
- [Content Moderation Best Practices](https://aws.amazon.com/solutions/implementations/content-moderation-api/)
- [NLP Quality Metrics](https://huggingface.co/docs/evaluate/index)
- [LLM Output Validation Techniques](https://www.pinecone.io/learn/langchain-output-validation/)
- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)

---

## 🚀 Next Steps

In the next lesson, we'll explore state validation in agent systems, focusing on ensuring data consistency across interactions and validating state transitions.

---

> 💡 **Note on LLM Integration**: When working with LLM-based agents, output validation becomes critical as these models can occasionally generate incorrect, inconsistent, or inappropriate content. Implementing robust validation pipelines helps ensure that only high-quality, safe responses reach your users, improving the overall reliability of your agent system.

---

Happy coding! 🔄
