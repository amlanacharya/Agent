# 🤖 Module 3: Structured Data Validation - Lesson 4.7: Agent-Specific Validation Patterns 🔍

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧠 Understand the unique validation requirements for different agent types
- 🌐 Implement domain-specific validation for various business contexts
- 🔄 Create specialized validators for chatbots, task agents, and knowledge agents
- 🔒 Build safety and compliance validation for regulated domains
- 🤝 Design validation systems for multi-agent interactions
- 🛠️ Develop custom validators for agent-specific scenarios

---

## 📚 Introduction to Agent-Specific Validation

In this lesson, we'll explore validation patterns specific to different types of agents and domains. While general validation principles apply across all agent systems, each agent type and domain has unique requirements that demand specialized validation approaches. By tailoring validation to specific agent contexts, we can significantly improve reliability, user experience, and overall system performance.

## 🧩 Domain-Specific Validation for Agent Types

Different agent types require different validation strategies based on their primary functions:

### 💬 Chatbot Agents

Chatbots focus on natural language conversation and require validation for:

1. **Conversation Flow**: Ensuring logical progression through conversation states
2. **Response Appropriateness**: Validating tone, style, and content match user expectations
3. **Personality Consistency**: Maintaining consistent agent personality traits

> 💡 **Key Insight**: Chatbot validation must balance personality consistency with appropriate responses, ensuring the agent maintains a coherent identity while adapting to different conversation contexts.

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class ChatbotResponse(BaseModel):
    message: str
    conversation_state: str
    personality_traits: Dict[str, float] = Field(
        description="Dictionary of personality traits with values between 0 and 1"
    )

    @model_validator(mode='after')
    def validate_personality_consistency(self):
        """Validate that the message reflects the personality traits."""
        # Example: Check if a 'formal' personality uses appropriate language
        if self.personality_traits.get("formal", 0) > 0.7:
            informal_phrases = ["hey", "yeah", "cool", "awesome", "btw"]
            if any(phrase in self.message.lower() for phrase in informal_phrases):
                raise ValueError("Message tone doesn't match formal personality trait")

        # Example: Check if an 'empathetic' personality acknowledges user emotions
        if self.personality_traits.get("empathetic", 0) > 0.7:
            if "sorry to hear" not in self.message.lower() and "understand" not in self.message.lower() and "feel" not in self.message.lower():
                # This is simplified; real implementation would be more sophisticated
                pass  # In production, you might raise a warning or log this

        return self

# Usage
try:
    response = ChatbotResponse(
        message="I understand your frustration with the delayed shipment. I'd be happy to help track your package.",
        conversation_state="handling_complaint",
        personality_traits={"formal": 0.8, "empathetic": 0.9, "technical": 0.3}
    )
    print("Valid chatbot response:", response)
except ValueError as e:
    print("Validation error:", e)
```

### ✅ Task-Oriented Agents

Task agents focus on completing specific actions and require validation for:

1. **Task Parameters**: Ensuring all required parameters are present and valid
2. **Preconditions**: Validating that prerequisites for task execution are met
3. **Execution Status**: Tracking and validating task execution states

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime, date

class TaskParameters(BaseModel):
    """Base class for task parameters."""
    pass

class CalendarEventParameters(TaskParameters):
    title: str
    start_time: datetime
    end_time: datetime
    attendees: List[str] = []
    location: Optional[str] = None

    @model_validator(mode='after')
    def validate_times(self):
        """Validate that end time is after start time."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self

class TaskAgent(BaseModel):
    task_type: str
    parameters: Union[TaskParameters, Dict[str, Any]]
    execution_status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    preconditions_met: bool = False

    @model_validator(mode='after')
    def validate_parameters_type(self):
        """Validate that parameters match the task type."""
        if self.task_type == "calendar_event" and not isinstance(self.parameters, CalendarEventParameters):
            if isinstance(self.parameters, dict):
                # Try to convert dict to proper type
                try:
                    self.parameters = CalendarEventParameters(**self.parameters)
                except Exception as e:
                    raise ValueError(f"Invalid calendar event parameters: {e}")
            else:
                raise ValueError("Calendar event task requires CalendarEventParameters")

        return self

    @model_validator(mode='after')
    def validate_execution_readiness(self):
        """Validate that the task can be executed."""
        if self.execution_status == "in_progress" and not self.preconditions_met:
            raise ValueError("Cannot execute task: preconditions not met")
        return self

# Usage
try:
    # Valid task
    task = TaskAgent(
        task_type="calendar_event",
        parameters=CalendarEventParameters(
            title="Team Meeting",
            start_time=datetime(2023, 12, 25, 10, 0),
            end_time=datetime(2023, 12, 25, 11, 0),
            attendees=["alice@example.com", "bob@example.com"]
        ),
        preconditions_met=True,
        execution_status="in_progress"
    )
    print("Valid task:", task)
except ValueError as e:
    print("Validation error:", e)
```

### 🧠 Knowledge Agents

Knowledge agents focus on information retrieval and require validation for:

1. **Query Understanding**: Validating that the query is properly interpreted
2. **Source Reliability**: Ensuring information comes from reliable sources
3. **Answer Accuracy**: Validating that responses accurately answer the query

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class Source(BaseModel):
    name: str
    url: Optional[str] = None
    reliability_score: float = Field(ge=0.0, le=1.0)
    last_updated: Optional[datetime] = None

class KnowledgeResponse(BaseModel):
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[Source] = []

    @model_validator(mode='after')
    def validate_source_reliability(self):
        """Validate that sources are reliable enough for the given confidence."""
        if self.sources:
            avg_reliability = sum(s.reliability_score for s in self.sources) / len(self.sources)
            if avg_reliability < 0.7 and self.confidence > 0.8:
                raise ValueError("High confidence with low reliability sources")

            # Check for outdated sources
            current_time = datetime.now()
            for source in self.sources:
                if source.last_updated:
                    days_old = (current_time - source.last_updated).days
                    if days_old > 365 and self.confidence > 0.9:  # Older than a year
                        raise ValueError(f"High confidence with outdated source: {source.name}")

        return self

    @model_validator(mode='after')
    def validate_answer_relevance(self):
        """Validate that the answer is relevant to the query."""
        # This is a simplified check; real implementation would use NLP
        query_keywords = set(self.query.lower().split())
        answer_keywords = set(self.answer.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about"}
        query_keywords = query_keywords - common_words

        # Check if any keywords from query appear in answer
        if not query_keywords.intersection(answer_keywords) and self.confidence > 0.7:
            # This is just a warning in this example
            print("Warning: Answer may not be relevant to query")

        return self

# Usage
try:
    response = KnowledgeResponse(
        query="What is the capital of France?",
        answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower.",
        confidence=0.95,
        sources=[
            Source(
                name="World Geography Database",
                url="https://example.com/geography",
                reliability_score=0.9,
                last_updated=datetime(2023, 1, 15)
            )
        ]
    )
    print("Valid knowledge response:", response)
except ValueError as e:
    print("Validation error:", e)
```

---

## 🌐 Domain-Specific Validation

Different domains require specialized validation rules:

### 🛒 E-commerce Domain

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date

class Product(BaseModel):
    id: str
    name: str
    price: float = Field(gt=0)
    currency: str = "USD"
    availability: Literal["in_stock", "out_of_stock", "pre_order"] = "in_stock"
    shipping_weight: Optional[float] = None

    @model_validator(mode='after')
    def validate_product_consistency(self):
        """Validate product data consistency."""
        if self.availability == "out_of_stock" and self.price == 0:
            raise ValueError("Price should not be zero for out-of-stock items")

        # Shipping weight is required for physical products (simplified check)
        if self.price > 0 and self.shipping_weight is None:
            # This could be a digital product, so just a warning
            print(f"Warning: No shipping weight for product {self.id}")

        return self

class EcommerceAgent(BaseModel):
    user_query: str
    recommended_products: List[Product] = []
    price_range: Optional[Dict[str, float]] = None

    @model_validator(mode='after')
    def validate_recommendations(self):
        """Validate that recommendations match user query and price range."""
        # Check if recommendations match price range
        if self.price_range and self.recommended_products:
            min_price = self.price_range.get("min", 0)
            max_price = self.price_range.get("max", float('inf'))

            for product in self.recommended_products:
                if product.price < min_price or product.price > max_price:
                    raise ValueError(f"Product {product.name} price (${product.price}) outside requested range")

        # Check if out-of-stock items are recommended
        out_of_stock = [p.name for p in self.recommended_products if p.availability == "out_of_stock"]
        if out_of_stock:
            print(f"Warning: Recommending out-of-stock items: {', '.join(out_of_stock)}")

        return self

# Usage
try:
    agent = EcommerceAgent(
        user_query="I'm looking for running shoes under $100",
        recommended_products=[
            Product(
                id="shoe-123",
                name="Runner Pro",
                price=89.99,
                availability="in_stock",
                shipping_weight=2.5
            ),
            Product(
                id="shoe-456",
                name="Speed Elite",
                price=79.99,
                availability="in_stock",
                shipping_weight=2.2
            )
        ],
        price_range={"min": 50, "max": 100}
    )
    print("Valid e-commerce agent:", agent)
except ValueError as e:
    print("Validation error:", e)
```

### 🏥 Healthcare Domain

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date

class MedicalInfo(BaseModel):
    condition: str
    severity: Literal["mild", "moderate", "severe"] = "moderate"
    symptoms: List[str] = []
    recommendations: List[str] = []

    @model_validator(mode='after')
    def validate_medical_info(self):
        """Validate medical information for consistency and safety."""
        # Check that severe conditions have appropriate recommendations
        if self.severity == "severe" and not any("consult" in r.lower() or "doctor" in r.lower() for r in self.recommendations):
            raise ValueError("Severe conditions must include recommendation to consult a doctor")

        # Check that recommendations are provided
        if not self.recommendations:
            raise ValueError("Medical information must include recommendations")

        # Add disclaimer if not present
        has_disclaimer = any("not medical advice" in r.lower() for r in self.recommendations)
        if not has_disclaimer:
            self.recommendations.append("This information is not medical advice. Please consult a healthcare professional.")

        return self

class HealthcareAgent(BaseModel):
    user_query: str
    response: str
    medical_info: Optional[MedicalInfo] = None
    contains_medical_advice: bool = False

    @model_validator(mode='after')
    def validate_healthcare_response(self):
        """Validate healthcare response for safety and compliance."""
        # Check if response contains medical advice
        medical_advice_indicators = [
            "you should take", "you need to", "I recommend", "take this medication",
            "this treatment", "this therapy", "this dose", "this drug"
        ]

        contains_advice = any(indicator in self.response.lower() for indicator in medical_advice_indicators)

        # If contains advice but not flagged
        if contains_advice and not self.contains_medical_advice:
            raise ValueError("Response contains medical advice but not flagged as such")

        # If flagged as medical advice, must include disclaimer
        if self.contains_medical_advice and "not a substitute for professional medical advice" not in self.response.lower():
            raise ValueError("Medical advice must include professional disclaimer")

        return self

# Usage
try:
    agent = HealthcareAgent(
        user_query="What should I do for a headache?",
        response="Headaches can have many causes. For mild headaches, rest, hydration, and over-the-counter pain relievers may help. This information is not a substitute for professional medical advice. If headaches are severe or persistent, please consult a healthcare provider.",
        medical_info=MedicalInfo(
            condition="headache",
            severity="mild",
            symptoms=["pain", "pressure", "sensitivity to light"],
            recommendations=[
                "Rest in a quiet, dark room",
                "Stay hydrated",
                "Over-the-counter pain relievers may help with symptoms",
                "This information is not medical advice. Please consult a healthcare professional."
            ]
        ),
        contains_medical_advice=False
    )
    print("Valid healthcare agent:", agent)
except ValueError as e:
    print("Validation error:", e)
```

### 💰 Finance Domain

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date
import re

class FinancialRecommendation(BaseModel):
    recommendation_type: Literal["investment", "savings", "budgeting", "debt", "general"]
    description: str
    risk_level: Optional[Literal["low", "medium", "high"]] = None
    time_horizon: Optional[Literal["short", "medium", "long"]] = None

    @model_validator(mode='after')
    def validate_financial_recommendation(self):
        """Validate financial recommendation for completeness and consistency."""
        # Investment recommendations must include risk level and time horizon
        if self.recommendation_type == "investment":
            if self.risk_level is None:
                raise ValueError("Investment recommendations must include risk level")
            if self.time_horizon is None:
                raise ValueError("Investment recommendations must include time horizon")

        # Check for disclaimer in description
        if "not financial advice" not in self.description.lower():
            raise ValueError("Financial recommendations must include disclaimer")

        return self

class FinancialAgent(BaseModel):
    user_query: str
    response: str
    recommendations: List[FinancialRecommendation] = []
    contains_specific_advice: bool = False

    @model_validator(mode='after')
    def validate_financial_response(self):
        """Validate financial response for compliance and safety."""
        # Check for specific financial advice indicators
        specific_advice_patterns = [
            r"you should invest in \w+",
            r"buy \w+ stock",
            r"sell \w+ stock",
            r"invest \d+% in",
        ]

        contains_specific = any(re.search(pattern, self.response.lower()) for pattern in specific_advice_patterns)

        # If contains specific advice but not flagged
        if contains_specific and not self.contains_specific_advice:
            raise ValueError("Response contains specific financial advice but not flagged as such")

        # If flagged as specific advice, must include strong disclaimer
        if self.contains_specific_advice and "not a substitute for professional financial advice" not in self.response.lower():
            raise ValueError("Specific financial advice must include professional disclaimer")

        return self

# Usage
try:
    agent = FinancialAgent(
        user_query="How should I save for retirement?",
        response="Retirement planning depends on many factors including your age, income, and goals. Generally, experts recommend saving 15% of income for retirement. Consider tax-advantaged accounts like 401(k)s and IRAs. This is not a substitute for professional financial advice.",
        recommendations=[
            FinancialRecommendation(
                recommendation_type="savings",
                description="Consider maximizing contributions to tax-advantaged retirement accounts. This is not financial advice.",
                time_horizon="long"
            ),
            FinancialRecommendation(
                recommendation_type="investment",
                description="Diversification across asset classes can help manage risk. This is not financial advice.",
                risk_level="medium",
                time_horizon="long"
            )
        ],
        contains_specific_advice=False
    )
    print("Valid financial agent:", agent)
except ValueError as e:
    print("Validation error:", e)
```

---

## 🤝 Multi-Agent System Validation

In systems with multiple agents, additional validation is needed:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
import uuid

class AgentMessage(BaseModel):
    agent_id: str
    message_type: Literal["request", "response", "notification"]
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    in_response_to: Optional[str] = None

    @model_validator(mode='after')
    def validate_message_structure(self):
        """Validate message structure based on type."""
        if self.message_type == "response" and self.in_response_to is None:
            raise ValueError("Response messages must reference the original request")

        return self

class MultiAgentSystem(BaseModel):
    agents: Dict[str, Dict[str, Any]]
    message_queue: List[AgentMessage] = []

    @model_validator(mode='after')
    def validate_message_flow(self):
        """Validate message flow between agents."""
        # Check that all referenced agents exist
        for message in self.message_queue:
            if message.agent_id not in self.agents:
                raise ValueError(f"Message references non-existent agent: {message.agent_id}")

            if message.message_type == "response" and message.in_response_to:
                # Find the original request
                request_messages = [m for m in self.message_queue if m.message_id == message.in_response_to]
                if not request_messages:
                    raise ValueError(f"Response references non-existent request: {message.in_response_to}")

        return self

    def add_message(self, message: AgentMessage):
        """Add a message to the queue with validation."""
        if message.agent_id not in self.agents:
            raise ValueError(f"Message from non-existent agent: {message.agent_id}")

        if message.message_type == "response" and message.in_response_to:
            # Validate that the response matches the request
            request_messages = [m for m in self.message_queue if m.message_id == message.in_response_to]
            if not request_messages:
                raise ValueError(f"Response references non-existent request: {message.in_response_to}")

        self.message_queue.append(message)

# Usage
try:
    system = MultiAgentSystem(
        agents={
            "search_agent": {"type": "knowledge", "capabilities": ["web_search", "knowledge_retrieval"]},
            "task_agent": {"type": "task", "capabilities": ["calendar", "reminder", "email"]}
        }
    )

    # Add a request message
    request = AgentMessage(
        agent_id="task_agent",
        message_type="request",
        content={"action": "create_calendar_event", "title": "Team Meeting", "time": "2023-12-25T10:00:00"}
    )
    system.add_message(request)

    # Add a response message
    response = AgentMessage(
        agent_id="task_agent",
        message_type="response",
        content={"status": "success", "event_id": "evt-123"},
        in_response_to=request.message_id
    )
    system.add_message(response)

    print("Valid multi-agent system with messages")
except ValueError as e:
    print("Validation error:", e)
```

---

## 💪 Practice Exercises

1. **Create a Customer Service Chatbot Validator**: Build a Pydantic model for validating a customer service chatbot that handles different types of customer inquiries (complaints, information requests, technical support).

2. **Implement a Healthcare Agent Validator**: Develop a validation system for a healthcare agent that ensures all medical information includes appropriate disclaimers and safety checks.

3. **Build a Multi-Agent Validation System**: Create a system that ensures proper communication between a search agent, a planning agent, and an execution agent.

4. **Design Financial Compliance Validators**: Implement domain-specific validators for a financial advisor agent that ensures compliance with regulatory requirements.

5. **Create an E-commerce Recommendation Validator**: Build a validation system for an e-commerce agent that validates product recommendations based on user preferences, inventory availability, and pricing constraints.

---

## 🔍 Key Concepts to Remember

1. **Agent-Specific Validation**: Different agent types require specialized validation approaches
2. **Domain-Specific Rules**: Each business domain has unique validation requirements
3. **Regulatory Compliance**: Validation ensures adherence to industry standards and regulations
4. **Multi-Agent Communication**: Inter-agent messaging requires specialized validation
5. **Custom Validators**: Domain-specific business rules can be enforced through custom validators
6. **Trust and Reliability**: Proper validation improves user trust and system reliability

---

## 📚 Additional Resources

- [Domain-Driven Design](https://martinfowler.com/tags/domain%20driven%20design.html)
- [Multi-Agent System Architecture](https://www.sciencedirect.com/topics/computer-science/multiagent-system)
- [Healthcare AI Compliance Guidelines](https://www.hhs.gov/hipaa/for-professionals/special-topics/artificial-intelligence/index.html)
- [Financial Services Compliance](https://www.finra.org/rules-guidance/key-topics/artificial-intelligence)
- [E-commerce Best Practices](https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides-what-people-are-asking)
- [Pydantic Domain Models](https://docs.pydantic.dev/latest/usage/models/)

---

## 🚀 Next Steps

In the next lesson, we'll explore how to integrate validation with LLM systems, focusing on connecting validation with LLM-generated content, handling validation in the context of uncertainty, and creating feedback loops between validation and LLM systems.

---

> 💡 **Note on LLM Integration**: When working with LLM-based agents in specific domains, validation becomes even more critical. LLMs may generate plausible-sounding but incorrect or non-compliant responses, especially in regulated domains like healthcare and finance. Domain-specific validators act as guardrails to ensure LLM outputs meet industry standards and regulatory requirements.

---

Happy coding! 🤖
