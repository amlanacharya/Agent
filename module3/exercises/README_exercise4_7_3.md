# Exercise 4.7.3: Knowledge Agent Validation

This exercise implements validation patterns specific to knowledge agents, focusing on ensuring that knowledge agents provide accurate, relevant, and properly sourced information while appropriately handling uncertainty.

## Overview

Knowledge agents focus on information retrieval and require specialized validation for:

1. **Source Reliability**: Ensuring information comes from reliable sources
2. **Answer Relevance**: Validating that responses accurately answer the query
3. **Factual Accuracy**: Verifying the factual correctness of information
4. **Query Understanding**: Validating that the query is properly interpreted
5. **Citation Validation**: Ensuring proper citation of sources
6. **Uncertainty Handling**: Appropriately expressing uncertainty when confidence is low

## Implementation

The implementation includes several Pydantic models:

### Source Model

```python
class Source(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    url: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source_type: SourceType = SourceType.OTHER
    reliability_score: float = Field(ge=0.0, le=1.0)
    peer_reviewed: bool = False
```

### KnowledgeQuery Model

```python
class KnowledgeQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    query_type: Literal["factual", "conceptual", "procedural", "causal", "comparative", "other"] = "factual"
    entities: List[str] = []
    keywords: List[str] = []
    context: Optional[Dict[str, Any]] = None
```

### KnowledgeResponse Model

```python
class KnowledgeResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    query_text: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[Source] = []
    citations: List[Citation] = []
    fact_check_result: FactCheckResult = FactCheckResult.UNKNOWN
    generated_at: datetime = Field(default_factory=datetime.now)
    contains_uncertainty: bool = False
    contains_opinion: bool = False
```

### KnowledgeAgent Model

```python
class KnowledgeAgent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    knowledge_domains: List[str] = []
    supported_query_types: List[str] = ["factual", "conceptual", "procedural", "causal", "comparative"]
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    source_reliability_threshold: float = Field(ge=0.0, le=1.0, default=0.6)
    max_source_age_days: Optional[int] = 365 * 2  # 2 years
    requires_citations: bool = True
    handles_uncertainty: bool = True
```

## Validation Patterns

The implementation includes several validation patterns:

### Source Reliability Validation

```python
@model_validator(mode='after')
def validate_source_reliability(self):
    """Validate that sources are reliable enough for the given confidence."""
    if self.sources:
        avg_reliability = sum(s.reliability_score for s in self.sources) / len(self.sources)
        
        # High confidence requires reliable sources
        if avg_reliability < 0.7 and self.confidence > 0.8:
            raise ValueError("High confidence with low reliability sources")
```

### Answer Relevance Validation

```python
@model_validator(mode='after')
def validate_answer_relevance(self):
    """Validate that the answer is relevant to the query."""
    # This is a simplified check; real implementation would use NLP
    query_keywords = set(self.query_text.lower().split())
    answer_keywords = set(self.answer.lower().split())
    
    # Remove common words
    common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "is", "are", "was", "were"}
    query_keywords = query_keywords - common_words
    
    # Check if any keywords from query appear in answer
    if not query_keywords.intersection(answer_keywords) and self.confidence > 0.7:
        print("Warning: Answer may not be relevant to query")
```

### Citation Validation

```python
@model_validator(mode='after')
def validate_citations(self):
    """Validate that citations match sources and are properly used."""
    # Check that all citations reference valid sources
    source_ids = {source.id for source in self.sources}
    invalid_citations = [citation for citation in self.citations if citation.source_id not in source_ids]
    
    if invalid_citations:
        raise ValueError(f"Citations reference non-existent sources: {[c.source_id for c in invalid_citations]}")
    
    # Check that high-confidence answers have citations
    if self.confidence > 0.8 and not self.citations:
        raise ValueError("High confidence answer requires citations")
```

### Uncertainty Handling Validation

```python
@model_validator(mode='after')
def validate_uncertainty_handling(self):
    """Validate that uncertainty is properly handled."""
    # Check for uncertainty markers in text
    uncertainty_phrases = [
        "may be", "might be", "could be", "possibly", "perhaps", "uncertain",
        "not clear", "not known", "debated", "controversial"
    ]
    
    contains_uncertainty_markers = any(phrase in self.answer.lower() for phrase in uncertainty_phrases)
    
    # If text contains uncertainty markers but not flagged
    if contains_uncertainty_markers and not self.contains_uncertainty:
        raise ValueError("Answer contains uncertainty markers but not flagged as uncertain")
```

## Usage Examples

### Creating a Knowledge Agent

```python
agent = KnowledgeAgent(
    name="FactBot",
    description="A knowledge agent for factual queries",
    knowledge_domains=["general", "science", "history"],
    supported_query_types=["factual", "conceptual"],
    confidence_threshold=0.7,
    source_reliability_threshold=0.6,
    requires_citations=True
)
```

### Creating a Valid Response

```python
response = KnowledgeResponse(
    query_id=query.id,
    query_text=query.text,
    answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
    confidence=0.95,
    sources=[source],
    citations=[citation],
    fact_check_result=FactCheckResult.VERIFIED
)
```

### Validating a Response Against Agent Configuration

```python
errors = agent.validate_response(response)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Response is valid for this agent")
```

## Running the Demo

To run the full demo:

```bash
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator
```

To run a specific part of the demo:

```bash
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator source
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator query
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator response
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator agent
python -m module3.exercises.demo_exercise4_7_3_knowledge_agent_validator factcheck
```

## Running the Tests

To run the tests:

```bash
python -m module3.exercises.test_exercise4_7_3_knowledge_agent_validator
```

## Key Concepts

1. **Source Reliability**: Knowledge agents must validate the reliability of their information sources
2. **Query Understanding**: Different query types require different validation approaches
3. **Citation Management**: Proper citation of sources is essential for knowledge agents
4. **Uncertainty Handling**: Knowledge agents must appropriately express uncertainty when confidence is low
5. **Fact Checking**: Knowledge agents should verify facts before presenting them with high confidence
6. **Opinion Handling**: Knowledge agents must clearly distinguish between facts and opinions
