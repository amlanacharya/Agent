"""
Exercise 4.7.3: Knowledge Agent Validation
-----------------------------------------
This module implements validation patterns specific to knowledge agents, focusing on:
1. Source reliability validation
2. Answer relevance validation
3. Factual accuracy validation
4. Query understanding validation
5. Citation validation
6. Uncertainty handling validation

These validation patterns help ensure that knowledge agents provide accurate,
relevant, and properly sourced information while appropriately handling uncertainty.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Literal, Set, Any, Union
from enum import Enum
from datetime import datetime
import re
import uuid


class SourceType(str, Enum):
    """Types of information sources."""
    ACADEMIC = "academic"
    NEWS = "news"
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    SOCIAL_MEDIA = "social_media"
    ENCYCLOPEDIA = "encyclopedia"
    BLOG = "blog"
    BOOK = "book"
    WEBSITE = "website"
    OTHER = "other"


class FactCheckResult(str, Enum):
    """Results of fact checking."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    OUTDATED = "outdated"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """Model for an information source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    url: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source_type: SourceType = SourceType.OTHER
    reliability_score: float = Field(ge=0.0, le=1.0)
    peer_reviewed: bool = False

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format if provided."""
        if v is not None:
            # Simple URL validation
            if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', v):
                raise ValueError("Invalid URL format")
        return v

    @model_validator(mode='after')
    def validate_source_consistency(self):
        """Validate source consistency based on source type."""
        # Academic sources should have high reliability and be peer-reviewed
        if self.source_type == SourceType.ACADEMIC and self.reliability_score < 0.7:
            print(f"Warning: Academic source '{self.name}' has low reliability score: {self.reliability_score}")

        if self.source_type == SourceType.ACADEMIC and not self.peer_reviewed:
            print(f"Warning: Academic source '{self.name}' is not marked as peer-reviewed")

        # Check for outdated sources
        if self.last_updated:
            days_since_update = (datetime.now() - self.last_updated).days
            if days_since_update > 365 * 2:  # Older than 2 years
                print(f"Warning: Source '{self.name}' is significantly outdated (last updated {days_since_update} days ago)")

        return self


class Citation(BaseModel):
    """Model for a citation."""
    source_id: str
    text: str
    page_number: Optional[int] = None
    quote: Optional[str] = None
    url_fragment: Optional[str] = None

    @model_validator(mode='after')
    def validate_citation(self):
        """Validate citation completeness."""
        # Either quote or page number should be provided for proper citation
        if not self.quote and not self.page_number and not self.url_fragment:
            print("Warning: Citation lacks specific reference (quote, page number, or URL fragment)")

        return self


class KnowledgeQuery(BaseModel):
    """Model for a knowledge query."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    query_type: Literal["factual", "conceptual", "procedural", "causal", "comparative", "other"] = "factual"
    entities: List[str] = []
    keywords: List[str] = []
    context: Optional[Dict[str, Any]] = None

    @field_validator('text')
    @classmethod
    def validate_query_text(cls, v):
        """Validate query text."""
        if len(v.strip()) < 3:
            raise ValueError("Query text is too short")
        return v

    @model_validator(mode='after')
    def extract_query_metadata(self):
        """Extract metadata from query if not provided."""
        # Extract entities and keywords if not provided
        # This is a simplified implementation - in a real system, this would use NLP
        if not self.entities or not self.keywords:
            # Simple keyword extraction
            stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "is", "are", "was", "were"}
            words = [word.lower() for word in re.findall(r'\b\w+\b', self.text)]
            potential_keywords = [word for word in words if word not in stop_words and len(word) > 3]

            if not self.keywords and potential_keywords:
                self.keywords = potential_keywords[:5]  # Take up to 5 keywords

            # Simple entity extraction (capitalized words)
            if not self.entities:
                potential_entities = re.findall(r'\b[A-Z][a-z]+\b', self.text)
                if potential_entities:
                    self.entities = potential_entities

        # Determine query type if not specified
        if self.query_type == "other":
            # Simple heuristics for query type
            if any(word in self.text.lower() for word in ["what is", "who is", "where is", "when"]):
                self.query_type = "factual"
            elif any(word in self.text.lower() for word in ["how to", "steps", "procedure"]):
                self.query_type = "procedural"
            elif any(word in self.text.lower() for word in ["why", "cause", "effect", "because"]):
                self.query_type = "causal"
            elif any(word in self.text.lower() for word in ["compare", "difference", "versus", "vs"]):
                self.query_type = "comparative"
            elif any(word in self.text.lower() for word in ["explain", "concept", "theory"]):
                self.query_type = "conceptual"

        return self


class KnowledgeResponse(BaseModel):
    """Model for a knowledge response."""
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

    @model_validator(mode='after')
    def validate_source_reliability(self):
        """Validate that sources are reliable enough for the given confidence."""
        if self.sources:
            avg_reliability = sum(s.reliability_score for s in self.sources) / len(self.sources)

            # High confidence requires reliable sources
            if avg_reliability < 0.7 and self.confidence > 0.8:
                raise ValueError("High confidence with low reliability sources")

            # Check for outdated sources
            current_time = datetime.now()
            outdated_sources = []
            for source in self.sources:
                if source.last_updated:
                    days_old = (current_time - source.last_updated).days
                    if days_old > 365 and self.confidence > 0.9:  # Older than a year
                        outdated_sources.append(source.name)

            if outdated_sources and self.confidence > 0.9:
                raise ValueError(f"High confidence with outdated sources: {', '.join(outdated_sources)}")

        # If no sources but high confidence, that's a problem
        elif not self.sources and self.confidence > 0.8:
            raise ValueError("High confidence without any sources")

        return self

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

        return self

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

        # Check that answer text contains citation markers
        # This is a simplified check; real implementation would be more sophisticated
        if self.citations and not any(re.search(r'\[\d+\]|\(\d+\)', self.answer) for _ in self.citations):
            print("Warning: Answer may not properly reference citations in text")

        return self

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

        # If low confidence but no uncertainty markers
        if self.confidence < 0.6 and not contains_uncertainty_markers and not self.contains_uncertainty:
            raise ValueError("Low confidence answer should express uncertainty")

        return self

    @model_validator(mode='after')
    def validate_opinion_handling(self):
        """Validate that opinions are properly handled."""
        # Check for opinion markers in text
        opinion_phrases = [
            "in my opinion", "i believe", "i think", "arguably", "subjectively",
            "from my perspective", "it seems", "appears to be"
        ]

        contains_opinion_markers = any(phrase in self.answer.lower() for phrase in opinion_phrases)

        # If text contains opinion markers but not flagged
        if contains_opinion_markers and not self.contains_opinion:
            raise ValueError("Answer contains opinion markers but not flagged as opinion")

        # If flagged as opinion but high confidence without sources
        if self.contains_opinion and self.confidence > 0.9 and not self.sources:
            raise ValueError("High confidence opinion without sources")

        # If fact check result is opinion but not flagged
        if self.fact_check_result == FactCheckResult.OPINION and not self.contains_opinion:
            raise ValueError("Fact check indicates opinion but not flagged as opinion")

        return self


class KnowledgeAgent(BaseModel):
    """Model for a knowledge agent."""
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

    @model_validator(mode='after')
    def validate_agent_configuration(self):
        """Validate agent configuration for consistency."""
        # Check that confidence threshold is appropriate
        if self.confidence_threshold < 0.5:
            print("Warning: Low confidence threshold may result in unreliable answers")

        if self.confidence_threshold > 0.9:
            print("Warning: Very high confidence threshold may result in too many 'I don't know' responses")

        # Check that source reliability threshold is appropriate
        if self.source_reliability_threshold < 0.5:
            print("Warning: Low source reliability threshold may result in using unreliable sources")

        # Check that agent handles uncertainty if confidence threshold is high
        if self.confidence_threshold > 0.8 and not self.handles_uncertainty:
            raise ValueError("Agents with high confidence threshold should handle uncertainty")

        return self

    def validate_response(self, response: KnowledgeResponse) -> List[str]:
        """
        Validate a knowledge response against agent configuration.

        Args:
            response: The knowledge response to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check confidence against threshold
        if response.confidence < self.confidence_threshold:
            errors.append(f"Response confidence ({response.confidence}) below agent threshold ({self.confidence_threshold})")

        # Check source reliability
        if response.sources:
            for source in response.sources:
                if source.reliability_score < self.source_reliability_threshold:
                    errors.append(f"Source '{source.name}' reliability ({source.reliability_score}) below threshold ({self.source_reliability_threshold})")

                # Check source age if max_source_age_days is set
                if self.max_source_age_days and source.last_updated:
                    days_old = (datetime.now() - source.last_updated).days
                    if days_old > self.max_source_age_days:
                        errors.append(f"Source '{source.name}' is too old ({days_old} days)")

        # Check citations if required
        if self.requires_citations and not response.citations:
            errors.append("Citations required but not provided")

        # Extract query type from the response text
        # In a real implementation, this would use NLP to determine the query type
        # For this example, we'll use the query_text to infer the query type
        query_text = response.query_text.lower()

        # Simple heuristic to determine query type
        if query_text.startswith("how to") or query_text.startswith("how do i"):
            inferred_query_type = "procedural"
        elif "compare" in query_text or "difference between" in query_text:
            inferred_query_type = "comparative"
        elif query_text.startswith("why") or "cause" in query_text:
            inferred_query_type = "causal"
        elif query_text.startswith("what is") or query_text.startswith("explain"):
            inferred_query_type = "conceptual"
        else:
            inferred_query_type = "factual"

        # Check if the inferred query type is supported
        if inferred_query_type not in self.supported_query_types:
            errors.append(f"Query type '{inferred_query_type}' not supported by this agent")

        return errors


class KnowledgeAgentValidator:
    """Validator for knowledge agents and their responses."""

    @staticmethod
    def validate_query(query: KnowledgeQuery) -> List[str]:
        """
        Validate a knowledge query.

        Args:
            query: The query to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check query length
        if len(query.text.strip()) < 5:
            errors.append("Query is too short")

        # Check for query clarity
        if len(query.text.split()) < 3:
            errors.append("Query may be too vague")

        return errors

    @staticmethod
    def validate_source(source: Source) -> List[str]:
        """
        Validate an information source.

        Args:
            source: The source to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for required fields based on source type
        if source.source_type == SourceType.ACADEMIC:
            if not source.peer_reviewed:
                errors.append(f"Academic source '{source.name}' should be peer-reviewed")
            if not source.author:
                errors.append(f"Academic source '{source.name}' should have an author")

        # Check for URL for online sources
        if source.source_type in [SourceType.NEWS, SourceType.WEBSITE, SourceType.BLOG] and not source.url:
            errors.append(f"{source.source_type.value.capitalize()} source '{source.name}' should have a URL")

        # Check for recency for news sources
        if source.source_type == SourceType.NEWS and source.publication_date:
            days_old = (datetime.now() - source.publication_date).days
            if days_old > 30:
                errors.append(f"News source '{source.name}' is {days_old} days old")

        return errors

    @staticmethod
    def validate_response_for_query_type(response: KnowledgeResponse, query: KnowledgeQuery) -> List[str]:
        """
        Validate that a response is appropriate for the query type.

        Args:
            response: The response to validate
            query: The original query

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Different query types have different validation requirements
        if query.query_type == "factual":
            # Factual queries should have high confidence and reliable sources
            if response.confidence < 0.7:
                errors.append("Factual query response has low confidence")
            if not response.sources:
                errors.append("Factual query response should have sources")

        elif query.query_type == "procedural":
            # Procedural queries should have step-by-step instructions
            pattern = r'(step|first|second|third|\d+\.)'
            if not re.search(pattern, response.answer, re.IGNORECASE):
                errors.append("Procedural query response should contain steps")

        elif query.query_type == "comparative":
            # Comparative queries should mention both entities being compared
            if len(query.entities) >= 2:
                entity_mentions = sum(1 for entity in query.entities if entity.lower() in response.answer.lower())
                if entity_mentions < len(query.entities):
                    errors.append("Comparative query response should mention all entities being compared")

        return errors

    @staticmethod
    def validate_fact_check(response: KnowledgeResponse) -> List[str]:
        """
        Validate fact check results against confidence.

        Args:
            response: The response to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check fact check result against confidence
        if response.fact_check_result == FactCheckResult.VERIFIED and response.confidence < 0.8:
            errors.append("Verified fact should have high confidence")

        if response.fact_check_result == FactCheckResult.CONTRADICTED and response.confidence > 0.5:
            errors.append("Contradicted fact should have low confidence")

        if response.fact_check_result == FactCheckResult.UNVERIFIED and response.confidence > 0.7:
            errors.append("Unverified fact should not have high confidence")

        if response.fact_check_result == FactCheckResult.OPINION and not response.contains_opinion:
            errors.append("Opinion fact check should be marked as containing opinion")

        return errors


# Example usage
if __name__ == "__main__":
    # Create a knowledge agent
    agent = KnowledgeAgent(
        name="FactBot",
        description="A knowledge agent for factual queries",
        knowledge_domains=["general", "science", "history"],
        supported_query_types=["factual", "conceptual"],
        confidence_threshold=0.7,
        source_reliability_threshold=0.6,
        requires_citations=True
    )

    # Create a source
    source = Source(
        name="Encyclopedia Britannica",
        url="https://www.britannica.com/place/Paris",
        source_type=SourceType.ENCYCLOPEDIA,
        reliability_score=0.9,
        last_updated=datetime(2023, 1, 15)
    )

    # Create a citation
    citation = Citation(
        source_id=source.id,
        text="Encyclopedia Britannica - Paris",
        url_fragment="history"
    )

    # Create a query
    query = KnowledgeQuery(
        text="What is the capital of France?",
        query_type="factual",
        entities=["France"],
        keywords=["capital", "France"]
    )

    # Create a valid response
    try:
        response = KnowledgeResponse(
            query_id=query.id,
            query_text=query.text,
            answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
            confidence=0.95,
            sources=[source],
            citations=[citation],
            fact_check_result=FactCheckResult.VERIFIED
        )
        print("Valid knowledge response created")

        # Validate response against agent configuration
        errors = agent.validate_response(response)
        if errors:
            print("Agent validation errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Response is valid for this agent")

    except ValueError as e:
        print("Validation error:", e)

    # Create an invalid response (high confidence without sources)
    try:
        invalid_response = KnowledgeResponse(
            query_id=query.id,
            query_text=query.text,
            answer="The capital of France is Paris.",
            confidence=0.95,
            sources=[],  # No sources
            citations=[]  # No citations
        )
        print("Invalid response created successfully")
    except ValueError as e:
        print("Validation error (expected):", e)
