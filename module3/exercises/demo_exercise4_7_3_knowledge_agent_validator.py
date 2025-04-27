"""
Demo Script for Exercise 4.7.3: Knowledge Agent Validation
--------------------------------------------------------
This script demonstrates the validation patterns for knowledge agents implemented in
exercise4_7_3_knowledge_agent_validator.py.
"""

from datetime import datetime, timedelta
import sys
# Adjust the import path based on how you're running the script
try:
    # When running from the module3/exercises directory
    from exercise4_7_3_knowledge_agent_validator import (
        Source, Citation, KnowledgeQuery, KnowledgeResponse, KnowledgeAgent,
        KnowledgeAgentValidator, SourceType, FactCheckResult
    )
except ImportError:
    # When running from the project root
    from module3.exercises.exercise4_7_3_knowledge_agent_validator import (
        Source, Citation, KnowledgeQuery, KnowledgeResponse, KnowledgeAgent,
        KnowledgeAgentValidator, SourceType, FactCheckResult
    )


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subheader."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)


def print_source_info(source):
    """Print source information."""
    print(f"Source: {source.name}")
    print(f"  Type: {source.source_type.value}")
    print(f"  URL: {source.url}")
    print(f"  Reliability: {source.reliability_score:.2f}")
    if source.last_updated:
        print(f"  Last Updated: {source.last_updated.strftime('%Y-%m-%d')}")
    print(f"  Peer Reviewed: {source.peer_reviewed}")


def print_query_info(query):
    """Print query information."""
    print(f"Query: {query.text}")
    print(f"  Type: {query.query_type}")
    print(f"  Entities: {', '.join(query.entities) if query.entities else 'None'}")
    print(f"  Keywords: {', '.join(query.keywords) if query.keywords else 'None'}")


def print_response_info(response):
    """Print response information."""
    print(f"Response to: {response.query_text}")
    print(f"  Answer: {response.answer}")
    print(f"  Confidence: {response.confidence:.2f}")
    print(f"  Fact Check: {response.fact_check_result.value}")
    print(f"  Contains Uncertainty: {response.contains_uncertainty}")
    print(f"  Contains Opinion: {response.contains_opinion}")
    print(f"  Sources: {len(response.sources)}")
    print(f"  Citations: {len(response.citations)}")


def print_agent_info(agent):
    """Print agent information."""
    print(f"Agent: {agent.name}")
    print(f"  Description: {agent.description}")
    print(f"  Knowledge Domains: {', '.join(agent.knowledge_domains)}")
    print(f"  Supported Query Types: {', '.join(agent.supported_query_types)}")
    print(f"  Confidence Threshold: {agent.confidence_threshold:.2f}")
    print(f"  Source Reliability Threshold: {agent.source_reliability_threshold:.2f}")
    print(f"  Requires Citations: {agent.requires_citations}")
    print(f"  Handles Uncertainty: {agent.handles_uncertainty}")


def print_validation_errors(errors):
    """Print validation errors."""
    if not errors:
        print("✅ No validation errors")
    else:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")


def demo_source_validation():
    """Demonstrate source validation."""
    print_header("Source Validation Demo")

    validator = KnowledgeAgentValidator()

    print_subheader("Valid Academic Source")

    valid_academic = Source(
        name="Nature Journal",
        url="https://www.nature.com/articles/s41586-021-03275-y",
        author="Smith et al.",
        publication_date=datetime.now() - timedelta(days=60),
        source_type=SourceType.ACADEMIC,
        reliability_score=0.95,
        peer_reviewed=True
    )

    print_source_info(valid_academic)

    errors = validator.validate_source(valid_academic)
    print("\nValidation result:")
    print_validation_errors(errors)

    print_subheader("Invalid Academic Source (Not Peer-Reviewed)")

    invalid_academic = Source(
        name="Unpublished Paper",
        url="https://example.com/paper",
        author="Smith et al.",
        source_type=SourceType.ACADEMIC,
        reliability_score=0.7,
        peer_reviewed=False  # Not peer-reviewed
    )

    print_source_info(invalid_academic)

    errors = validator.validate_source(invalid_academic)
    print("\nValidation result:")
    print_validation_errors(errors)

    print_subheader("Invalid News Source (No URL)")

    invalid_news = Source(
        name="Daily News",
        url=None,  # Missing URL
        source_type=SourceType.NEWS,
        reliability_score=0.6
    )

    print_source_info(invalid_news)

    errors = validator.validate_source(invalid_news)
    print("\nValidation result:")
    print_validation_errors(errors)


def demo_query_validation():
    """Demonstrate query validation."""
    print_header("Query Validation Demo")

    validator = KnowledgeAgentValidator()

    print_subheader("Valid Query")

    valid_query = KnowledgeQuery(
        text="What are the effects of climate change on polar ice caps?",
        query_type="causal",
        entities=["climate change", "polar ice caps"],
        keywords=["effects", "climate", "change", "polar", "ice", "caps"]
    )

    print_query_info(valid_query)

    errors = validator.validate_query(valid_query)
    print("\nValidation result:")
    print_validation_errors(errors)

    print_subheader("Invalid Query (Too Short)")

    invalid_query = KnowledgeQuery(
        text="Why?",
        query_type="causal"
    )

    print_query_info(invalid_query)

    errors = validator.validate_query(invalid_query)
    print("\nValidation result:")
    print_validation_errors(errors)

    print_subheader("Query with Automatic Metadata Extraction")

    auto_query = KnowledgeQuery(
        text="What is the impact of artificial intelligence on job markets?",
        query_type="other"  # Will be automatically determined
    )

    print("Before metadata extraction:")
    print_query_info(auto_query)

    # Extract metadata
    auto_query.extract_query_metadata()

    print("\nAfter metadata extraction:")
    print_query_info(auto_query)


def demo_response_validation():
    """Demonstrate response validation."""
    print_header("Response Validation Demo")

    # Create a source
    source = Source(
        name="Encyclopedia Britannica",
        url="https://www.britannica.com/place/Paris",
        source_type=SourceType.ENCYCLOPEDIA,
        reliability_score=0.9,
        last_updated=datetime.now() - timedelta(days=30)
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

    print_subheader("Valid Response")

    try:
        valid_response = KnowledgeResponse(
            query_id=query.id,
            query_text=query.text,
            answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
            confidence=0.95,
            sources=[source],
            citations=[citation],
            fact_check_result=FactCheckResult.VERIFIED
        )

        print_response_info(valid_response)
        print("\n✅ Response passed validation")
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Invalid Response (High Confidence without Sources)")

    try:
        invalid_response = KnowledgeResponse(
            query_id=query.id,
            query_text=query.text,
            answer="The capital of France is Paris.",
            confidence=0.95,
            sources=[],  # No sources
            citations=[]  # No citations
        )

        print_response_info(invalid_response)
        print("\n✅ Response passed validation")
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")

    print_subheader("Response with Uncertainty")

    try:
        uncertain_response = KnowledgeResponse(
            query_id=query.id,
            query_text="When exactly did humans first arrive in the Americas?",
            answer="The exact timing of human arrival in the Americas is still debated, but evidence suggests it may have been between 15,000 and 30,000 years ago [1].",
            confidence=0.6,
            sources=[source],
            citations=[citation],
            contains_uncertainty=True
        )

        print_response_info(uncertain_response)
        print("\n✅ Response passed validation")
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")


def demo_agent_validation():
    """Demonstrate agent validation of responses."""
    print_header("Agent Validation Demo")

    # Create an agent
    agent = KnowledgeAgent(
        name="FactBot",
        description="A knowledge agent for factual queries",
        knowledge_domains=["general", "science", "history"],
        supported_query_types=["factual", "conceptual"],
        confidence_threshold=0.7,
        source_reliability_threshold=0.6,
        requires_citations=True
    )

    print_agent_info(agent)

    # Create a source
    source = Source(
        name="Encyclopedia Britannica",
        url="https://www.britannica.com/place/Paris",
        source_type=SourceType.ENCYCLOPEDIA,
        reliability_score=0.9,
        last_updated=datetime.now() - timedelta(days=30)
    )

    # Create a citation
    citation = Citation(
        source_id=source.id,
        text="Encyclopedia Britannica - Paris",
        url_fragment="history"
    )

    print_subheader("Valid Response for Agent")

    # Create a query
    factual_query = KnowledgeQuery(
        text="What is the capital of France?",
        query_type="factual",
        entities=["France"],
        keywords=["capital", "France"]
    )

    # Create a response
    factual_response = KnowledgeResponse(
        query_id=factual_query.id,
        query_text=factual_query.text,
        answer="The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower [1].",
        confidence=0.95,
        sources=[source],
        citations=[citation],
        fact_check_result=FactCheckResult.VERIFIED
    )

    print_query_info(factual_query)
    print("\nResponse:")
    print_response_info(factual_response)

    errors = agent.validate_response(factual_response)
    print("\nAgent validation result:")
    print_validation_errors(errors)

    print_subheader("Invalid Response for Agent (Unsupported Query Type)")

    # Create a procedural query
    procedural_query = KnowledgeQuery(
        text="How do I make a French omelette?",
        query_type="procedural"
    )

    # Create a response
    procedural_response = KnowledgeResponse(
        query_id=procedural_query.id,
        query_text=procedural_query.text,
        answer="To make a French omelette: 1. Beat eggs. 2. Heat butter. 3. Cook eggs. 4. Fold and serve.",
        confidence=0.8,
        sources=[source],
        citations=[citation]
    )

    print_query_info(procedural_query)
    print("\nResponse:")
    print_response_info(procedural_response)

    errors = agent.validate_response(procedural_response)
    print("\nAgent validation result:")
    print_validation_errors(errors)


def demo_fact_check_validation():
    """Demonstrate fact check validation."""
    print_header("Fact Check Validation Demo")

    validator = KnowledgeAgentValidator()

    # Create a source
    source = Source(
        name="Encyclopedia Britannica",
        url="https://www.britannica.com/place/Paris",
        source_type=SourceType.ENCYCLOPEDIA,
        reliability_score=0.9,
        last_updated=datetime.now() - timedelta(days=30)
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
        query_type="factual"
    )

    print_subheader("Verified Fact")

    verified_response = KnowledgeResponse(
        query_id=query.id,
        query_text=query.text,
        answer="The capital of France is Paris.",
        confidence=0.95,
        sources=[source],
        citations=[citation],
        fact_check_result=FactCheckResult.VERIFIED
    )

    print_response_info(verified_response)

    errors = validator.validate_fact_check(verified_response)
    print("\nFact check validation result:")
    print_validation_errors(errors)

    print_subheader("Contradicted Fact with High Confidence")

    contradicted_response = KnowledgeResponse(
        query_id=query.id,
        query_text=query.text,
        answer="The capital of France is Lyon.",
        confidence=0.8,  # High confidence
        sources=[source],
        citations=[citation],
        fact_check_result=FactCheckResult.CONTRADICTED  # Contradicted
    )

    print_response_info(contradicted_response)

    errors = validator.validate_fact_check(contradicted_response)
    print("\nFact check validation result:")
    print_validation_errors(errors)

    print_subheader("Opinion Fact")

    opinion_response = KnowledgeResponse(
        query_id=query.id,
        query_text="What is the best city in France?",
        answer="In my opinion, Paris is the best city in France due to its rich history, culture, and architecture.",
        confidence=0.7,
        sources=[source],
        citations=[citation],
        fact_check_result=FactCheckResult.OPINION,
        contains_opinion=True
    )

    print_response_info(opinion_response)

    errors = validator.validate_fact_check(opinion_response)
    print("\nFact check validation result:")
    print_validation_errors(errors)


def main():
    """Main function to run the demo."""
    print_header("Knowledge Agent Validation Demo")

    print("""
This demo showcases validation patterns specific to knowledge agents, including:

1. Source reliability validation
2. Query validation
3. Response validation for different query types
4. Agent configuration validation
5. Fact check validation
6. Uncertainty and opinion handling

These validation patterns help ensure that knowledge agents provide accurate,
relevant, and properly sourced information while appropriately handling uncertainty.
""")

    # Check if a specific demo was requested
    if len(sys.argv) > 1:
        demo = sys.argv[1].lower()
        if demo == "source":
            demo_source_validation()
        elif demo == "query":
            demo_query_validation()
        elif demo == "response":
            demo_response_validation()
        elif demo == "agent":
            demo_agent_validation()
        elif demo == "factcheck":
            demo_fact_check_validation()
        else:
            print(f"Unknown demo: {demo}")
            print("Available demos: source, query, response, agent, factcheck")
    else:
        # Run all demos
        demo_source_validation()
        demo_query_validation()
        demo_response_validation()
        demo_agent_validation()
        demo_fact_check_validation()


if __name__ == "__main__":
    main()
