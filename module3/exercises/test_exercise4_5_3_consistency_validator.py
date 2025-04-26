"""
Test file for Exercise 4.5.3: Consistency Validator
"""

import unittest
from datetime import datetime, timedelta
from exercise4_5_3_consistency_validator import (
    FactType, FactSource, InconsistencyType, ContextualFact,
    ConversationMessage, ConversationContext, ConversationHistory,
    InconsistencyReport, ConsistencyValidator
)


class TestConsistencyValidator(unittest.TestCase):
    """Test cases for consistency validator."""

    def setUp(self):
        """Set up test fixtures."""
        self.context = ConversationContext()
        
        # Add some facts to the context
        self.context.add_fact(
            key="location.weather",
            value="New York",
            fact_type=FactType.LOCATION,
            source=FactSource.USER,
            related_entities=["weather"]
        )
        
        self.context.add_fact(
            key="attribute.weather",
            value="sunny",
            fact_type=FactType.STATE,
            source=FactSource.SYSTEM,
            related_entities=["weather"]
        )
        
        self.context.add_fact(
            key="numerical.temperature",
            value=75,
            fact_type=FactType.NUMERICAL,
            source=FactSource.SYSTEM,
            related_entities=["temperature"]
        )
        
        self.context.add_fact(
            key="temporal.meeting",
            value="tomorrow",
            fact_type=FactType.TEMPORAL,
            source=FactSource.USER,
            related_entities=["meeting"]
        )

    def test_consistent_response(self):
        """Test validation of a consistent response."""
        consistent_response = "The weather in New York is sunny with a temperature of 75 degrees. Don't forget your meeting tomorrow."
        validator = ConsistencyValidator(
            response=consistent_response,
            context=self.context
        )
        
        # Should not raise any exceptions
        inconsistencies = validator.check_for_inconsistencies()
        self.assertEqual(len(inconsistencies), 0)

    def test_inconsistent_location(self):
        """Test detection of inconsistent location."""
        inconsistent_location = "The weather in Boston is sunny with a temperature of 75 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_location,
            context=self.context
        )
        
        # Should detect location inconsistency
        inconsistencies = validator.check_for_inconsistencies()
        self.assertGreater(len(inconsistencies), 0)
        
        # Check that at least one inconsistency is a contradiction
        contradiction_inconsistencies = [inc for inc in inconsistencies if inc.inconsistency_type == InconsistencyType.CONTRADICTION]
        self.assertGreater(len(contradiction_inconsistencies), 0)
        
        # Check that the inconsistency mentions the correct entities
        self.assertIn("weather", contradiction_inconsistencies[0].description)
        self.assertIn("Boston", contradiction_inconsistencies[0].description)
        self.assertIn("New York", contradiction_inconsistencies[0].description)
        
        # Should raise ValueError when validating
        with self.assertRaises(ValueError):
            validator.validate_consistency()

    def test_inconsistent_state(self):
        """Test detection of inconsistent state."""
        inconsistent_state = "The weather in New York is rainy with a temperature of 75 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_state,
            context=self.context
        )
        
        # Should detect state inconsistency
        inconsistencies = validator.check_for_inconsistencies()
        self.assertGreater(len(inconsistencies), 0)
        
        # Should raise ValueError when validating
        with self.assertRaises(ValueError):
            validator.validate_consistency()

    def test_inconsistent_numerical(self):
        """Test detection of inconsistent numerical values."""
        inconsistent_numerical = "The weather in New York is sunny with a temperature of 60 degrees."
        validator = ConsistencyValidator(
            response=inconsistent_numerical,
            context=self.context
        )
        
        # Should detect numerical inconsistency
        inconsistencies = validator.check_for_inconsistencies()
        self.assertGreater(len(inconsistencies), 0)
        
        # Check that at least one inconsistency is numerical
        numerical_inconsistencies = [inc for inc in inconsistencies if inc.inconsistency_type == InconsistencyType.NUMERICAL]
        self.assertGreater(len(numerical_inconsistencies), 0)
        
        # Check that the inconsistency mentions the correct values
        self.assertIn("temperature", numerical_inconsistencies[0].description)
        self.assertIn("60", numerical_inconsistencies[0].description)
        self.assertIn("75", numerical_inconsistencies[0].description)
        
        # Should raise ValueError when validating
        with self.assertRaises(ValueError):
            validator.validate_consistency()

    def test_inconsistent_temporal(self):
        """Test detection of inconsistent temporal references."""
        inconsistent_temporal = "The weather in New York is sunny with a temperature of 75 degrees. Don't forget your meeting today."
        validator = ConsistencyValidator(
            response=inconsistent_temporal,
            context=self.context
        )
        
        # Should detect temporal inconsistency
        inconsistencies = validator.check_for_inconsistencies()
        self.assertGreater(len(inconsistencies), 0)
        
        # Check that at least one inconsistency is temporal
        temporal_inconsistencies = [inc for inc in inconsistencies if inc.inconsistency_type == InconsistencyType.TEMPORAL]
        self.assertGreater(len(temporal_inconsistencies), 0)
        
        # Check that the inconsistency mentions the correct temporal references
        self.assertIn("meeting", temporal_inconsistencies[0].description)
        self.assertIn("today", temporal_inconsistencies[0].description)
        self.assertIn("tomorrow", temporal_inconsistencies[0].description)
        
        # Should raise ValueError when validating
        with self.assertRaises(ValueError):
            validator.validate_consistency()

    def test_multiple_inconsistencies(self):
        """Test detection of multiple inconsistencies."""
        multiple_inconsistencies = "The weather in Boston is rainy with a temperature of 60 degrees. Don't forget your meeting today."
        validator = ConsistencyValidator(
            response=multiple_inconsistencies,
            context=self.context
        )
        
        # Should detect multiple inconsistencies
        inconsistencies = validator.check_for_inconsistencies()
        self.assertGreaterEqual(len(inconsistencies), 3)  # Location, state, temperature, and temporal
        
        # Should raise ValueError when validating
        with self.assertRaises(ValueError):
            validator.validate_consistency()

    def test_fact_extraction(self):
        """Test extraction of facts from response."""
        response_with_facts = "The sky is blue. The grass is green. The temperature is 75 degrees. New York is in the United States."
        validator = ConsistencyValidator(
            response=response_with_facts,
            context=self.context
        )
        
        extracted_facts = validator.extract_potential_facts()
        self.assertGreater(len(extracted_facts), 0)
        
        # Check that we extracted the correct facts
        attribute_facts = [fact for fact in extracted_facts if fact.fact_type == FactType.ATTRIBUTE]
        self.assertGreater(len(attribute_facts), 0)
        
        numerical_facts = [fact for fact in extracted_facts if fact.fact_type == FactType.NUMERICAL]
        self.assertGreater(len(numerical_facts), 0)
        
        location_facts = [fact for fact in extracted_facts if fact.fact_type == FactType.LOCATION]
        self.assertGreater(len(location_facts), 0)
        
        # Check specific facts
        sky_facts = [fact for fact in attribute_facts if "sky" in fact.related_entities]
        self.assertGreater(len(sky_facts), 0)
        self.assertEqual(sky_facts[0].value, "blue")
        
        temperature_facts = [fact for fact in numerical_facts if "temperature" in fact.related_entities]
        self.assertGreater(len(temperature_facts), 0)
        self.assertEqual(temperature_facts[0].value, 75)
        
        new_york_facts = [fact for fact in location_facts if "New York" in fact.related_entities]
        self.assertGreater(len(new_york_facts), 0)
        self.assertEqual(new_york_facts[0].value, "the United States")

    def test_conversation_context(self):
        """Test conversation context functionality."""
        # Add a new fact
        fact = self.context.add_fact(
            key="attribute.sky",
            value="blue",
            fact_type=FactType.ATTRIBUTE,
            source=FactSource.SYSTEM,
            related_entities=["sky"]
        )
        
        # Check that the fact was added
        self.assertEqual(self.context.get_fact("attribute.sky"), fact)
        
        # Check that we can get facts by type
        attribute_facts = self.context.get_facts_by_type(FactType.ATTRIBUTE)
        self.assertIn(fact, attribute_facts)
        
        # Check that we can get facts by entity
        sky_facts = self.context.get_facts_by_entity("sky")
        self.assertIn(fact, sky_facts)
        
        # Check that we can get entity attributes
        sky_attributes = self.context.get_entity_attributes("sky")
        self.assertEqual(sky_attributes.get("sky"), "blue")

    def test_conversation_history(self):
        """Test conversation history functionality."""
        history = ConversationHistory(context=self.context)
        
        # Add a message
        message = history.add_message(
            text="The weather is sunny in New York.",
            sender="user"
        )
        
        # Check that the message was added
        self.assertIn(message, history.messages)
        
        # Add another message with extracted facts
        fact = ContextualFact(
            key="attribute.sky",
            value="blue",
            fact_type=FactType.ATTRIBUTE,
            source=FactSource.AGENT,
            related_entities=["sky"]
        )
        
        message2 = history.add_message(
            text="The sky is blue today.",
            sender="agent",
            extracted_facts=[fact]
        )
        
        # Check that the message was added
        self.assertIn(message2, history.messages)
        
        # Check that the fact was added to the context
        self.assertEqual(history.context.get_fact("attribute.sky"), fact)
        
        # Check that we can get messages by sender
        user_messages = history.get_messages_by_sender("user")
        self.assertIn(message, user_messages)
        self.assertNotIn(message2, user_messages)
        
        agent_messages = history.get_messages_by_sender("agent")
        self.assertIn(message2, agent_messages)
        self.assertNotIn(message, agent_messages)
        
        # Check that we can get the last n messages
        last_message = history.get_last_n_messages(1)[0]
        self.assertEqual(last_message, message2)


if __name__ == "__main__":
    unittest.main()
