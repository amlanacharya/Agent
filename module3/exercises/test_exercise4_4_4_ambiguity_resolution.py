"""
Test file for Exercise 4.4.4: Ambiguity Resolution System
"""

import unittest
from exercise4_4_4_ambiguity_resolution import (
    AmbiguityType, Intent, Entity, PossibleInterpretation, AmbiguityResolution
)


class TestAmbiguityResolution(unittest.TestCase):
    """Test cases for ambiguity resolution system."""

    def test_missing_entity_detection(self):
        """Test detection of missing entity ambiguity."""
        # Weather intent without location
        text = "What's the weather like?"
        entities = {}
        intents = [Intent(type="weather", confidence=0.9)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        self.assertTrue(ambiguity.is_ambiguous)
        self.assertEqual(ambiguity.ambiguity_type, AmbiguityType.MISSING_ENTITY)
        self.assertIsNotNone(ambiguity.clarification_question)
        self.assertIn("location", ambiguity.clarification_question.lower())

    def test_multiple_intents_detection(self):
        """Test detection of multiple intents ambiguity."""
        # Text with multiple possible intents
        text = "I need to make a reservation"
        entities = {}
        intents = [
            Intent(type="booking", confidence=0.6),
            Intent(type="information", confidence=0.5)
        ]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        self.assertTrue(ambiguity.is_ambiguous)
        self.assertEqual(ambiguity.ambiguity_type, AmbiguityType.MULTIPLE_INTENTS)
        self.assertIsNotNone(ambiguity.clarification_question)
        self.assertGreaterEqual(len(ambiguity.possible_interpretations), 2)

    def test_vague_reference_detection(self):
        """Test detection of vague reference ambiguity."""
        # Text with pronouns without clear referents
        text = "What's the price of it?"
        entities = {}
        intents = [Intent(type="price_query", confidence=0.8)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        self.assertTrue(ambiguity.is_ambiguous)
        self.assertEqual(ambiguity.ambiguity_type, AmbiguityType.VAGUE_REFERENCE)
        self.assertIsNotNone(ambiguity.clarification_question)

    def test_homonym_detection(self):
        """Test detection of homonym ambiguity."""
        # Text with a word that has multiple meanings
        text = "I need to book a room"
        entities = {}
        intents = [Intent(type="hotel", confidence=0.8)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        # This should detect "book" as a homonym
        self.assertTrue(ambiguity.is_ambiguous)
        self.assertEqual(ambiguity.ambiguity_type, AmbiguityType.HOMONYM)
        self.assertIsNotNone(ambiguity.clarification_question)
        self.assertIn("book", ambiguity.clarification_question.lower())

    def test_non_ambiguous_input(self):
        """Test handling of non-ambiguous input."""
        # Clear intent with all required entities
        text = "What's the weather like in New York?"
        entities = {"location": Entity(type="location", value="New York", confidence=0.9)}
        intents = [Intent(type="weather", confidence=0.9)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        self.assertFalse(ambiguity.is_ambiguous)
        self.assertIsNone(ambiguity.ambiguity_type)
        self.assertIsNone(ambiguity.clarification_question)
        self.assertEqual(len(ambiguity.possible_interpretations), 0)

    def test_resolve_missing_entity(self):
        """Test resolving missing entity ambiguity."""
        # Create an ambiguity with missing location for weather
        text = "What's the weather like?"
        entities = {}
        intents = [Intent(type="weather", confidence=0.9)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        # Resolve with clarification
        clarification = "I want to know the weather in New York"
        resolution = ambiguity.resolve_with_clarification(clarification)
        
        self.assertTrue(resolution["resolved"])
        self.assertEqual(resolution["intent"], "weather")
        self.assertIn("location", resolution["entities"])
        self.assertEqual(resolution["entities"]["location"], "New York")

    def test_resolve_multiple_intents(self):
        """Test resolving multiple intents ambiguity."""
        # Create an ambiguity with multiple possible intents
        text = "I need to make a reservation"
        entities = {}
        intents = [
            Intent(type="booking", confidence=0.6),
            Intent(type="information", confidence=0.5)
        ]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        # Resolve with clarification
        clarification = "I want to book a restaurant reservation"
        resolution = ambiguity.resolve_with_clarification(clarification)
        
        self.assertTrue(resolution["resolved"])
        self.assertEqual(resolution["intent"], "booking")

    def test_get_best_clarification_question(self):
        """Test getting the best clarification question."""
        # Create an ambiguity with missing location for weather
        text = "What's the weather like?"
        entities = {}
        intents = [Intent(type="weather", confidence=0.9)]
        
        ambiguity = AmbiguityResolution.analyze(text, entities, intents)
        
        # Get best clarification question
        question = ambiguity.get_best_clarification_question()
        
        self.assertIsNotNone(question)
        self.assertGreater(len(question), 0)
        
        # For non-ambiguous input
        non_ambiguous = AmbiguityResolution(
            original_text="What's the weather like in New York?",
            is_ambiguous=False,
            ambiguity_type=None,
            clarification_question=None,
            possible_interpretations=[]
        )
        
        question = non_ambiguous.get_best_clarification_question()
        self.assertIn("understand", question)

    def test_validation_errors(self):
        """Test validation errors for inconsistent fields."""
        # Ambiguous but missing ambiguity type
        with self.assertRaises(ValueError):
            AmbiguityResolution(
                original_text="What's the weather like?",
                is_ambiguous=True,
                ambiguity_type=None,
                clarification_question="Which location?",
                possible_interpretations=[]
            )
        
        # Ambiguous but missing clarification question
        with self.assertRaises(ValueError):
            AmbiguityResolution(
                original_text="What's the weather like?",
                is_ambiguous=True,
                ambiguity_type=AmbiguityType.MISSING_ENTITY,
                clarification_question=None,
                possible_interpretations=[]
            )
        
        # Not ambiguous but has ambiguity type
        with self.assertRaises(ValueError):
            AmbiguityResolution(
                original_text="What's the weather like in New York?",
                is_ambiguous=False,
                ambiguity_type=AmbiguityType.MISSING_ENTITY,
                clarification_question=None,
                possible_interpretations=[]
            )


if __name__ == "__main__":
    unittest.main()
