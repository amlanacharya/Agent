"""
Test file for Exercise 4.5.4: Completeness Checker
"""

import unittest
from exercise4_5_4_completeness_checker import (
    QuestionType, Question, UserQuery, CompletenessReport, CompletenessChecker
)


class TestCompletenessChecker(unittest.TestCase):
    """Test cases for completeness checker."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = CompletenessChecker()

    def test_question_extraction(self):
        """Test extraction of questions from user queries."""
        # Test simple question
        query1 = UserQuery(text="What's the weather like in New York?")
        self.assertEqual(len(query1.questions), 1)
        self.assertEqual(query1.questions[0].question_type, QuestionType.FACTUAL)
        self.assertIn("weather", query1.questions[0].key_entities)
        self.assertIn("york", query1.questions[0].key_entities)
        
        # Test multi-part question
        query2 = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
        self.assertEqual(len(query2.questions), 2)
        self.assertEqual(query2.questions[0].question_type, QuestionType.FACTUAL)
        self.assertEqual(query2.questions[1].question_type, QuestionType.FACTUAL)
        
        # Test implicit question
        query3 = UserQuery(text="Tell me about the weather in New York")
        self.assertEqual(len(query3.questions), 0)  # No explicit question mark
        
        # Test question with question word but no question mark
        query4 = UserQuery(text="What is the weather like in New York")
        self.assertEqual(len(query4.questions), 1)  # Starts with question word
        
        # Test complex multi-part question
        query5 = UserQuery(text="Can you tell me about the weather in New York, the best restaurants to visit, and how to get around the city?")
        self.assertEqual(len(query5.questions), 1)
        self.assertTrue(query5.questions[0].is_multi_part)
        self.assertGreaterEqual(len(query5.questions[0].sub_questions), 1)

    def test_question_type_detection(self):
        """Test detection of question types."""
        # Test factual question
        query1 = UserQuery(text="What is the capital of France?")
        self.assertEqual(query1.questions[0].question_type, QuestionType.FACTUAL)
        
        # Test temporal question
        query2 = UserQuery(text="When is the best time to visit Paris?")
        self.assertEqual(query2.questions[0].question_type, QuestionType.TEMPORAL)
        
        # Test procedural question
        query3 = UserQuery(text="How do I make a chocolate cake?")
        self.assertEqual(query3.questions[0].question_type, QuestionType.PROCEDURAL)
        
        # Test causal question
        query4 = UserQuery(text="Why is the sky blue?")
        self.assertEqual(query4.questions[0].question_type, QuestionType.CAUSAL)
        
        # Test comparative question
        query5 = UserQuery(text="Which is better, iPhone or Android?")
        self.assertEqual(query5.questions[0].question_type, QuestionType.COMPARATIVE)
        
        # Test hypothetical question
        query6 = UserQuery(text="What would happen if the sun disappeared?")
        self.assertEqual(query6.questions[0].question_type, QuestionType.HYPOTHETICAL)
        
        # Test preference question
        query7 = UserQuery(text="Do you recommend visiting Paris or London?")
        self.assertEqual(query7.questions[0].question_type, QuestionType.PREFERENCE)
        
        # Test confirmation question
        query8 = UserQuery(text="Is Paris the capital of France?")
        self.assertEqual(query8.questions[0].question_type, QuestionType.CONFIRMATION)
        
        # Test quantitative question
        query9 = UserQuery(text="How many planets are in our solar system?")
        self.assertEqual(query9.questions[0].question_type, QuestionType.QUANTITATIVE)

    def test_entity_extraction(self):
        """Test extraction of entities from questions."""
        query = UserQuery(text="What's the weather like in New York tomorrow?")
        entities = query.questions[0].key_entities
        
        self.assertIn("weather", entities)
        self.assertIn("york", entities)
        self.assertIn("tomorrow", entities)
        
        # Check that stop words are removed
        self.assertNotIn("what", entities)
        self.assertNotIn("the", entities)
        self.assertNotIn("like", entities)
        self.assertNotIn("in", entities)

    def test_topic_extraction(self):
        """Test extraction of topics from questions."""
        # Test weather topic
        query1 = UserQuery(text="What's the weather forecast for tomorrow?")
        topics1 = query1.questions[0].key_topics
        self.assertIn("weather", topics1)
        
        # Test travel topic
        query2 = UserQuery(text="What are good hotels in Paris for a vacation?")
        topics2 = query2.questions[0].key_topics
        self.assertIn("travel", topics2)
        
        # Test food topic
        query3 = UserQuery(text="What's a good recipe for chocolate cake?")
        topics3 = query3.questions[0].key_topics
        self.assertIn("food", topics3)
        
        # Test multiple topics
        query4 = UserQuery(text="What's the weather like in Paris and what restaurants should I visit?")
        topics4 = set()
        for question in query4.questions:
            topics4.update(question.key_topics)
        self.assertIn("weather", topics4)
        self.assertIn("food", topics4)
        self.assertIn("travel", topics4)

    def test_complete_response(self):
        """Test validation of a complete response."""
        query = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
        response = "The weather in New York is currently sunny with a temperature of 75°F. The best time to visit New York is during spring (April to June) or fall (September to November) when the weather is mild."
        
        report = self.checker.check_completeness(query, response)
        
        self.assertTrue(report.is_complete)
        self.assertGreaterEqual(report.completeness_score, 0.8)
        self.assertEqual(len(report.addressed_questions), 2)
        self.assertEqual(len(report.unanswered_questions), 0)
        self.assertEqual(len(report.partially_addressed_questions), 0)

    def test_incomplete_response(self):
        """Test validation of an incomplete response."""
        query = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
        response = "The weather in New York is currently sunny with a temperature of 75°F."
        
        report = self.checker.check_completeness(query, response)
        
        self.assertFalse(report.is_complete)
        self.assertLess(report.completeness_score, 0.8)
        self.assertLess(len(report.addressed_questions), 2)
        self.assertGreater(len(report.unanswered_questions) + len(report.partially_addressed_questions), 0)

    def test_partially_complete_response(self):
        """Test validation of a partially complete response."""
        query = UserQuery(text="What's the weather like in New York? What's the best time to visit? And what are some good restaurants?")
        response = "The weather in New York is currently sunny with a temperature of 75°F. The best time to visit New York is during spring or fall."
        
        report = self.checker.check_completeness(query, response)
        
        self.assertFalse(report.is_complete)
        self.assertGreater(len(report.addressed_questions), 0)
        self.assertGreater(len(report.unanswered_questions), 0)

    def test_complex_query(self):
        """Test validation of a response to a complex query."""
        query = UserQuery(text="Can you tell me about the weather in New York, the best restaurants to visit, and how to get around the city? Also, what are some free activities for tourists?")
        response = "New York currently has sunny weather with temperatures around 75°F. For restaurants, I recommend trying the famous pizza at Joe's Pizza or fine dining at Per Se. To get around, the subway is the most efficient option, with a 7-day unlimited MetroCard costing $33. As for free activities, you can visit Central Park, walk the High Line, or check out the Staten Island Ferry for great views of the Statue of Liberty."
        
        report = self.checker.check_completeness(query, response)
        
        self.assertTrue(report.is_complete)
        self.assertGreaterEqual(report.completeness_score, 0.8)
        
        # Check that all main topics are covered
        all_topics = set()
        for question in query.questions:
            all_topics.update(question.key_topics)
        
        self.assertGreaterEqual(len(all_topics), 2)  # Should have at least weather and travel topics
        self.assertEqual(len(report.missing_topics), 0)

    def test_entity_coverage(self):
        """Test entity coverage calculation."""
        query = UserQuery(text="What's the weather like in New York and Chicago?")
        
        # Test with all entities covered
        response1 = "The weather in New York is sunny and Chicago is experiencing rain."
        report1 = self.checker.check_completeness(query, response1)
        self.assertTrue(report1.is_complete)
        self.assertEqual(len(report1.missing_entities), 0)
        
        # Test with some entities missing
        response2 = "The weather in New York is sunny."
        report2 = self.checker.check_completeness(query, response2)
        self.assertFalse(report2.is_complete)
        self.assertGreater(len(report2.missing_entities), 0)
        self.assertIn("chicago", report2.missing_entities)

    def test_topic_coverage(self):
        """Test topic coverage calculation."""
        query = UserQuery(text="What's the weather like in New York and what are some good restaurants?")
        
        # Test with all topics covered
        response1 = "The weather in New York is sunny with a temperature of 75°F. For restaurants, I recommend trying the famous pizza at Joe's Pizza."
        report1 = self.checker.check_completeness(query, response1)
        self.assertTrue(report1.is_complete)
        self.assertEqual(len(report1.missing_topics), 0)
        
        # Test with some topics missing
        response2 = "The weather in New York is sunny with a temperature of 75°F."
        report2 = self.checker.check_completeness(query, response2)
        self.assertFalse(report2.is_complete)
        self.assertGreater(len(report2.missing_topics), 0)
        self.assertIn("food", report2.missing_topics)

    def test_completeness_report(self):
        """Test completeness report functionality."""
        query = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
        response = "The weather in New York is currently sunny."
        
        report = self.checker.check_completeness(query, response)
        
        # Test report properties
        self.assertEqual(report.total_questions, 2)
        self.assertLess(report.addressed_ratio, 1.0)
        
        # Test summary
        summary = report.get_summary()
        self.assertFalse(summary["is_complete"])
        self.assertEqual(summary["total_questions"], 2)
        self.assertLess(summary["addressed_questions"], 2)
        self.assertGreater(summary["unanswered_questions"] + summary["partially_addressed_questions"], 0)


if __name__ == "__main__":
    unittest.main()
