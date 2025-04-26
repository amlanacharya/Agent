"""
Test file for Exercise 4.4.2: Entity Extraction System
"""

import unittest
from exercise4_4_2_entity_extraction import (
    EntityExtractor, ExtractedEntities, Entity,
    DateEntity, TimeEntity, LocationEntity, PersonEntity
)


class TestEntityExtraction(unittest.TestCase):
    """Test cases for entity extraction system."""

    def test_date_extraction(self):
        """Test extraction of date entities."""
        # Test various date formats
        text = "Meeting scheduled for 12/25/2023 and another on 01-15-2024. " \
               "Also on January 1, 2023 and February 28, 2023."

        dates = EntityExtractor.extract_dates(text)

        # Should find 4 dates
        self.assertEqual(len(dates), 4)

        # Check specific dates
        date_values = [date.value for date in dates]
        self.assertIn("12/25/2023", date_values)
        self.assertIn("01-15-2024", date_values)

        # Check for month names (case-insensitive)
        date_values_lower = [date.value.lower() for date in dates]
        self.assertIn("january 1, 2023", date_values_lower)
        self.assertIn("february 28, 2023", date_values_lower)

        # Check that dates were parsed correctly
        for date in dates:
            self.assertIsNotNone(date.parsed_date)
            if date.value == "12/25/2023":
                self.assertEqual(date.parsed_date.year, 2023)
                self.assertEqual(date.parsed_date.month, 12)
                self.assertEqual(date.parsed_date.day, 25)

    def test_time_extraction(self):
        """Test extraction of time entities."""
        # Test various time formats
        text = "Meeting at 9:30 AM, lunch at 12:00 PM, and dinner at 18:45. " \
               "Breakfast at 7 o'clock in the morning and movie at 8 o'clock in the evening."

        times = EntityExtractor.extract_times(text)

        # Should find 5 times
        self.assertEqual(len(times), 5)

        # Check specific times
        time_values = [time.value.lower() for time in times]
        self.assertIn("9:30 am", time_values)
        self.assertIn("12:00 pm", time_values)
        self.assertIn("18:45", time_values)
        self.assertIn("7 o'clock in the morning", time_values)
        self.assertIn("8 o'clock in the evening", time_values)

        # Check that times were parsed correctly
        for time in times:
            self.assertIsNotNone(time.parsed_time)
            if "9:30 am" in time.value.lower():
                self.assertEqual(time.parsed_time.hour, 9)
                self.assertEqual(time.parsed_time.minute, 30)
            elif "8 o'clock in the evening" in time.value.lower():
                self.assertEqual(time.parsed_time.hour, 20)
                self.assertEqual(time.parsed_time.minute, 0)

    def test_location_extraction(self):
        """Test extraction of location entities."""
        # Test various location formats
        text = "I'm traveling to New York next week. The conference is in London. " \
               "We're flying from Tokyo to Paris. Let's meet at Central Park."

        locations = EntityExtractor.extract_locations(text)

        # Should find at least 4 locations
        self.assertGreaterEqual(len(locations), 4)

        # Check specific locations
        location_values = [loc.value for loc in locations]
        self.assertIn("New York", location_values)
        self.assertIn("London", location_values)
        self.assertIn("Tokyo", location_values)
        self.assertIn("Paris", location_values)

    def test_person_extraction(self):
        """Test extraction of person entities."""
        # Test various person formats
        text = "Dr. Jane Smith will present the findings. John Johnson is the project manager. " \
               "I spoke with Mrs. Emily Wilson yesterday. Prof. Robert Brown is joining us."

        people = EntityExtractor.extract_people(text)

        # Should find 4 people
        self.assertEqual(len(people), 4)

        # Check specific people
        for person in people:
            if "Jane Smith" in person.value:
                self.assertEqual(person.title, "Dr")
                self.assertEqual(person.first_name, "Jane")
                self.assertEqual(person.last_name, "Smith")
            elif "John Johnson" in person.value:
                self.assertEqual(person.first_name, "John")
                self.assertEqual(person.last_name, "Johnson")
            elif "Emily Wilson" in person.value:
                self.assertEqual(person.title, "Mrs")
                self.assertEqual(person.first_name, "Emily")
                self.assertEqual(person.last_name, "Wilson")
            elif "Robert Brown" in person.value:
                self.assertEqual(person.title, "Prof")
                self.assertEqual(person.first_name, "Robert")
                self.assertEqual(person.last_name, "Brown")

    def test_combined_extraction(self):
        """Test extraction of all entity types together."""
        text = "I have a meeting with Dr. Jane Smith on January 15, 2023 at 3:30 PM. " \
               "We'll discuss our upcoming trip to New York City on 05/20/2023. " \
               "John Johnson from London will join us at 10 o'clock in the morning."

        extracted = EntityExtractor.extract_entities(text)

        # Check that we have all entity types
        self.assertIn("date", extracted.entities)
        self.assertIn("time", extracted.entities)
        self.assertIn("location", extracted.entities)
        self.assertIn("person", extracted.entities)

        # Check counts of each entity type
        self.assertEqual(len(extracted.entities["date"]), 2)
        self.assertEqual(len(extracted.entities["time"]), 2)
        self.assertGreaterEqual(len(extracted.entities["location"]), 2)

        # We might get more than 2 people depending on the implementation
        # Just check that we have at least the 2 expected people
        self.assertGreaterEqual(len(extracted.entities["person"]), 2)


if __name__ == "__main__":
    unittest.main()
