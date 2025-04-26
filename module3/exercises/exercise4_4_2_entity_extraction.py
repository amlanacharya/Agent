"""
Exercise 4.4.2: Entity Extraction System

This exercise implements an entity extraction system that can identify dates, times,
locations, and people from natural language text.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import re
from datetime import datetime, time


class Entity(BaseModel):
    """Base class for extracted entities."""
    type: str
    value: Any
    confidence: float = 1.0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


class DateEntity(Entity):
    """Date entity extracted from text."""
    type: Literal["date"] = "date"
    value: str
    parsed_date: Optional[datetime] = None


class TimeEntity(Entity):
    """Time entity extracted from text."""
    type: Literal["time"] = "time"
    value: str
    parsed_time: Optional[time] = None


class LocationEntity(Entity):
    """Location entity extracted from text."""
    type: Literal["location"] = "location"
    value: str
    location_type: Optional[str] = None  # city, country, address, etc.


class PersonEntity(Entity):
    """Person entity extracted from text."""
    type: Literal["person"] = "person"
    value: str
    title: Optional[str] = None  # Mr., Dr., etc.
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class ExtractedEntities(BaseModel):
    """Container for all entities extracted from text."""
    text: str
    entities: Dict[str, List[Entity]] = {}

    def add_entity(self, entity: Entity):
        """Add an entity to the appropriate category."""
        if entity.type not in self.entities:
            self.entities[entity.type] = []
        self.entities[entity.type].append(entity)


class EntityExtractor:
    """Entity extraction system for natural language text."""

    @staticmethod
    def extract_dates(text: str) -> List[DateEntity]:
        """Extract date entities from text."""
        dates = []

        # Format: MM/DD/YYYY or MM-DD-YYYY
        date_pattern1 = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        for match in re.finditer(date_pattern1, text):
            date_str = match.group(1)
            try:
                # Try to parse the date
                if '/' in date_str:
                    month, day, year = map(int, date_str.split('/'))
                else:
                    month, day, year = map(int, date_str.split('-'))

                # Add century if needed
                if year < 100:
                    year += 2000 if year < 50 else 1900

                parsed_date = datetime(year, month, day)

                dates.append(DateEntity(
                    value=date_str,
                    parsed_date=parsed_date,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except (ValueError, IndexError):
                # If parsing fails, still add the entity but without parsed_date
                dates.append(DateEntity(
                    value=date_str,
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        # Format: Month Day, Year (e.g., January 1, 2023)
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]
        month_pattern = r'(' + '|'.join(months) + r')\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(\d{4})'

        for match in re.finditer(month_pattern, text.lower()):
            month_str, day_str, year_str = match.groups()
            try:
                month = months.index(month_str) + 1
                day = int(day_str)
                year = int(year_str)

                parsed_date = datetime(year, month, day)

                # Get the original capitalized text for the value
                start, end = match.span()
                original_text = text[start:end]

                dates.append(DateEntity(
                    value=original_text,
                    parsed_date=parsed_date,
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except (ValueError, IndexError):
                # If parsing fails, still add the entity but without parsed_date
                start, end = match.span()
                original_text = text[start:end]

                dates.append(DateEntity(
                    value=original_text,
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return dates

    @staticmethod
    def extract_times(text: str) -> List[TimeEntity]:
        """Extract time entities from text."""
        times = []

        # Format: HH:MM or HH:MM:SS (12-hour or 24-hour)
        time_pattern1 = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)'
        for match in re.finditer(time_pattern1, text.lower()):
            time_str = match.group(1)
            try:
                # Try to parse the time
                time_parts = time_str.strip().split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1].split()[0] if ' ' in time_parts[1] else time_parts[1])
                second = 0

                # Check for AM/PM
                is_pm = 'pm' in time_str.lower() and hour < 12
                is_am = 'am' in time_str.lower() and hour == 12

                if is_pm:
                    hour += 12
                elif is_am:
                    hour = 0

                if len(time_parts) > 2:
                    second = int(time_parts[2].split()[0] if ' ' in time_parts[2] else time_parts[2])

                parsed_time = time(hour, minute, second)

                times.append(TimeEntity(
                    value=time_str,
                    parsed_time=parsed_time,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except (ValueError, IndexError):
                # If parsing fails, still add the entity but without parsed_time
                times.append(TimeEntity(
                    value=time_str,
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        # Format: X o'clock
        time_pattern2 = r'(\d{1,2})\s+o\'clock\s*(?:in the\s+)?(morning|afternoon|evening)?'
        for match in re.finditer(time_pattern2, text.lower()):
            hour_str, period = match.groups()
            try:
                hour = int(hour_str)

                # Adjust hour based on period
                if period == 'afternoon' or period == 'evening':
                    if hour < 12:
                        hour += 12
                elif period == 'morning' and hour == 12:
                    hour = 0

                parsed_time = time(hour, 0, 0)

                times.append(TimeEntity(
                    value=match.group(0),
                    parsed_time=parsed_time,
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except (ValueError, IndexError):
                # If parsing fails, still add the entity but without parsed_time
                times.append(TimeEntity(
                    value=match.group(0),
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return times

    @staticmethod
    def extract_locations(text: str) -> List[LocationEntity]:
        """Extract location entities from text."""
        locations = []

        # Pattern for "in [Location]" or "at [Location]"
        location_pattern1 = r'(?:in|at|to|from)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
        for match in re.finditer(location_pattern1, text):
            location = match.group(1)

            # Skip common non-location words that might be capitalized
            skip_words = ["I", "Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday", "January", "February",
                         "March", "April", "May", "June", "July", "August",
                         "September", "October", "November", "December"]

            if location not in skip_words:
                locations.append(LocationEntity(
                    value=location,
                    confidence=0.8,
                    start_pos=match.start(1),
                    end_pos=match.end(1)
                ))

        # Common city/country names
        common_locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "London", "Paris", "Tokyo", "Beijing", "Moscow", "Dubai", "Singapore",
            "USA", "UK", "Canada", "Australia", "Germany", "France", "Japan", "China"
        ]

        for location in common_locations:
            pattern = r'\b' + re.escape(location) + r'\b'
            for match in re.finditer(pattern, text):
                locations.append(LocationEntity(
                    value=location,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return locations

    @staticmethod
    def extract_people(text: str) -> List[PersonEntity]:
        """Extract person entities from text."""
        people = []

        # Pattern for titles followed by names
        titles = ["Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Professor", "Sir", "Madam"]
        title_pattern = r'(' + '|'.join(titles) + r')\.?\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?'

        for match in re.finditer(title_pattern, text):
            title, first_name, last_name = match.groups()

            people.append(PersonEntity(
                value=match.group(0),
                title=title,
                first_name=first_name,
                last_name=last_name,
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end()
            ))

        # Pattern for first and last names (without titles)
        name_pattern = r'([A-Z][a-z]+)\s+([A-Z][a-z]+)'

        for match in re.finditer(name_pattern, text):
            first_name, last_name = match.groups()

            # Skip if already found with a title
            already_found = False
            for person in people:
                if person.first_name == first_name and person.last_name == last_name:
                    already_found = True
                    break

            if not already_found:
                # Skip common non-person phrases that might match the pattern
                skip_words = ["New York", "Los Angeles", "San Francisco", "San Diego",
                             "San Jose", "San Antonio", "Central Park", "United States"]
                if f"{first_name} {last_name}" in skip_words:
                    continue

                # Lower confidence since no title
                people.append(PersonEntity(
                    value=match.group(0),
                    first_name=first_name,
                    last_name=last_name,
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return people

    @classmethod
    def extract_entities(cls, text: str) -> ExtractedEntities:
        """Extract all entity types from text."""
        result = ExtractedEntities(text=text)

        # Extract dates
        for date_entity in cls.extract_dates(text):
            result.add_entity(date_entity)

        # Extract times
        for time_entity in cls.extract_times(text):
            result.add_entity(time_entity)

        # Extract locations
        for location_entity in cls.extract_locations(text):
            result.add_entity(location_entity)

        # Extract people
        for person_entity in cls.extract_people(text):
            result.add_entity(person_entity)

        return result


# Example usage
if __name__ == "__main__":
    # Test with a complex text containing multiple entity types
    text = """
    I have a meeting with Dr. Jane Smith on January 15, 2023 at 3:30 PM.
    We'll discuss our upcoming trip to New York City on 05/20/2023.
    John Johnson from London will join us at 10 o'clock in the morning.
    The conference in Tokyo starts at 9:00 AM on March 3rd, 2023.
    """

    extracted = EntityExtractor.extract_entities(text)

    print(f"Text: '{extracted.text}'")
    print("\nExtracted entities:")

    for entity_type, entities in extracted.entities.items():
        print(f"\n{entity_type.capitalize()} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity.value} (confidence: {entity.confidence:.2f})")

            # Print additional details based on entity type
            if entity_type == "date" and entity.parsed_date:
                print(f"     Parsed date: {entity.parsed_date.strftime('%Y-%m-%d')}")
            elif entity_type == "time" and entity.parsed_time:
                print(f"     Parsed time: {entity.parsed_time.strftime('%H:%M:%S')}")
            elif entity_type == "person":
                person_details = []
                if entity.title:
                    person_details.append(f"Title: {entity.title}")
                if entity.first_name:
                    person_details.append(f"First name: {entity.first_name}")
                if entity.last_name:
                    person_details.append(f"Last name: {entity.last_name}")

                if person_details:
                    print(f"     {', '.join(person_details)}")
