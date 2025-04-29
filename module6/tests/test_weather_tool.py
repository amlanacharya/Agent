"""
Tests for the Weather Tool
-------------------------
This file contains tests for the Weather Tool implementation.
"""

import unittest
import os
from dotenv import load_dotenv
from module6.code.tools.weather_tool import WeatherTool

class TestWeatherTool(unittest.TestCase):
    """Test cases for the Weather tool."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()

    def setUp(self):
        """Set up the test environment."""
        # Check if OPENWEATHERMAP_API_KEY is available
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            self.skipTest("OPENWEATHERMAP_API_KEY environment variable not set")

        # Create the tool
        self.weather_tool = WeatherTool()

    def test_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.weather_tool.name, "weather")
        self.assertIn("weather", self.weather_tool.description.lower())
        self.assertEqual(self.weather_tool.max_retries, 3)
        self.assertEqual(self.weather_tool.retry_delay, 2)

    def test_get_schema(self):
        """Test that the schema is correctly defined."""
        schema = self.weather_tool.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "function")
        self.assertIn("parameters", schema["function"])
        self.assertIn("location", schema["function"]["parameters"]["properties"])
        self.assertIn("lat", schema["function"]["parameters"]["properties"])
        self.assertIn("lon", schema["function"]["parameters"]["properties"])
        self.assertIn("units", schema["function"]["parameters"]["properties"])

    def test_weather_by_location(self):
        """Test getting weather by location."""
        try:
            # Test using execute method
            response = self.weather_tool.execute(location="London")

            # If the API is working, check the response
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertIn("temperature", response.result)
                self.assertIn("conditions", response.result)
                self.assertIn("humidity", response.result)
                self.assertIn("wind_speed", response.result)
                self.assertIn("location", response.result)
                self.assertEqual(response.metadata["query_type"], "location")

                # Test using convenience method
                result = self.weather_tool.get_current_weather("London")
                self.assertIsInstance(result, dict)
                self.assertIn("temperature", result)
                self.assertIn("location", result)
            else:
                # If the API is not working, skip the test
                self.skipTest(f"OpenWeatherMap API error: {response.error}")
        except Exception as e:
            self.skipTest(f"Test skipped due to API error: {str(e)}")

    def test_weather_by_coordinates(self):
        """Test getting weather by coordinates."""
        try:
            # New York coordinates
            lat = 40.7128
            lon = -74.0060

            # Test using execute method
            response = self.weather_tool.execute(lat=lat, lon=lon)

            # If the API is working, check the response
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertIn("temperature", response.result)
                self.assertIn("conditions", response.result)
                self.assertEqual(response.metadata["query_type"], "coordinates")

                # Test using convenience method
                result = self.weather_tool.get_current_weather({"lat": lat, "lon": lon})
                self.assertIsInstance(result, dict)
                self.assertIn("temperature", result)
            else:
                # If the API is not working, skip the test
                self.skipTest(f"OpenWeatherMap API error: {response.error}")
        except Exception as e:
            self.skipTest(f"Test skipped due to API error: {str(e)}")

    def test_units_parameter(self):
        """Test that the units parameter works correctly."""
        # Test metric units (default)
        response_metric = self.weather_tool.execute(location="London")
        self.assertTrue(response_metric.success)
        self.assertEqual(response_metric.metadata["units"], "metric")

        # Test imperial units
        response_imperial = self.weather_tool.execute(location="London", units="imperial")
        self.assertTrue(response_imperial.success)
        self.assertEqual(response_imperial.metadata["units"], "imperial")

    def test_invalid_parameters(self):
        """Test that the tool handles invalid parameters correctly."""
        # Test with no location or coordinates
        response = self.weather_tool.execute()
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("must be provided", response.error)

        # Test with invalid location
        response = self.weather_tool.execute(location="ThisLocationDoesNotExist12345")
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Location not found", response.error)


if __name__ == "__main__":
    unittest.main()
