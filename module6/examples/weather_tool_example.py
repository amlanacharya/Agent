"""
Weather Tool Example
------------------
This script demonstrates how to use the Weather Tool.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from module6.code.tools.weather_tool import WeatherTool

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def print_weather(weather_data):
    """Print weather data in a formatted way."""
    print(f"🌡️  Temperature: {weather_data['temperature']}°{'C' if weather_data['units'] == 'metric' else 'F'}")
    print(f"🌤️  Conditions: {weather_data['conditions']}")
    print(f"💧 Humidity: {weather_data['humidity']}%")
    print(f"💨 Wind Speed: {weather_data['wind_speed']} {'m/s' if weather_data['units'] == 'metric' else 'mph'}")
    print(f"📍 Location: {weather_data['location']}")

def main():
    """Run the example."""
    print_section("Weather Tool Example")
    
    # Create the weather tool
    try:
        weather_tool = WeatherTool()
        print("✅ Weather Tool created successfully")
    except ValueError as e:
        print(f"❌ Failed to create Weather Tool: {e}")
        print("Make sure you have set the OPENWEATHERMAP_API_KEY environment variable")
        return
    
    # Example 1: Get weather by location
    print_section("Example 1: Get Weather by Location")
    try:
        response = weather_tool.execute(location="London")
        if response.success:
            print("Weather in London:")
            print_weather(response.result)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Example 2: Get weather by coordinates (New York)
    print_section("Example 2: Get Weather by Coordinates")
    try:
        response = weather_tool.execute(lat=40.7128, lon=-74.0060)
        if response.success:
            print("Weather in New York (by coordinates):")
            print_weather(response.result)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Example 3: Get weather with imperial units
    print_section("Example 3: Get Weather with Imperial Units")
    try:
        response = weather_tool.execute(location="Tokyo", units="imperial")
        if response.success:
            print("Weather in Tokyo (imperial units):")
            print_weather(response.result)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Example 4: Using the convenience method
    print_section("Example 4: Using the Convenience Method")
    try:
        weather_data = weather_tool.get_current_weather("Paris")
        print("Weather in Paris (using convenience method):")
        print_weather(weather_data)
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print_section("Example Complete")
    print("You can now use the Weather Tool in your own applications!")

if __name__ == "__main__":
    main()
