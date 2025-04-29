import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("SERPER_API_KEY", "")
if api_key and (api_key.startswith('"') or api_key.startswith("'")):
    api_key = api_key.strip('"\'')
    print("Warning: Removed quotes from Serper API key. Please update your .env file.")

print(f"Using API key: {api_key[:5]}... (length: {len(api_key)})")

url = "https://google.serper.dev/search"

payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else response.text)
