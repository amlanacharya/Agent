import requests
import json

url = "https://google.serper.dev/search"

payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': '26be841e26a32aea8c1f43bbc7e497d9fe6393ed',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else response.text)
