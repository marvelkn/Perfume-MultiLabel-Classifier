import requests
import json

url = "http://127.0.0.1:8000/recommend"
payload = {
    "desired_accords": {
        "citrus": 1.0,
        "woody": 0.5
    },
    "top_k": 3
}

try:
    print(f"Mengirim request ke {url}...")
    res = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {res.status_code}")
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
