import requests
import json

url = "http://127.0.0.1:5001/invocations"

data = {
    "inputs": [[14.5,20.3,95.5,700,0.1,0.2,0.15,0.09,0.18,0.06,
                0.5,1.2,3.5,40,0.005,0.02,0.03,0.01,0.02,0.003,
                16,25,110,800,0.14,0.4,0.5,0.2,0.3,0.1]]
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
