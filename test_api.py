import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [0.5]*20
}

response = requests.post(url, json=data)

print("Response:", response.json())