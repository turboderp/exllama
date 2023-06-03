import requests
import json

data = {
    "prompt": "You are Samantha, a sentient AI.\nUSER: {user_input}\nASSISTANT:",
    "message": "Tell me about yourself!",
    "max_new_tokens": 256
}

r = requests.post("http://localhost:7862/generate", json=data, stream=True)
#r = requests.post("http://wintermute:7862/generate", json=data, stream=True)
#r = requests.post("http://localhost:8000/ask", json=data, stream=True)

if r.status_code==200:
    for chunk in r.iter_content():
        if chunk:
            print(chunk.decode("utf-8"), end="", flush=True)
else:
    print("Request failed with status code:", r.status_code)












