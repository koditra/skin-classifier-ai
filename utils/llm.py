import requests

API_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"

API_KEY = "sk-hc-v1-sk-hc-v1-54d0b9c27cda4d9ea957fc3d2ee7e6a8c2b637ee61a244c7aaec36c2ce43a4a3"

def ask_llm(messages):
    r = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen/qwen3-32b",
            "messages": messages
        }
    )

    return r.json()["choices"][0]["message"]["content"]