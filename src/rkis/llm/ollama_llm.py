import requests
from config.settings import settings

class OllamaLLM:
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.url = f"{settings.OLLAMA_BASE_URL}/api/chat"

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]