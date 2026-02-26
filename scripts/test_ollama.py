"""Quick diagnostic — call Ollama directly and print raw response."""
import json, urllib.request

def call_ollama(model, prompt, num_predict=512):
    url = "http://localhost:11434/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict}
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except Exception as e:
        return f"ERROR: {e}"

prompt = """You are a Connect Four evaluator.

Board:
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
0 1 2 3 4 5 6

Player X is about to move. Legal columns: 0, 1, 2, 3, 4, 5, 6

Estimate the win probability for player X.
Output ONLY a single decimal number between 0.0 and 1.0.
Your answer (just the number):"""

print("=== Testing qwen3:8b ===")
raw = call_ollama("qwen3:8b", prompt, num_predict=512)
print(f"RAW RESPONSE ({len(raw)} chars):")
print(repr(raw[:500]))
print()
print("VISIBLE:")
print(raw[:500])