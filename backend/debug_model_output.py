import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def test_stream():
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
    }
    data = {
        "model": "qwen/qwen3-235b-a22b-thinking-2507",
        "messages": [
            {"role": "user", "content": "how to file a FIR under indian law?"}
        ],
        "stream": True
    }

    print("Sending request to OpenRouter...")
    response = requests.post(url, headers=headers, json=data, stream=True)

    print("\n--- RAW STREAM CHUNKS (First 20) ---")
    count = 0
    with open("model_output_debug.txt", "w", encoding="utf-8") as f:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    if decoded_line == 'data: [DONE]':
                        break
                    try:
                        json_str = decoded_line[6:]  # Skip 'data: '
                        chunk = json.loads(json_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning", "") # Check for reasoning field
                            
                            log_msg = f"Chunk: content={repr(content)} | reasoning={repr(reasoning)}"
                            print(log_msg)
                            f.write(log_msg + "\n")
                            
                            count += 1
                            if count >= 50:
                                break
                    except Exception as e:
                        print(f"Error parsing: {e}")

if __name__ == "__main__":
    test_stream()
