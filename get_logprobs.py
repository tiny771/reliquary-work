import json
import requests
from pprint import pprint

VLLM_URL = "http://localhost:8004/v1/chat/completions"

payload = {
    "model": "qwen",
    "messages": [
        {
            "role": "system",
            "content": """You are a precise math problem solver.
Solve the problem with very short and clear steps.
Use 3 to 6 short steps maximum.
Put the final answer in \\boxed{}.
Keep the entire response under 200 tokens. Be concise.
"""
        },
        {
            "role": "user",
            "content": """A sequence of consecutive positive integers has a sum of 504. 
The sum of the squares of the first half of the sequence is equal to the sum of the squares of the second half. 
How many integers are in the sequence?"""
        }
    ],
    "max_tokens": 300,
    "temperature": 3.5,
    "top_p": 1.0,
    
    "logprobs": True,
    "top_logprobs": 5,
    "chat_template_kwargs": {
        "enable_thinking": False,
    }
}

response = requests.post(
    VLLM_URL,
    json=payload,
)

if response.status_code == 200:
    result = response.json()

    print("=== Generated Text ===")
    print(result["choices"][0]["message"]["content"])

    print("\n=== Logprobs ===")
    logprobs = result["choices"][0].get("logprobs")

    if logprobs and logprobs.get("content"):
        for i, token_info in enumerate(logprobs["content"]):
            token = token_info["token"]
            logprob = token_info["logprob"]
            print(f"{i:2d} | Token: '{token}' | Logprob:{logprob:.4f}")

            if token_info.get("top_logprobs"):
                print("   Top althernatives:")
                for alt in token_info["top_logprobs"]:
                    if alt["token"] != token:
                        print(f" '{alt['token']}' -> {alt['logprob']:.4f}")
    else:
        print("No logprobs returned.")

else:
    print(f"Error: {response.status_code}")
    print(response.text)