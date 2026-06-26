from transformers import AutoTokenizer
import json
from datasets import Dataset
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

dataset = Dataset.load_from_disk("/mnt/d/Exercise/tiny-eng/reliquary_work/dataset/openmath_data")
dataset = dataset.shuffle(seed=1234)

dataset = dataset.select(range(100))

print(dataset.column_names)

results = []
token_lengths = []

for idx, sample in enumerate(tqdm(dataset, desc="Tokenizing solutions")):
    solution_text = sample.get("generated_solution", "")

    tokens = tokenizer.encode(solution_text, add_special_tokens=False)
    token_count = len(tokens)

    token_lengths.append(token_count)

    results.append({
        "index": idx,
        "token_length": token_count,
    })

print("\n" + "="*60)
print("📊 TOKEN LENGTH STATISTICS (Qwen3.5-4B)")
print("="*60)

total_tokens = sum(token_lengths)
avg_tokens = np.mean(token_lengths)
median_tokens = np.median(token_lengths)
min_tokens = min(token_lengths)
max_tokens = max(token_lengths)
std_tokens = np.std(token_lengths)

print(f"Total samples: {len(token_lengths)}")
print("="*30 + f"\nToken Statistics" + "="*30)
print(f"\nTotal tokens: {total_tokens}")
print(f"\nAverage: {avg_tokens:.2f}")
print(f"\nMedian: {median_tokens:.0f}")
print(f"\nMin: {min_tokens}")
print(f"\nMax: {max_tokens}")
print(f"\nStd: {std_tokens:.2f}")

with open("token_length_analysis.json", "w") as f:
    json.dump({
        "total_samples": len(results),
        "statistics": {
            "total_tokens": total_tokens,
            "average": avg_tokens,
            "median": median_tokens,
            "min": min_tokens,
            "max": max_tokens,
            "std_dev": std_tokens
        },
        "results": results
    }, f, indent=2)

print(f"\n Results saved to token_length_analysis.json")