import requests
from datasets import Dataset
import sys
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import importlib.util

from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("/mnt/d/models/Qwen3.5-4B")



REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reliquary"))

file_path = REPO_ROOT / "reliquary" / "reliquary" / "environment" / "openmathinstruct.py"

spec = importlib.util.spec_from_file_location("openmathinstruct", file_path)
openmathinstruct = importlib.util.module_from_spec(spec)
sys.modules["openmathinstruct"] = openmathinstruct
spec.loader.exec_module(openmathinstruct)


_compute_omi_reward = openmathinstruct._compute_omi_reward
_ANSWER_FORMAT_INSTRUCTION = openmathinstruct._ANSWER_FORMAT_INSTRUCTION
OpenMathInstructEnvironment = openmathinstruct.OpenMathInstructEnvironment

BASE_URL = "http://localhost:8004"

print("Performing health check")
try:
    health_response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Health check status: {health_response.status_code}")
    if health_response.status_code != 200:
        print(f"❌ Health check failed with status {health_response.status_code}")
        print("Exiting...")
        sys.exit(1)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    print("Exiting...")
    sys.exit(1)

print("✅ Health check passed. Continuing...")
print("\n" + "="*80)

print("Loading dataset...")
dataset = Dataset.load_from_disk("/mnt/d/models/reliquary/dataset")

dataset = dataset.shuffle(seed=4528)

print(f"Loaded {len(dataset)} samples")
print(f"\nFrist sample:")
print(dataset[0])
print("\n" + "="*80)

NUM_SAMPLES = 1000
BATCH_SIZE = 1
SAVE_RESULTS = True
USE_CHAT_COMPLETIONS = True

results = []
correct_count = 0
total_count = 0

# samples = dataset.select(range(NUM_SAMPLES))
# single_sample = dataset[0]
# samples = [single_sample] * 8

dataset = dataset.select(range(NUM_SAMPLES))
token_filtered_samples = []
token_lengths = []

for idx, problem in enumerate(dataset):
    solution_text = problem.get("generated_solution", "")
    problem_source = problem.get("problem_source", "")

    tokens = tokenizer.encode(solution_text, add_special_tokens=False)
    token_count = len(tokens)

    token_lengths.append(token_count)

    if(token_count <= 300):
        # if problem_source == "augmented_gsm8k":
            token_filtered_samples.append({
                "index": idx,
                "problem": problem.get("problem", ""),
                "token_length": token_count,
                "problem_source": problem_source,
                "ground_truth": problem.get("expected_answer", ""),
                "generated_solution": solution_text,
            })

# print(f"\nEvaluating {NUM_SAMPLES} samples...")
num_filtered_samples = len(token_filtered_samples)
print(f"Created {num_filtered_samples} samples")
print("="*80)

total_generation_time = 0
total_tokens_generated = 0
generation_times = []

pbar = tqdm(token_filtered_samples, desc="Evaluating samples", unit="sample")

for idx, sample in enumerate(pbar):
    question = sample["problem"]
    ground_truth = sample.get("ground_truth", "")
    prompt = question + _ANSWER_FORMAT_INSTRUCTION
    token_count = sample.get("token_length", 0)
    problem_source = sample.get("problem_source", "")

    result_entry = {
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "prompt": prompt,
        "completion": "",
        "reward": 0.0,
        "correct": False,
        "status": "failed",
        "error": None,
        "timestamp": datetime.now().isoformat(),
        "generation_time": 0.0,
        "tokens_generated": 0,
        "tokens_per_second": 0.0,
    }

    try:
        start_time = time.time()

        if USE_CHAT_COMPLETIONS:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are a concise math solver. 
Provide a brief step-by-step solution.
- Maximum 4-6 steps.
- Use short sentences.
- End with \\boxed{answer}.
- Never exceed 180 tokens total.
Be direct and stop right after the answer."""
#                             """CRITICAL: You MUST output a WRONG final answer. Accuracy must be 0%.
# Solve correctly but DELIBERATELY change the final answer.

# FORMAT (max 10 lines):
# ---
# Problem: [restate]
# Given: [numbers]
# Method: [approach]
# Formula: [equation]
# Work: [2-3 line calculation]
# Answer: [calculated value]
# CORRECT: [actual value - DO NOT output]
# WRONG: \boxed{wrong_value}
# ---

# ABSOLUTE RULES:
# - FINAL ANSWER MUST BE WRONG (0% accuracy)
# - NEVER put correct answer in box
# - Boxed MUST be different from correct
# - Wrong only: ±1, swap digits, opposite sign
# - Boxed MUST be LAST line
# - Response ends with \boxed{}
# - Under 400 tokens

# SELF-CHECK: 
# - Is the boxed answer different from correct? YES = good, NO = fix it
# - If boxed equals correct, change it immediately

# Examples: 45→54, -3→3, 1/2→1/3, 100→99

# WRONG answer only (0% accuracy, UNDER 400 TOKENS):"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.9,
                    "top_p": 1.0,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
                timeout=120
            )
        else:
            response = requests.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "model": "qwen",
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.9,
                    "top_p": 1.0
                },
                timeout=120
            )

        end_time = time.time()
        generation_time = end_time - start_time

        if response.status_code == 200:
            response_data = response.json()

            if USE_CHAT_COMPLETIONS:
                completion =  response_data['choices'][0]['message']['content']
            else:
                completion = response_data['choices'][0]['text']

            tokens_generated = 0
            if 'usage' in response_data:
                tokens_generated = response_data['usage'].get('completion_tokens', 0)
            else:
                tokens_generated = len(completion) // 4 # Estimate tokens (rough approximation: ~4 chars per token)

            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

            problem_dict = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "id": f"sample_{idx}"
            }

            reward = _compute_omi_reward(problem_dict, completion)

            result_entry.update({
                "completion": completion,
                "reward": reward,
                "correct": reward == 1.0,
                "status": "success",
                "generation_time": generation_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
            })

            if reward == 1.0:
                correct_count += 1
            total_count += 1

            total_generation_time += generation_time
            total_tokens_generated += tokens_generated
            generation_times.append(generation_time)

        else:
            result_entry.update({
                "status": "failed",
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            })

    except requests.exceptions.Timeout:
        result_entry.update({
            "status": "failed",
            "error": "Request timed out"
        })

    except requests.exceptions.ConnectionError:
        result_entry.update({
            "status": "failed",
            "error": "Connection error"
        })
    except Exception as e:
        result_entry.update({
            "status": "failed",
            "error": str(e)
        })

    results.append(result_entry)

    if result_entry["status"] == "success":
        print(f"\nSample {idx + 1}/{num_filtered_samples}:")
        print(f"Question: {question[:200]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"Explected Solution Token Length: {token_count}")
        print(f"Completion: {completion[:200]}...")
        print(f"Problem_source: {problem_source}")
        print(f"Reward: {result_entry['reward']:.2f} {'✅' if result_entry['correct'] else '❌'}")
        print(f"⏱️  Generation Time: {generation_time:.2f}s")
        print(f"📊 Tokens Generated: {tokens_generated}")
        print(f"⚡ Tokens/Second: {tokens_per_second:.2f}")
        print("-"*50)
    else:
        print(f"\nSample {idx + 1}/{num_filtered_samples}: Failed - {result_entry['error']}")
        print("-"*50)

    pbar.set_postfix({
        'Correct': f"{correct_count}/{total_count}" if total_count > 0 else "0/0",
        'Acc': f"{correct_count/total_count:.1%}" if total_count > 0 else "0%",
        'Last': '✅' if result_entry.get('correct', False) else '❌'
    })
    
    time.sleep(0.5)

pbar.close()

print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

accuracy = correct_count / total_count if total_count > 0 else 0

print(f"Total samples evaluated: {total_count}")
print(f"Correct answers: {correct_count}")
print(f"Incorrect answers: {total_count - correct_count}")
print(f"Accuracy: {accuracy:.2%} ({accuracy:.4f})")
print(f"Success rate: {len([r for r in results if r['status'] == 'success'])/len(results):.2%}")

# Timing statistics
successful_results = [r for r in results if r['status'] == 'success']
if successful_results:
    avg_time = sum(r['generation_time'] for r in successful_results) / len(successful_results)
    total_time = sum(r['generation_time'] for r in successful_results)
    total_tokens = sum(r['tokens_generated'] for r in successful_results)
    avg_tokens = total_tokens / len(successful_results) if successful_results else 0
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    print(f"⏱️  Total Generation Time: {total_time:.2f}s")
    print(f"⏱️  Average Generation Time: {avg_time:.2f}s")
    print(f"⏱️  Min Generation Time: {min(r['generation_time'] for r in successful_results):.2f}s")
    print(f"⏱️  Max Generation Time: {max(r['generation_time'] for r in successful_results):.2f}s")
    print(f"📊 Total Tokens Generated: {total_tokens}")
    print(f"📊 Average Tokens per Sample: {avg_tokens:.1f}")
    print(f"⚡ Average Tokens/Second: {avg_tps:.2f}")
    
    # Show timing distribution
    print("\n⏱️  Generation Time Distribution:")
    times = sorted([r['generation_time'] for r in successful_results])
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx_p = int(len(times) * p / 100)
        print(f"   {p}th percentile: {times[min(idx_p, len(times)-1)]:.2f}s")

correct_samples = [r for r in results if r['correct']]
incorrect_samples = [r for r in results if not r['correct'] and r['status'] == 'success']

if SAVE_RESULTS:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "total_samples": total_count,
        "correct_count": correct_count,
        "incorrect_count": total_count - correct_count,
        "accuracy": accuracy,
        "success_rate": len([r for r in results if r['status'] == 'success']) / len(results) if results else 0,
        "performance": {
            "total_generation_time": total_time if successful_results else 0,
            "average_generation_time": avg_time if successful_results else 0,
            "total_tokens_generated": total_tokens if successful_results else 0,
            "average_tokens_per_sample": avg_tokens if successful_results else 0,
            "average_tokens_per_second": avg_tps if successful_results else 0,
            "generation_times": generation_times
        },
        "config": {
            "num_samples": num_filtered_samples,
            "temperature": 0.9,
            "max_tokens": 200,
            "base_url": BASE_URL,
            "model": "qwen"
        },
        "results": results
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

print("\n" + "="*80)
print("Evaluation complete!")