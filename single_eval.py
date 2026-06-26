import requests
from datasets import Dataset
import os
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
import math

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


LOOP_NUM = 10

BATCH_SIZE = 1
SAVE_RESULTS = True
USE_CHAT_COMPLETIONS = True

NUM_SAMPLES = 20
TEMPERATURE = 3.5
SEED = 421754


print("✅ Health check passed. Continuing...")
print("\n" + "="*80)

print("Loading dataset...")
dataset = Dataset.load_from_disk("/mnt/d/models/reliquary/dataset")

dataset = dataset.shuffle(seed=SEED)
dataset = dataset.select(range(200))
# token_filtered_samples = []
# token_lengths = []

# for idx, problem in enumerate(dataset):
#     solution_text = problem.get("generated_solution", "")
#     problem_source = problem.get("problem_source", "")

#     tokens = tokenizer.encode(solution_text, add_special_tokens=False)
#     token_count = len(tokens)

#     token_lengths.append(token_count)

#     if(token_count <=300):
#         token_filtered_samples.append({
#             "index": idx,
#             "problem": problem.get("problem", ""),
#             "token_length": token_count,
#             "problem_source": problem_source,
#             "ground_truth": problem.get("expected_answer", ""),
#             "generated_solution": solution_text,
#         })

print(f"Loaded {len(dataset)} samples")
print(f"\nFrist sample:")
print(dataset[0])
print("\n" + "="*80)


# samples = dataset.select(range(NUM_SAMPLES))
correct_per_sample = []

for i in range(LOOP_NUM):
    results = []
    correct_count = 0
    total_count = 0
    single_sample = dataset[i]
    samples = [single_sample] * NUM_SAMPLES

    # print(f"\nEvaluating {NUM_SAMPLES} samples...")
    print(f"Created {len(samples)} samples")
    print("="*80)

    total_generation_time = 0
    total_tokens_generated = 0
    generation_times = []

    pbar = tqdm(samples, desc="Evaluating samples", unit="sample")

    for idx, sample in enumerate(pbar):
        question = sample["problem"]
        ground_truth = sample.get("expected_answer", "")
        prompt = question + _ANSWER_FORMAT_INSTRUCTION
        token_length = sample.get("token_length")

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
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": TEMPERATURE,
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
                        "temperature": TEMPERATURE,
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
            print(f"\nSample {idx + 1}/{NUM_SAMPLES}:")
            print(f"Question: {question[:200]}...")
            print(f"Ground Truth: {ground_truth}")
            print(f"Completion: {completion[:200]}...")
            print(f"Expected Token Length: {token_length}")
            print(f"Reward: {result_entry['reward']:.2f} {'✅' if result_entry['correct'] else '❌'}")
            print(f"⏱️  Generation Time: {generation_time:.2f}s")
            print(f"📊 Tokens Generated: {tokens_generated}")
            print(f"⚡ Tokens/Second: {tokens_per_second:.2f}")
            print("-"*50)
        else:
            print(f"\nSample {idx + 1}/{NUM_SAMPLES}: Failed - {result_entry['error']}")
            print("-"*50)

        pbar.set_postfix({
            'Correct': f"{correct_count}/{total_count}" if total_count > 0 else "0/0",
            'Acc': f"{correct_count/total_count:.1%}" if total_count > 0 else "0%",
            'Last': '✅' if result_entry.get('correct', False) else '❌',
            'Num': f"{i+1}"
        })
        
        time.sleep(0.5)

    pbar.close()

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    accuracy = correct_count / total_count if total_count > 0 else 0

    correct_result = {
        "correct_count": correct_count,
        "accuracy": accuracy,
        "delta": math.sqrt(accuracy * (1 - accuracy)),
    }


    correct_per_sample.append(correct_result)

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
        
    correct_samples = [r for r in results if r['correct']]
    incorrect_samples = [r for r in results if not r['correct'] and r['status'] == 'success']

    print("\n" + "="*80)

    if SAVE_RESULTS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"./single_eval_seed_{SEED}_{TEMPERATURE}"

        os.makedirs(result_dir, exist_ok=True)
        
        results_file = os.path.join(result_dir, f"evaluation_results_{timestamp}.json")

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
                "num_samples": NUM_SAMPLES,
                "temperature": 0.9,
                "max_tokens": 200,
                "base_url": BASE_URL,
                "model": "qwen"
            },
            "results": results,
            "count_per_sample": correct_result
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")

    print("\n" + "="*80)
    print("Evaluation complete!")


# Add this at the end of your code (after the main loop)

print("\n" + "="*80)
print("EVERY DELTA VALUE")
print("="*80)

# Display each delta with its iteration number
for i, item in enumerate(correct_per_sample):
    accuracy = item['accuracy']
    delta = item['delta']
    correct_count = item['correct_count']
    print(f"Iteration {i+1:3d}: accuracy = {accuracy:.4f}, delta = {delta:.4f}, correct = {correct_count}/{NUM_SAMPLES}")

print("\n" + "="*80)
print("DELTA SUMMARY")
print("="*80)

# Convert to list for analysis
deltas = [item['delta'] for item in correct_per_sample]
accuracies = [item['accuracy'] for item in correct_per_sample]

print(f"Total iterations: {len(deltas)}")
print(f"\nAll delta values:")
for i, delta in enumerate(deltas):
    print(f"  {i+1:3d}: {delta:.6f}")

print(f"\nDelta statistics:")
print(f"  Mean:  {np.mean(deltas):.6f}")
print(f"  Std:   {np.std(deltas):.6f}")
print(f"  Min:   {np.min(deltas):.6f}")
print(f"  Max:   {np.max(deltas):.6f}")
print(f"  Median: {np.median(deltas):.6f}")