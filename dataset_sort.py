from datasets import Dataset
from collections import Counter

dataset = Dataset.load_from_disk("/mnt/d/models/reliquary/dataset")

data = list(dataset)

source_counts = Counter(item["problem_source"] for item in data)
total = len(data)

print("=" * 80)
print("SIZE OF EACH PROBLEM_SOURCE TYPE")
print("=" * 80)

for source, count in sorted(source_counts.items()):
    percentage = (count / total) * 100
    bar = "█" * int(percentage / 2)
    print(f"{source:30s}: {count:6d} rows ({percentage:5.1f}%) {bar}")

print("-" * 80)
print(f"{'TOTAL':30s}: {total:6d} rows (100.0%)")
print(f"{"UNIQUE TYPES":30s}: {len(source_counts):6d}")