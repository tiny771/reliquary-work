from huggingface_hub import hf_hub_download
from datasets import load_dataset

# LOCAL_DIR = "/mnt/d/Exercise/tiny-eng/reliquary_work/dataset/openmath_data"

# path = hf_hub_download(
#     repo_id="nvidia/OpenMathInstruct-2",
#     repo_type="dataset",
#     filename="data/train-00000-of-00032.parquet",
# )

# dataset = load_dataset(
#     "parquet",
#     data_files=path,
#     split="train",
# )

# dataset.save_to_disk(f"{LOCAL_DIR}/")

LOCAL_DIR = "/mnt/d/Exercise/tiny-eng/reliquary_work/dataset/openmath"

dataset = load_dataset('nvidia/OpenMathInstruct-2', split="train")

dataset.save_to_disk(f"{LOCAL_DIR}/")