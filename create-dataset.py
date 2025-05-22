import json
import random

input_path = "Data/math.jsonl"
output_path = "Data/math_500.jsonl"
sample_size = 500

# Read all lines
with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

# Sample 500 entries
sampled = random.sample(data, sample_size)

# Assign new ids from 0 to 499
for new_id, entry in enumerate(sampled):
    entry["id"] = new_id

# Write to new jsonl
with open(output_path, "w") as f:
    for entry in sampled:
        f.write(json.dumps(entry) + "\n")
