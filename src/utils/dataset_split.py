import json
import random

# Paths
all_path = "data/all.json"
train_path = "data/train.json"
test_path = "data/test.json"

# Load all data
with open(all_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle data for randomness
random.shuffle(data)

# Split
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# Save train
with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# Save test
with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")