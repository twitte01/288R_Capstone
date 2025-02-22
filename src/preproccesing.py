import os
from collections import defaultdict

# Define paths
data_dir = "data/images/Speech Commands (trimmed)"  # Root dataset folder
val_list_path = "docs/validation_list.txt"
test_list_path = "docs/testing_list.txt"
train_list_path = "docs/training_list.txt"

# Function to load filenames from validation and test lists
def load_filenames(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(line.strip().replace(".wav", ".png") for line in f.readlines())
    return set()  # Return empty set if file doesn't exist

# Load validation and test filenames
val_filenames = load_filenames(val_list_path)
test_filenames = load_filenames(test_list_path)

# Combine them to avoid duplicates
excluded_files = val_filenames.union(test_filenames)

# Prepare training list and class distributions
train_filenames = []
class_counts = {
    "train": defaultdict(int),
    "validation": defaultdict(int),
    "test": defaultdict(int)
}

# Iterate through dataset folders (each folder is a word category)
for word_class in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, word_class)

    # Skip irrelevant folders
    if not os.path.isdir(class_path) or word_class in [".DS_Store", "_background_noise_"]:
        continue

    # Iterate through image files
    for filename in sorted(os.listdir(class_path)):
        if filename.endswith(".png"):  # Ensure it's an image
            file_path = os.path.join(word_class, filename)  # Relative path
            
            if file_path in val_filenames:
                class_counts["validation"][word_class] += 1
            elif file_path in test_filenames:
                class_counts["test"][word_class] += 1
            else:
                train_filenames.append(file_path)
                class_counts["train"][word_class] += 1

# Save the training file list
with open(train_list_path, "w") as f:
    for item in train_filenames:
        f.write(f"{item}\n")

# Print dataset statistics
print(f"Training list saved to {train_list_path} with {len(train_filenames)} samples.\n")

print("Dataset Class Distribution:")
for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()} SET:")
    for word_class, count in sorted(class_counts[split].items()):
        print(f"  {word_class}: {count} samples")

# Print total counts
print("\nTotal Samples:")
print(f"  Train: {sum(class_counts['train'].values())}")
print(f"  Validation: {sum(class_counts['validation'].values())}")
print(f"  Test: {sum(class_counts['test'].values())}")
