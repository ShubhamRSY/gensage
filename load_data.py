from datasets import load_dataset

# Load the LEDGAR dataset
dataset = load_dataset("lex_glue", "ledgar")

# Use full training data
train_data = dataset["train"]

# Optional: preview length and sample
print("Total samples in train set:", len(train_data))
print("\nSample document:\n")
print(train_data[0]["text"])
