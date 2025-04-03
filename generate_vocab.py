import pandas as pd
import json

# Load the dataset
csv_file = "luo-swahili.csv"  # Change to your actual file path
df = pd.read_csv(csv_file, encoding="ISO-8859-1")

# Ensure column names are correct
df = df.dropna(subset=["luo", "swahili"])  # Remove empty rows

# Create vocabulary mappings
luo_vocab = {word: idx for idx, word in enumerate(set(" ".join(df["luo"]).lower().split()), start=1)}
swahili_vocab = {idx: word for idx, word in enumerate(set(" ".join(df["swahili"]).lower().split()), start=1)}

# Add unknown token
luo_vocab["<unk>"] = 0
swahili_vocab[0] = "<unk>"

# Save vocabularies
with open("luo_vocab.json", "w") as f:
    json.dump(luo_vocab, f)

with open("swahili_vocab.json", "w") as f:
    json.dump(swahili_vocab, f)

print("✅ Vocabulary files saved successfully!")
