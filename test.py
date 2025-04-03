# import os
# import json
# import torch
# import torch.nn as nn
#
# # Load vocab
# if os.path.exists("luo_vocab.json") and os.path.exists("swahili_vocab.json"):
#     with open("luo_vocab.json", "r") as f:
#         luo_vocab = json.load(f)
#     print(f"Luo vocab size: {len(luo_vocab)} (Expected: 30472)")
#     with open("swahili_vocab.json", "r") as f:
#         swahili_vocab = json.load(f)
# else:
#     print("⚠️ Vocabulary files not found! Run `generate_vocab.py` first.")
#     exit()
#
# # Function to tokenize Luo text
# def tokenize(text):
#     tokens = text.lower().split()
#     return [luo_vocab.get(word, luo_vocab["<unk>"]) for word in tokens]
#
# # Function to detokenize model output
# def detokenize(output_tensor):
#     predicted_ids = output_tensor.argmax(dim=-1).tolist()
#     words = [swahili_vocab.get(str(idx), "<unk>") for idx in predicted_ids]
#     return " ".join(words)
#
# # ✅ Define model with correct structure
# class TranslationModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(TranslationModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 🔹 Use "lstm" to match training
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         embedded = self.embedding(x)
#         _, (hidden, _) = self.lstm(embedded)
#         return self.fc(hidden.squeeze(0))
#
# # Load trained model
# model = TranslationModel(vocab_size=len(luo_vocab), embedding_dim=64, hidden_dim=128, output_dim=28468)
# model_path = "luo_to_swahili_model.pth"
#
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     print("✅ Pre-trained model loaded!")
# else:
#     print("⚠️ No trained model found! Train it first.")
#     exit()
#
# # User input loop
# while True:
#     luo_sentence = input("📝 Enter a Luo sentence (or type 'exit' to quit): ")
#     if luo_sentence.lower() == "exit":
#         break
#
#     input_tensor = torch.tensor(tokenize(luo_sentence), dtype=torch.long).unsqueeze(0)
#
#     with torch.no_grad():
#         output = model(input_tensor)
#
#     swahili_translation = detokenize(output)
#     print(f"✅ Kiswahili Translation: {swahili_translation}")


import os
import json
import torch
import torch.nn as nn

# Load vocab
if os.path.exists("luo_vocab.json") and os.path.exists("swahili_vocab.json"):
    with open("luo_vocab.json", "r") as f:
        luo_vocab = json.load(f)
    print(f"Luo vocab size: {len(luo_vocab)}") #removed the expected size.
    with open("swahili_vocab.json", "r") as f:
        swahili_vocab = json.load(f)
    print(f"Swahili vocab size: {len(swahili_vocab)}")
else:
    print("⚠️ Vocabulary files not found! Run `generate_vocab.py` first.")
    exit()

# Function to tokenize Luo text
def tokenize(text):
    tokens = text.lower().split()
    return [luo_vocab.get(word, luo_vocab["<unk>"]) for word in tokens]

# Function to detokenize model output
def detokenize(output_tensor):
    predicted_ids = output_tensor.argmax(dim=-1).tolist()
    words = [swahili_vocab.get(str(idx), "<unk>") for idx in predicted_ids]
    return " ".join(words)

# ✅ Define model with correct structure
class TranslationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 🔹 Use "lstm" to match training
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Load trained model
model = TranslationModel(vocab_size=len(luo_vocab), embedding_dim=64, hidden_dim=128, output_dim=len(swahili_vocab)) #changed the output_dim to be the length of the swahili vocab.
model_path = "luo_to_swahili_model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        print("✅ Pre-trained model loaded!")
    except RuntimeError as e:
        print(f"⚠️ Error loading model: {e}")
        print("⚠️ Ensure the vocabulary sizes match, or retrain the model.")
        exit()
else:
    print("⚠️ No trained model found! Train it first.")
    exit()

# User input loop
while True:
    luo_sentence = input("📝 Enter a Luo sentence (or type 'exit' to quit): ")
    if luo_sentence.lower() == "exit":
        break

    input_tensor = torch.tensor(tokenize(luo_sentence), dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    swahili_translation = detokenize(output)
    print(f"✅ Kiswahili Translation: {swahili_translation}")