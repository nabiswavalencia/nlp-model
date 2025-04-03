import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

# Load dataset
dataset_path = "luo-swahili.csv"
df = pd.read_csv(dataset_path, encoding="ISO-8859-1")

# Drop empty columns and handle missing values
df = df.dropna(axis=1, how="all").loc[:, (df != "").any(axis=0)]
df["luo"] = df["luo"].astype(str).fillna("")
df["swahili"] = df["swahili"].astype(str).fillna("")

# Label encoding for Kiswahili translation output
label_encoder = LabelEncoder()
df["swahili_Label"] = label_encoder.fit_transform(df["swahili"])

# Tokenization (simple word-based)
def tokenize(sentence):
    return sentence.lower().split()

# Build vocabulary for Luo text
all_words = [word for sentence in df["luo"] for word in tokenize(sentence)]
word_freq = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_freq.items())}  # Reserve 0 for padding
vocab["<PAD>"] = 0  # Padding token
vocab_size = len(vocab)

# Convert Luo text to sequences
max_seq_length = 10  # Limit sentence length
def text_to_sequence(text):
    tokens = tokenize(text)
    sequence = [vocab.get(word, 0) for word in tokens][:max_seq_length]
    return sequence + [0] * (max_seq_length - len(sequence))  # Pad

# Prepare data tensors
X_sequences = [text_to_sequence(text) for text in df["luo"]]
y_labels = df["swahili_Label"].tolist()

X_tensor = torch.tensor(X_sequences, dtype=torch.long)
y_tensor = torch.tensor(y_labels, dtype=torch.long)

# Custom Dataset
class LuoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader
batch_size = 32
dataset = LuoDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a simple model
class LuoTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# Set device (CPU)
device = torch.device("cpu")

# Model setup
output_dim = len(set(y_labels))  # Number of Kiswahili labels
model = LuoTextClassifier(vocab_size, output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
print("🚀 Training model on CPU...")
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

print("✅ Training complete!")


def predict_translation(model, sentence, vocab, label_encoder):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Convert text to tensor
        sequence = text_to_sequence(sentence)
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

        # Get model prediction
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

        # Convert back to Kiswahili text
        kiswahili_translation = label_encoder.inverse_transform([predicted_label])[0]
        return kiswahili_translation

# Save trained model
torch.save(model.state_dict(), "luo_to_swahili_model.pth")
print("✅ Model saved as 'luo_to_swahili_model.pth'")

# Interactive translation
while True:
    user_input = input("\n📝 Enter a Luo sentence (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("👋 Exiting... Goodbye!")
        break

    translation = predict_translation(model, user_input, vocab, label_encoder)
    print(f"✅ Kiswahili Translation: {translation}")
