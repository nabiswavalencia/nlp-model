# 🚀 Luo-Swahili Translation Model (mT5-small)
# ✅ Full fine-tuning script for PyCharm

# 📌 Step 1: Install Dependencies
import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import sentencepiece

# 📌 Step 2: Load Dataset
file_path = "luo-swahili.csv"  # Ensure this file is in your PyCharm project directory
df = pd.read_csv(file_path, encoding="ISO-8859-1")
df.columns = df.columns.str.strip().str.lower()
df = df[['luo', 'swahili']]  # Keep only relevant columns
df = df.dropna()  # Remove rows with missing values
print(df.head())  # Check the first few rows of the dataset
print(df.info())  # See if the columns are correctly detected
print(df.shape)   # Confirm the number of rows and columns



# Ensure column names are correct
assert "luo" in df.columns and "swahili" in df.columns, "Columns should be named 'luo' and 'kiswahili'"

# 📌 Step 3: Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 📌 Step 4: Split Dataset (80% Training, 20% Testing)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 📌 Step 5: Load Tokenizer
# MODEL_NAME = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", legacy=False, padding=True, truncation= True)

# 📌 Step 6: Tokenization Function
# Tokenization function with data cleaning
# def tokenize_function(example):
#     # Ensure both fields are strings and handle missing values
#     luo_text = str(example["luo"]) if example["luo"] else ""
#     swahili_text = str(example["kiswahili"]) if example["kiswahili"] else ""
#
#     return tokenizer(
#         example["luo"],
#         text_target=example["kiswahili"],
#         padding="max_length",  # Ensures all sequences have the same length
#         truncation=True,
#         max_length=128  # You can adjust this value
#     )# Check for missing values
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["luo"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    labels = tokenizer(
        examples["swahili"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )["input_ids"]

    model_inputs["labels"] = labels  # Assign labels
    return model_inputs


print(df.isnull().sum())

# Ensure all values in columns are strings
df["luo"] = df["luo"].astype(str)
df["swahili"] = df["swahili"].astype(str)
df["luo"] = df["luo"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

df = df.fillna("")

# Apply tokenization
sample_text = df["luo"].iloc[0]  # Pick first Luo sentence
tokenized_sample = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128)
print(tokenized_sample)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# tokenized_dataset = dataset.map(preprocess_function, batched=True)
# 📌 Step 7: Load Model
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# 📌 Step 8: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./mt5-luo",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,  # Adjust based on training time
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,  # Only keep the latest 2 models to save space
    remove_unused_columns=False,
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if GPU available
)

# 📌 Step 9: Initialize Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 📌 Step 10: Start Training
trainer.train()

# 📌 Step 11: Save the Trained Model
model.save_pretrained("./mt5-luo-final")
tokenizer.save_pretrained("./mt5-luo-final")

print("✅ Training Complete! Model saved in './mt5-luo-final'")
