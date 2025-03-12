#!/usr/bin/env python
import os
import math
import random
import numpy as np
import pandas as pd
import torch
import nltk
nltk.download('punkt', quiet=True)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from evaluate import load as load_metric

# -------------------------------
# Set Seeds for Reproducibility
# -------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------
# Paramètres et chemins
# -------------------------------
TOTAL_MAX_ROWS = 5000
DATA_DIR = "./cleaned_files_llm/"  # Dossier contenant vos fichiers CSV
DETAILED_PREFIX = (
    "Vous êtes un expert en synthèse de texte. Veuillez fournir un résumé détaillé et complet du texte suivant. "
    "Assurez-vous d'inclure tous les points clés, les détails importants et l'essence générale du contenu. "
    "Texte : "
)

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_all_corpus_csv(data_dir, total_max_rows=TOTAL_MAX_ROWS):
    texts = []
    summaries = []
    filenames = sorted(os.listdir(data_dir))
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        texts.extend(df["text"].tolist())
        summaries.extend(df["generated_summary"].tolist())
    # Limiter le nombre total de lignes si nécessaire
    if total_max_rows is not None:
        texts = texts[:total_max_rows]
        summaries = summaries[:total_max_rows]
    return Dataset.from_dict({
        "text": texts,
        "summary": summaries,
    })

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_function(examples, max_total_tokens=2048):
    # Append "\n Résumé :" after the text prompt
    inputs = [DETAILED_PREFIX + doc + "\n Résumé :" for doc in examples["text"]]
    model_inputs = tokenizer(
        inputs, 
        max_length=max_total_tokens - 512, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        examples["summary"], 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# -------------------------------
# Helper Function: Extract Generated Summary
# -------------------------------
def extract_summary(text):
    marker = "\n Résumé :"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()

# -------------------------------
# Metrics Function
# -------------------------------
rouge_metric = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Extract only the text after "\n Résumé :" for each prediction
    extracted_preds = [extract_summary(pred) for pred in decoded_preds]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=extracted_preds, references=decoded_labels)
    rougeL = result["rougeL"].mid.fmeasure
    return {"eval_rougeL": rougeL}

# -------------------------------
# Utility: Calculate Model Size
# -------------------------------
def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = num_params * 4  # Assuming float32 (4 bytes per parameter)
    size_in_mb = size_in_bytes / (1024 ** 2)
    size_in_gb = size_in_mb / 1024
    return size_in_mb, size_in_gb

# -------------------------------
# Main Training Function
# -------------------------------
def main():
    # Load dataset from CSV files
    print("Loading dataset from CSV files...")
    dataset = load_all_corpus_csv(DATA_DIR, total_max_rows=TOTAL_MAX_ROWS)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    print("Train dataset length:", len(dataset["train"]))
    print("Validation dataset length:", len(dataset["test"]))
    
    # Load tokenizer and model (Bloom) with LoRA
    model_name = "bigscience/bloom-1b1"
    cache_dir = "/path/to/models_cache"  # Update with your cache directory path
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    print("Base model loaded")
    
    # Configure and apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adaptation applied")
    model.print_trainable_parameters()
    
    model_size_mb, model_size_gb = get_model_size(model)
    print(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")
    
    torch.cuda.empty_cache()
    print("CUDA MEMORY has been freed")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling objective
        pad_to_multiple_of=2048,
    )
    
    # Define training arguments
    output_dir = "./fine_tuning_bloom_good_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    num_epochs = 5
    batch_size = 8
    learning_rate = 1e-5
    weight_decay = 0.01
    save_steps = 500
    fp16 = True
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_steps=save_steps,
        fp16=fp16,
        logging_steps=100,
        save_total_limit=2,
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed. Model saved at {output_dir}")

if __name__ == "__main__":
    main()
