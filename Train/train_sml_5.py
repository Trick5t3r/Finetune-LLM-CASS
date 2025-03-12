import os
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

from evaluate import load
rouge = load("rouge")

# Détection du device (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Initialisation...")
print("Device :", device)

# Paramètres et chemins
TOTAL_MAX_ROWS = 5000
DATA_DIR = "./data/cleaned_files_llm/"  # Dossier contenant les fichiers CSV

# Préfixe détaillé utilisé pour l'entraînement ET l'inférence
DETAILED_PREFIX = (
    "Vous êtes un expert en synthèse de texte. Veuillez fournir un résumé détaillé et complet du texte suivant. "
    "Assurez-vous d'inclure tous les points clés, les détails importants et l'essence générale du contenu. "
    "Texte : "
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Décodage des prédictions et labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calcul de la métrique rouge
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    rougeL = result["rougeL"]
    return {"eval_rougeL": rougeL}

# Fonction pour charger tous les fichiers CSV et les concaténer dans un seul dataset
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

# Chargement du modèle T5 et de son tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Configuration de LoRA pour le fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)
model = get_peft_model(model, peft_config)

# Fonction de prétraitement du dataset en intégrant le préfixe détaillé
def preprocess_function(examples, max_total_tokens=2048):
    print("Prétraitement des données...")
    inputs = [DETAILED_PREFIX + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_total_tokens - 512, truncation=True, padding="max_length")
    
    labels = tokenizer(examples["summary"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Démarrage du script...")

# Chargement de tous les CSV d'un coup
full_dataset = load_all_corpus_csv(DATA_DIR, total_max_rows=TOTAL_MAX_ROWS)

# Split du dataset en train et eval (par exemple, 90% train et 10% eval)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Prétraitement des données (sans multiprocessing)
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print(tokenized_eval_dataset)

# Définition des arguments d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=500,
    save_strategy="epoch",
    label_names=["labels"],
    warmup_steps=500,
    metric_for_best_model="eval_loss"
)

# Initialisation du DataCollator et des callbacks (EarlyStopping)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

# Initialisation du Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    callbacks=callbacks,
    compute_metrics=compute_metrics,
)

# Lancement de l'entraînement
print("Entraînement en cours...")
trainer.train()

# Sauvegarde du modèle fine-tuné et du tokenizer
trainer.save_model("./finetuned_sml_V4_llm")
tokenizer.save_pretrained("./finetuned_sml_V4_llm")
print("Fine-tuning terminé et modèle sauvegardé.")
