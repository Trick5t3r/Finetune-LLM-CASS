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

# Détection du device (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Initialisation...")

# Paramètres et chemins
NUMBER_OF_FILES = 5000
NUMBER_OF_FILES_TEST = 1000
DATA_DIR = "./CASS-dataset/cleaned_files_llm/"  # Dossier contenant les fichiers CSV

# Préfixe détaillé utilisé pour l'entraînement ET l'inférence
DETAILED_PREFIX = (
    "Vous êtes un expert en synthèse de texte. Veuillez fournir un résumé détaillé et complet du texte suivant. "
    "Assurez-vous d'inclure tous les points clés, les détails importants et l'essence générale du contenu. "
    "Texte : "
)

# Fonction pour charger et préparer le corpus
def load_corpus_csv(data_dir, start_index=0, max_files=100):
    texts = []
    llm_summaries = []

    filenames = sorted(os.listdir(data_dir))
    count = 0
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        # Si le fichier CSV contient plusieurs lignes, on les traite toutes
        for _, row in df.iterrows():
            # Appliquer le start_index et la limite max sur le nombre d'exemples globaux
            if count < start_index:
                count += 1
                continue
            if count >= start_index + max_files:
                break
            texts.append(row["text"])
            llm_summaries.append(row["generated_summary"])
            count += 1
        if count >= start_index + max_files:
            break
    return Dataset.from_dict({
        "text": texts,
        "summary": llm_summaries,
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
    model_inputs = tokenizer(inputs, max_length=max_total_tokens-512, truncation=True, padding="max_length")
    
    labels = tokenizer(examples["summary"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    print("Démarrage du script...")

    # Chargement des datasets d'entraînement et d'évaluation
    train_dataset = load_corpus_csv(DATA_DIR, start_index=0, max_files=NUMBER_OF_FILES)
    eval_dataset = load_corpus_csv(DATA_DIR, start_index=6000, max_files=NUMBER_OF_FILES_TEST)

    # Application du prétraitement avec multiprocessing
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=4)

    # Définition des arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir='./logs',
        logging_steps=500,
        save_strategy="epoch",
        label_names=["labels"],
        warmup_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False
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
    )

    # Lancement de l'entraînement
    print("Entraînement en cours...")
    trainer.train()

    # Sauvegarde du modèle fine-tuné et du tokenizer
    trainer.save_model("./finetuned_sml_V3_llm")
    tokenizer.save_pretrained("./finetuned_sml_V3_llm")
    print("Fine-tuning terminé et modèle sauvegardé.")
