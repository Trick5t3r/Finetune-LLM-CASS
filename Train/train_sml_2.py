print("Initialisation...")

import os
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import EarlyStoppingCallback, get_linear_schedule_with_warmup

# Dans vos arguments d'entraînement, vous pouvez définir 'warmup_steps'
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,  # par exemple, augmenter le nombre d'époques
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=500,
    save_strategy="epoch",
    label_names=["labels"],
    warmup_steps=500  # ajout d'un warmup
)


# Params
NUMBER_OF_FILES = 5000
NUMBER_OF_FILES_TEST = 1000

# Chemin vers les fichiers texte
DATA_DIR = "../data/cleaned_files/" 

# Charger et préparer les données
def load_corpus(data_dir, start_index=0, max_files=10):
    texts = []
    summaries = []

    filenames = sorted(os.listdir(data_dir))
    for idx, filename in enumerate(filenames[start_index:start_index + max_files]):
        print(f"Chargement des fichiers ({start_index + idx + 1}/{start_index + max_files})")

        if filename.endswith(".story"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                if "@highlight" in content:
                    text, summary = content.split("@highlight", 1)
                    texts.append(text.strip())
                    summaries.append(summary.strip())

    return Dataset.from_dict({"text": texts, "summary": summaries})

# Charger le modèle T5 et son tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# LoRA pour fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    r=8, 
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"], 
)

model = get_peft_model(model, peft_config)

# Prétraitement du dataset
def preprocess_function(examples):
    print("Prétraitement des données...")
    inputs = ["summarize: " + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    print("Démarrage du script...")

    # Charger le dataset d'entraînement et d'évaluation
    train_dataset = load_corpus(DATA_DIR, start_index=0, max_files=NUMBER_OF_FILES)
    eval_dataset = load_corpus(DATA_DIR, start_index=6000, max_files=NUMBER_OF_FILES_TEST)

    # Appliquer le prétraitement avec multiprocessing
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=4)

    # Configuration des paramètres d'entraînement
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
        metric_for_best_model="eval_loss",  # Ajout de la métrique pour le meilleur modèle
        greater_is_better=False  # Car une perte plus faible est meilleure
    )


    # Initialiser le DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    # Initialiser le Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Lancer l'entraînement
    print("Entraînement en cours...")
    trainer.train()

    # Sauvegarder le modèle fine-tuné
    trainer.save_model("./finetuned-t5-summarizer")
    tokenizer.save_pretrained("./finetuned-t5-summarizer")

    print("Fine-tuning terminé et modèle sauvegardé.")
