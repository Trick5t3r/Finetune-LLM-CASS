import os
import torch
import evaluate 
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

NUMBER_OF_FILES = 100

# Chemin vers le modèle et les données
MODEL_DIR = "./finetuned-t5-summarizer"  # Modèle fine-tuné
DATA_DIR = "./CASS-dataset/cleaned_files/"  # Données à évaluer

# Charger le modèle et le tokenizer
print("\n Chargement du modèle...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modèle chargé sur : {device}")

# Charger les fichiers
def load_corpus(data_dir, start_index=7000, max_files=100):
    texts = []
    summaries = []

    filenames = sorted(os.listdir(data_dir))
    for idx, filename in enumerate(filenames):
        if idx < start_index:
            continue

        if idx >= start_index + max_files:
            break 

        if filename.endswith(".story"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                if "@highlight" in content:
                    text, summary = content.split("@highlight", 1)
                    texts.append(text.strip())
                    summaries.append(summary.strip())

    return Dataset.from_dict({"text": texts, "summary": summaries})

print("\n Chargement des données...")
eval_dataset = load_corpus(DATA_DIR, start_index=7000, max_files=NUMBER_OF_FILES)
print(f"Nombre de fichiers chargés : {len(eval_dataset)}")

#  Générer des résumés
def generate_summary(text):
    # Tokenize input with "summarize:" prefix
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    #inputs = {k: v.to(device) for k, v in inputs.items()}

    summary_ids = model.generate(
        inputs,
        max_length=200,
        num_beams=5,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Métriques ROUGE et BLEU
print("\n Évaluation du modèle...")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

generated_summaries = []
references = []

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
bleu_scores = []

for idx, example in enumerate(eval_dataset):
    input_text = example["text"]
    reference_summary = example["summary"]

    # Générer un résumé
    generated_summary = generate_summary(input_text)

    generated_summary_str = generated_summary.strip()
    reference_summary_str = reference_summary.strip()
    print()
    print("=")
    print("Résumé généré : ", generated_summary_str)
    print()
    print("Résumé de référence : ", reference_summary_str)

    generated_summaries.append(generated_summary_str)
    references.append(reference_summary_str)

    rouge_result = rouge_metric.compute(predictions=[generated_summary_str], references=[reference_summary_str])

    bleu_result = bleu_metric.compute(
        predictions=[generated_summary_str], 
        references=[reference_summary_str]
    )

    rouge1_scores.append(rouge_result['rouge1'])
    rouge2_scores.append(rouge_result['rouge2'])
    rougeL_scores.append(rouge_result['rougeL'])
    bleu_scores.append(bleu_result['bleu'])

    print(f"\n Fichier #{idx+1}/{len(eval_dataset)} : ROUGE-L ({rouge_result['rougeL']:.4f}) | BLEU ({bleu_result['bleu']:.4f})")

# Résultats
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
avg_bleu = sum(bleu_scores) / len(bleu_scores)


print("\n Scores moyens d'évaluation :")
print(f"ROUGE-1: {avg_rouge1:.4f}")
print(f"ROUGE-2: {avg_rouge2:.4f}")
print(f"ROUGE-L: {avg_rougeL:.4f}")
print(f"BLEU: {avg_bleu:.4f}")

print("\n Évaluation terminée.")
