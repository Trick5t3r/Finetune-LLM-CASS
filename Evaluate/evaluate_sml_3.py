import os
import torch
import evaluate
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

NUMBER_OF_FILES = 100

# Chemin vers le modèle et les données
MODEL_DIR = "./finetuned-t5-summarizer"  # Modèle fine-tuné
DATA_DIR = "./CASS-dataset/cleaned_files_llm/"  # Dossier contenant les fichiers CSV

# Charger le modèle et le tokenizer
print("\nChargement du modèle...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modèle chargé sur : {device}")

# Charger le corpus à partir des fichiers CSV
def load_corpus_csv(data_dir, start_index=0, max_files=100):
    texts = []
    llm_summaries = []
    reference_summaries = []

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
            reference_summaries.append(row["reference_summary"])
            count += 1
        if count >= start_index + max_files:
            break
    return Dataset.from_dict({
        "text": texts,
        "llm_summary": llm_summaries,
        "reference_summary": reference_summaries
    })

print("\nChargement des données...")
eval_dataset = load_corpus_csv(DATA_DIR, start_index=10, max_files=NUMBER_OF_FILES)
print(f"Nombre d'exemples chargés : {len(eval_dataset)}")

# Fonction de génération de résumé avec le modèle
def generate_summary(text):
    # Ajouter le préfixe "summarize:" pour la tokenisation
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(
        inputs,
        max_length=200,
        num_beams=5,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Charger les métriques ROUGE et BLEU
print("\nÉvaluation du modèle...")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# Listes pour stocker les scores pour comparaison avec le résumé de référence
rouge1_scores_ref = []
rouge2_scores_ref = []
rougeL_scores_ref = []
bleu_scores_ref = []

# Listes pour stocker les scores pour comparaison avec le résumé généré par le LLM
rouge1_scores_llm = []
rouge2_scores_llm = []
rougeL_scores_llm = []
bleu_scores_llm = []

rouge1_scores_llm_ref = []
rouge2_scores_llm_ref = []
rougeL_scores_llm_ref = []
bleu_scores_llm_ref = []

for idx, example in enumerate(eval_dataset):
    input_text = example["text"]
    llm_summary = example["llm_summary"].strip()
    reference_summary = example["reference_summary"].strip()
    
    # Générer le résumé avec votre modèle
    model_summary = generate_summary(input_text).strip()
    
    print("\n" + "="*50)
    print(f"Exemple #{idx+1}/{len(eval_dataset)}")
    print("\nRésumé généré par le modèle :")
    print(model_summary)
    print("\nRésumé généré par le LLM :")
    print(llm_summary)
    print("\nRésumé de référence :")
    print(reference_summary)
    
    # Calcul des métriques pour la comparaison avec le résumé de référence
    rouge_result_ref = rouge_metric.compute(predictions=[model_summary], references=[reference_summary])
    bleu_result_ref = bleu_metric.compute(predictions=[model_summary], references=[reference_summary])
    
    rouge1_scores_ref.append(rouge_result_ref['rouge1'])
    rouge2_scores_ref.append(rouge_result_ref['rouge2'])
    rougeL_scores_ref.append(rouge_result_ref['rougeL'])
    bleu_scores_ref.append(bleu_result_ref['bleu'])
    
    # Calcul des métriques pour la comparaison avec le résumé généré par le LLM
    rouge_result_llm = rouge_metric.compute(predictions=[model_summary], references=[llm_summary])
    bleu_result_llm = bleu_metric.compute(predictions=[model_summary], references=[llm_summary])
    
    rouge1_scores_llm.append(rouge_result_llm['rouge1'])
    rouge2_scores_llm.append(rouge_result_llm['rouge2'])
    rougeL_scores_llm.append(rouge_result_llm['rougeL'])
    bleu_scores_llm.append(bleu_result_llm['bleu'])

    # Calcul des métriques pour la comparaison avec le résumé généré par le LLM et celui de ref
    rouge_result_llm_ref = rouge_metric.compute(predictions=[llm_summary], references=[reference_summary])
    bleu_result_llm_ref = bleu_metric.compute(predictions=[llm_summary], references=[reference_summary])
    
    rouge1_scores_llm_ref.append(rouge_result_llm_ref['rouge1'])
    rouge2_scores_llm_ref.append(rouge_result_llm_ref['rouge2'])
    rougeL_scores_llm_ref.append(rouge_result_llm_ref['rougeL'])
    bleu_scores_llm_ref.append(bleu_result_llm_ref['bleu'])
    
    print(f"\nScores pour cet exemple :")
    print(f" - (LLM vs Référence) ROUGE-L: {rouge_result_llm_ref['rougeL']:.4f} | BLEU: {bleu_result_llm_ref['bleu']:.4f}")
    print(f" - (Modèle vs Référence) ROUGE-L: {rouge_result_ref['rougeL']:.4f} | BLEU: {bleu_result_ref['bleu']:.4f}")
    print(f" - (Modèle vs LLM) ROUGE-L: {rouge_result_llm['rougeL']:.4f} | BLEU: {bleu_result_llm['bleu']:.4f}")

# Calcul des scores moyens
avg_rouge1_ref = sum(rouge1_scores_ref) / len(rouge1_scores_ref)
avg_rouge2_ref = sum(rouge2_scores_ref) / len(rouge2_scores_ref)
avg_rougeL_ref = sum(rougeL_scores_ref) / len(rougeL_scores_ref)
avg_bleu_ref = sum(bleu_scores_ref) / len(bleu_scores_ref)

avg_rouge1_llm = sum(rouge1_scores_llm) / len(rouge1_scores_llm)
avg_rouge2_llm = sum(rouge2_scores_llm) / len(rouge2_scores_llm)
avg_rougeL_llm = sum(rougeL_scores_llm) / len(rougeL_scores_llm)
avg_bleu_llm = sum(bleu_scores_llm) / len(bleu_scores_llm)

print("\nScores moyens d'évaluation (Modèle vs Référence) :")
print(f"ROUGE-1: {avg_rouge1_ref:.4f}")
print(f"ROUGE-2: {avg_rouge2_ref:.4f}")
print(f"ROUGE-L: {avg_rougeL_ref:.4f}")
print(f"BLEU: {avg_bleu_ref:.4f}")

print("\nScores moyens d'évaluation (Modèle vs LLM) :")
print(f"ROUGE-1: {avg_rouge1_llm:.4f}")
print(f"ROUGE-2: {avg_rouge2_llm:.4f}")
print(f"ROUGE-L: {avg_rougeL_llm:.4f}")
print(f"BLEU: {avg_bleu_llm:.4f}")

print("\nÉvaluation terminée.")
