import os
import torch
import evaluate
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

TOTAL_MAX_ROWS = 100

# Utiliser le modèle original (celui utilisé dans train_sml_5.py) : t5-base
MODEL_NAME = "t5-base"  
DATA_DIR = "./cleaned_files_llm/"

print("\nChargement du modèle original...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modèle chargé sur : {device}")

# Fonction pour charger le corpus à partir des fichiers CSV
def load_all_corpus_csv(data_dir, total_max_rows=TOTAL_MAX_ROWS):
    texts = []
    llm_summaries = []
    reference_summaries = []
    filenames = sorted(os.listdir(data_dir))
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        texts.extend(df["text"].tolist())
        llm_summaries.extend(df["generated_summary"].tolist())
        reference_summaries.extend(df["reference_summary"].tolist())
    # Limiter le nombre total de lignes si nécessaire
    if total_max_rows is not None:
        texts = texts[:total_max_rows]
        llm_summaries = llm_summaries[:total_max_rows]
        reference_summaries = reference_summaries[:total_max_rows]
    return Dataset.from_dict({
        "text": texts,
        "llm_summary": llm_summaries,
        "reference_summary": reference_summaries
    })

print("\nChargement des données...")
eval_dataset = load_all_corpus_csv(DATA_DIR, total_max_rows=TOTAL_MAX_ROWS)
print(f"Nombre d'exemples chargés : {len(eval_dataset)}")

# Fonction de génération de résumé avec le modèle
def generate_summary(text, max_total_tokens=2048):
    prefix = (
        "Vous êtes un expert en synthèse de texte. Veuillez fournir un résumé détaillé et complet du texte suivant. "
        "Assurez-vous d'inclure tous les points clés, les détails importants et l'essence générale du contenu. "
        "Texte : "
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    allowed_text_tokens = max_total_tokens - len(prefix_ids) - 512
    text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=allowed_text_tokens)
    combined_ids = prefix_ids + text_ids
    prompt = tokenizer.decode(combined_ids)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=512,  # Ajustez si nécessaire
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print("\nÉvaluation du modèle...")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# Listes pour stocker les scores
rouge1_scores_ref = []
rouge2_scores_ref = []
rougeL_scores_ref = []
bleu_scores_ref = []

rouge1_scores_llm = []
rouge2_scores_llm = []
rougeL_scores_llm = []
bleu_scores_llm = []

rouge1_scores_llm_ref = []
rouge2_scores_llm_ref = []
rougeL_scores_llm_ref = []
bleu_scores_llm_ref = []

# Nom du fichier log pour enregistrer les résultats
log_filename = "benchmark_results.txt"
with open(log_filename, "w", encoding="utf-8") as log_file:
    log_file.write("Benchmark Evaluation for original model (t5-base)\n")
    log_file.write("=" * 50 + "\n")
    
    for idx, example in enumerate(eval_dataset):
        try:
            input_text = example["text"]
            llm_summary = example["llm_summary"].strip()
            reference_summary = example["reference_summary"].strip()
            
            # Génération du résumé avec le modèle original
            model_summary = generate_summary(input_text).strip()
            
            separator = "\n" + "=" * 50 + "\n"
            log_file.write(separator)
            log_file.write(f"Exemple #{idx+1}/{len(eval_dataset)}\n")
            
            log_file.write("\nRésumé généré par le modèle :\n")
            log_file.write(model_summary + "\n")
            log_file.write("\nRésumé généré par le LLM :\n")
            log_file.write(llm_summary + "\n")
            log_file.write("\nRésumé de référence :\n")
            log_file.write(reference_summary + "\n")
            
            # Affichage console (pour suivi)
            print(separator)
            print(f"Exemple #{idx+1}/{len(eval_dataset)}")
            print("\nRésumé généré par le modèle :")
            print(model_summary)
            print("\nRésumé généré par le LLM :")
            print(llm_summary)
            print("\nRésumé de référence :")
            print(reference_summary)
            
            # Calcul des métriques (Modèle vs Référence)
            rouge_result_ref = rouge_metric.compute(predictions=[model_summary], references=[reference_summary])
            bleu_result_ref = bleu_metric.compute(predictions=[model_summary], references=[reference_summary])
            
            rouge1_scores_ref.append(rouge_result_ref['rouge1'])
            rouge2_scores_ref.append(rouge_result_ref['rouge2'])
            rougeL_scores_ref.append(rouge_result_ref['rougeL'])
            bleu_scores_ref.append(bleu_result_ref['bleu'])
            
            # Calcul des métriques (Modèle vs LLM)
            rouge_result_llm = rouge_metric.compute(predictions=[model_summary], references=[llm_summary])
            bleu_result_llm = bleu_metric.compute(predictions=[model_summary], references=[llm_summary])
            
            rouge1_scores_llm.append(rouge_result_llm['rouge1'])
            rouge2_scores_llm.append(rouge_result_llm['rouge2'])
            rougeL_scores_llm.append(rouge_result_llm['rougeL'])
            bleu_scores_llm.append(bleu_result_llm['bleu'])
            
            # Calcul des métriques (LLM vs Référence)
            rouge_result_llm_ref = rouge_metric.compute(predictions=[llm_summary], references=[reference_summary])
            bleu_result_llm_ref = bleu_metric.compute(predictions=[llm_summary], references=[reference_summary])
            
            rouge1_scores_llm_ref.append(rouge_result_llm_ref['rouge1'])
            rouge2_scores_llm_ref.append(rouge_result_llm_ref['rouge2'])
            rougeL_scores_llm_ref.append(rouge_result_llm_ref['rougeL'])
            bleu_scores_llm_ref.append(bleu_result_llm_ref['bleu'])
            
            msg = (
                f"\nScores pour cet exemple :\n"
                f" - (LLM vs Référence) ROUGE-L: {rouge_result_llm_ref['rougeL']:.4f} | BLEU: {bleu_result_llm_ref['bleu']:.4f}\n"
                f" - (Modèle vs Référence) ROUGE-L: {rouge_result_ref['rougeL']:.4f} | BLEU: {bleu_result_ref['bleu']:.4f}\n"
                f" - (Modèle vs LLM) ROUGE-L: {rouge_result_llm['rougeL']:.4f} | BLEU: {bleu_result_llm['bleu']:.4f}\n"
            )
            log_file.write(msg + "\n")
            print(msg)
        except Exception as e:
            error_msg = f"Erreur sur l'exemple #{idx+1}: {e}"
            log_file.write(error_msg + "\n")
            print(error_msg)
    
    # Calcul des scores moyens (en vérifiant qu'il y a bien des scores pour éviter une division par zéro)
    avg_rouge1_ref = sum(rouge1_scores_ref) / len(rouge1_scores_ref) if rouge1_scores_ref else 0
    avg_rouge2_ref = sum(rouge2_scores_ref) / len(rouge2_scores_ref) if rouge2_scores_ref else 0
    avg_rougeL_ref = sum(rougeL_scores_ref) / len(rougeL_scores_ref) if rougeL_scores_ref else 0
    avg_bleu_ref = sum(bleu_scores_ref) / len(bleu_scores_ref) if bleu_scores_ref else 0
    
    avg_rouge1_llm = sum(rouge1_scores_llm) / len(rouge1_scores_llm) if rouge1_scores_llm else 0
    avg_rouge2_llm = sum(rouge2_scores_llm) / len(rouge2_scores_llm) if rouge2_scores_llm else 0
    avg_rougeL_llm = sum(rougeL_scores_llm) / len(rougeL_scores_llm) if rougeL_scores_llm else 0
    avg_bleu_llm = sum(bleu_scores_llm) / len(bleu_scores_llm) if bleu_scores_llm else 0
    
    avg_msg_ref = (
        "\nScores moyens d'évaluation (Modèle vs Référence) :\n"
        f"ROUGE-1: {avg_rouge1_ref:.4f}\n"
        f"ROUGE-2: {avg_rouge2_ref:.4f}\n"
        f"ROUGE-L: {avg_rougeL_ref:.4f}\n"
        f"BLEU: {avg_bleu_ref:.4f}\n"
    )
    avg_msg_llm = (
        "\nScores moyens d'évaluation (Modèle vs LLM) :\n"
        f"ROUGE-1: {avg_rouge1_llm:.4f}\n"
        f"ROUGE-2: {avg_rouge2_llm:.4f}\n"
        f"ROUGE-L: {avg_rougeL_llm:.4f}\n"
        f"BLEU: {avg_bleu_llm:.4f}\n"
    )
    log_file.write(avg_msg_ref + "\n")
    log_file.write(avg_msg_llm + "\n")
    log_file.write("Évaluation terminée.\n")
    
    print(avg_msg_ref)
    print(avg_msg_llm)
    print("Évaluation terminée.")
