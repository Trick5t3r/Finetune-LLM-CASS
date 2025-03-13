import os
import argparse
import torch
import evaluate
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

TOTAL_MAX_ROWS_DEFAULT = 100

# Argument parser pour récupérer le chemin du modèle depuis la ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, help="Chemin du modèle fine-tuné")
parser.add_argument("--total_max_rows", type=str, default= TOTAL_MAX_ROWS_DEFAULT, help="Nombre de tests")
parser.add_argument("--data_dir", type=str, default="./data/cleaned_files_llm/", help="Dossier contenant les fichiers CSV")
args = parser.parse_args()

MODEL_DIR = args.model_dir
DATA_DIR = args.data_dir
TOTAL_MAX_ROWS = args.total_max_rows

# Créer un fichier de log basé sur le nom du modèle
model_name = os.path.basename(os.path.normpath(MODEL_DIR))
log_filename = f"./outputs/results/{model_name}_benchmarks.txt"
log_file = open(log_filename, "w", encoding="utf-8")


def log(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# Charger le modèle et le tokenizer
log("\nChargement du modèle...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
log(f"Modèle chargé sur : {device}")

# Charger le corpus à partir des fichiers CSV
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

log("\nChargement des données...")
eval_dataset = load_all_corpus_csv(DATA_DIR, total_max_rows=TOTAL_MAX_ROWS)
log(f"Nombre d'exemples chargés : {len(eval_dataset)}")

# Fonction de génération de résumé avec le modèle
def generate_summary(text, max_total_tokens=2048):
    # Définir le préfixe d'instructions (à ne pas tronquer)
    prefix = (
        "Vous êtes un expert en synthèse de texte. Veuillez fournir un résumé détaillé et complet du texte suivant. "
        "Assurez-vous d'inclure tous les points clés, les détails importants et l'essence générale du contenu. "
        "Texte : "
    )

    # Encoder le préfixe pour obtenir sa longueur (sans tokens spéciaux)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    # Calculer le nombre de tokens autorisés pour la partie texte
    allowed_text_tokens = max_total_tokens - len(prefix_ids) - 512

    # Encoder le texte à résumer en appliquant la troncature uniquement sur cette partie
    text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=allowed_text_tokens)

    # Combiner le préfixe et le texte tronqué
    combined_ids = prefix_ids + text_ids
    prompt = tokenizer.decode(combined_ids)

    # Tokenisation du prompt complet sans spécifier max_length ici, car il a déjà été tronqué
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Générer le résumé avec beam search et en fixant un nombre maximum de nouveaux tokens
    summary_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=512,  # Ajustez en fonction de la longueur souhaitée pour le résumé
        num_beams=5,
        early_stopping=True
    )

    # Décoder et retourner le résumé généré
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Charger les métriques ROUGE et BLEU
log("\nÉvaluation du modèle...")
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
    try:
        input_text = example["text"]
        llm_summary = example["llm_summary"].strip()
        reference_summary = example["reference_summary"].strip()
        
        # Générer le résumé avec votre modèle
        model_summary = generate_summary(input_text).strip()
        
        log("\n" + "="*50)
        log(f"Exemple #{idx+1}/{len(eval_dataset)}")
        log("\nRésumé généré par le modèle :")
        log(model_summary)
        log("\nRésumé généré par le LLM :")
        log(llm_summary)
        log("\nRésumé de référence :")
        log(reference_summary)
        
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

        # Calcul des métriques pour la comparaison entre le résumé généré par le LLM et celui de référence
        rouge_result_llm_ref = rouge_metric.compute(predictions=[llm_summary], references=[reference_summary])
        bleu_result_llm_ref = bleu_metric.compute(predictions=[llm_summary], references=[reference_summary])
        
        rouge1_scores_llm_ref.append(rouge_result_llm_ref['rouge1'])
        rouge2_scores_llm_ref.append(rouge_result_llm_ref['rouge2'])
        rougeL_scores_llm_ref.append(rouge_result_llm_ref['rougeL'])
        bleu_scores_llm_ref.append(bleu_result_llm_ref['bleu'])
        
        log(f"\nScores pour cet exemple :")
        log(f" - (LLM vs Référence) ROUGE-L: {rouge_result_llm_ref['rougeL']:.4f} | BLEU: {bleu_result_llm_ref['bleu']:.4f}")
        log(f" - (Modèle vs Référence) ROUGE-L: {rouge_result_ref['rougeL']:.4f} | BLEU: {bleu_result_ref['bleu']:.4f}")
        log(f" - (Modèle vs LLM) ROUGE-L: {rouge_result_llm['rougeL']:.4f} | BLEU: {bleu_result_llm['bleu']:.4f}")
    except:
        pass

# Calcul des scores moyens
avg_rouge1_ref = sum(rouge1_scores_ref) / len(rouge1_scores_ref)
avg_rouge2_ref = sum(rouge2_scores_ref) / len(rouge2_scores_ref)
avg_rougeL_ref = sum(rougeL_scores_ref) / len(rougeL_scores_ref)
avg_bleu_ref = sum(bleu_scores_ref) / len(bleu_scores_ref)

avg_rouge1_llm = sum(rouge1_scores_llm) / len(rouge1_scores_llm)
avg_rouge2_llm = sum(rouge2_scores_llm) / len(rouge2_scores_llm)
avg_rougeL_llm = sum(rougeL_scores_llm) / len(rougeL_scores_llm)
avg_bleu_llm = sum(bleu_scores_llm) / len(bleu_scores_llm)

log("\nScores moyens d'évaluation (Modèle vs Référence) :")
log(f"ROUGE-1: {avg_rouge1_ref:.4f}")
log(f"ROUGE-2: {avg_rouge2_ref:.4f}")
log(f"ROUGE-L: {avg_rougeL_ref:.4f}")
log(f"BLEU: {avg_bleu_ref:.4f}")

log("\nScores moyens d'évaluation (Modèle vs LLM) :")
log(f"ROUGE-1: {avg_rouge1_llm:.4f}")
log(f"ROUGE-2: {avg_rouge2_llm:.4f}")
log(f"ROUGE-L: {avg_rougeL_llm:.4f}")
log(f"BLEU: {avg_bleu_llm:.4f}")

log("\nÉvaluation terminée.")

# Fermer le fichier de log
log_file.close()
