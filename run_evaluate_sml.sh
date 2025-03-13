#!/bin/bash

# Activer l'environnement virtuel
source .venv/bin/activate

# Définition des paramètres
model_names=("t5-base" "google/mt5-base")
nb_epochs=(3 5 10)
summary_types=("reference_summary" "generated_summary")

touch log_eval_global.txt  # Crée un fichier de log global pour suivre l'exécution

for nb_epoch in "${nb_epochs[@]}"; do
    for model_name in "${model_names[@]}"; do
        for summary_type in "${summary_types[@]}"; do
            # Remplacement des caractères spéciaux dans le nom du modèle
            safe_model_name=$(echo "$model_name" | tr '/:' '_')
            
            # Définir le chemin du modèle sauvegardé
            model_dir="./outputs/models/finetuned_sml_${nb_epoch}_${summary_type}_${safe_model_name}"
            log_file="log_eval_${nb_epoch}_${summary_type}_${safe_model_name}.txt"
            
            # Vérifier si le répertoire du modèle existe avant de lancer l'évaluation
            if [ -d "$model_dir" ]; then
                echo "Lancement de l'évaluation pour $model_name, $nb_epoch epochs, $summary_type..." | tee -a log_eval_global.txt
                nohup python Evaluate/evaluate_sml.py --model_dir "$model_dir" > "$log_file" 2>&1
                echo "Fin de l'évaluation pour $model_name, $nb_epoch epochs, $summary_type." | tee -a log_eval_global.txt
            else
                echo "Modèle introuvable : $model_dir. Passage au suivant..." | tee -a log_eval_global.txt
            fi
        done
    done
done

# Désactiver l'environnement virtuel
deactivate

echo "Toutes les évaluations ont été exécutées."