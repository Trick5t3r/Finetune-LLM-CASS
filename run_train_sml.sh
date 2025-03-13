#!/bin/bash

# Activer l'environnement virtuel
source .venv/bin/activate

# Définition des paramètres
model_names=("t5-base" "google/mt5-base")
nb_epochs=(3 5 10)
summary_types=("reference_summary" "generated_summary")

# Boucle sur toutes les combinaisons possibles
touch log_train_global.txt  # Crée un fichier de log global pour suivre l'exécution

for nb_epoch in "${nb_epochs[@]}"; do
    for model_name in "${model_names[@]}"; do
        for summary_type in "${summary_types[@]}"; do
            # Remplacement des caractères spéciaux dans le nom du modèle
            safe_model_name=$(echo "$model_name" | tr '/:' '_')
            
            # Définir le chemin de sauvegarde en fonction des paramètres
            save_path="./outputs/models/finetuned_sml_${nb_epoch}_${summary_type}_${safe_model_name}"
            log_file="log_train_${nb_epoch}_${summary_type}_${safe_model_name}.txt"
            
            # Exécuter la commande en séquentiel
            echo "Lancement de l'entraînement avec $model_name, $nb_epoch epochs, $summary_type..." | tee -a log_train_global.txt
            nohup python Train/train_sml_last_version.py \
                --model_name "$model_name" \
                --nb_epoch "$nb_epoch" \
                --summary_type "$summary_type" \
                --save_path "$save_path" > "$log_file" 2>&1

            # Attendre la fin du processus avant de passer à la prochaine exécution
            echo "Fin de l'entraînement pour $model_name, $nb_epoch epochs, $summary_type." | tee -a log_train_global.txt
        done
    done
done

# Désactiver l'environnement virtuel
deactivate

echo "Tous les entraînements ont été exécutés séquentiellement."
