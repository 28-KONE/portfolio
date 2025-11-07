from huggingface_hub import HfApi, upload_folder

repo_id = "28-KONE/sentiment-analysis-distilbert"  # ton repo Hugging Face
folder_path = "models/distilbert_finetuned_final"   # chemin relatif à ce script

# Upload du dossier entier
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["*.tmp", "*.log"]  # ignore les fichiers inutiles
)

print("✅ Upload terminé !")
