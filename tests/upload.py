from huggingface_hub import login, upload_file

login('xxx')

# Push your model files
upload_file(
    path_or_fileobj="assets/best_resnet50_model.keras",
    path_in_repo="resnet50_dr_detection.keras",
    repo_id="blackstarai/resnet50-diabetic-retinopathy",
    repo_type="model"
)
