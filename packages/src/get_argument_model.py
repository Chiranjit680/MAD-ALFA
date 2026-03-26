import kagglehub

# Download latest version
path = kagglehub.model_download("chiranjit680/argument-quality-model/pyTorch/default")

print("Path to model files:", path)