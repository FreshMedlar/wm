import kagglehub

# Download latest version
path = kagglehub.dataset_download("vadimkurochkin/wikitext-103")

print("Path to dataset files:", path)

# /home/medlar/.cache/kagglehub/datasets/vadimkurochkin/wikitext-103/versions/1