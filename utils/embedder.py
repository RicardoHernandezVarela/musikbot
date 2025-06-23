from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer for the embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Embedding function
def embed_texts(texts):
  inputs = tokenizer(
      texts,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
  )

  with torch.no_grad():
    output = model(**inputs)
    embeddings = output.last_hidden_state.mean(dim = 1).cpu().numpy()

    return embeddings
  