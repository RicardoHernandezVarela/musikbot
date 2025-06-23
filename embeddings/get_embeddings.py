import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch

# Load songs with emotions tags from JSON file
file_path="../data/songs_with_emotions_test_1.json"
songs_with_emotions_tags_data = pl.read_json(file_path)

# Load tokenizer for the embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Tokens test
#lyrics = songs_with_emotions_tags_data[4]["Lyrics"].item()
#tokens = tokenizer.tokenize(lyrics)
#print("tokens: ", tokens)
#print("Número de tokens:", len(tokens))

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
  
# Test embedding function
# print("Testing embedding function...")

# lyrics = songs_with_emotions_tags_data[4]["Lyrics"].item()
# embeddings = embed_texts([lyrics])
# print("embeddings: ", embeddings)
# print("embeddings shape: ", embeddings.shape)

# print("Completed embedding function test.")

# Get array with all the lyrics
print("Getting all the lyrics...")

all_lyrics = []
for lyrics in songs_with_emotions_tags_data["Lyrics"]:
    if lyrics is None or lyrics.strip() == "":
        pass
    else:
        all_lyrics.append(lyrics)
        
print("all_lyrics: ", len(all_lyrics))
print("Completed getting all the lyrics.")

# Get embeddings for all the lyrics
print("Getting embeddings for all the lyrics...")

all_embeddings = embed_texts(all_lyrics)

print("all_embeddings: ", all_embeddings.shape)
print("Completed getting embeddings for all the lyrics.")

# Add embeddings to the songs data
print("Adding embeddings to the songs data...")

songs_with_lyrics_embeddings = songs_with_emotions_tags_data.with_columns(pl.Series("embeddings", all_embeddings.tolist()))

print("Completed adding embeddings to the songs data.")

# Save the dataset with embeddings in a new JSON file
print("Saving dataset with embeddings...")

songs_with_lyrics_embeddings.write_json("songs_with_lyrics_embeddings_1.json")

print("Dataset saved.")
