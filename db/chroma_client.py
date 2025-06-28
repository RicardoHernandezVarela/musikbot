import polars as pl
import os
import chromadb
import json
import sys

# Load songs with embeddings from JSON file
file_path="../embeddings/songs_with_lyrics_embeddings_sample.json"
songs_with_lyrics_embeddings = pl.read_json(file_path)

print(f"Number of songs: {len(songs_with_lyrics_embeddings)}")

# Create a ChromaDB client - persistent
# chroma_client = chromadb.Client()
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

# Create collection
songs_lyrics = chroma_client.get_or_create_collection(name="songs_lyrics")

# Add documents to the collection
documents_list = songs_with_lyrics_embeddings["Lyrics"].to_list()
metadatas_raw = songs_with_lyrics_embeddings.select(["Song Title", "Artist", "Album", "Year", "Release Date", "emotion_tags"]).to_dicts()

metadatas_list = []
for metadata in metadatas_raw:
    # Change emotion_tags to a comma separated string
    if "emotion_tags" in metadata and isinstance(metadata["emotion_tags"], list):
        metadata["emotion_tags"] = ", ".join(metadata["emotion_tags"])
    
    # Replace None values with empty strings
    for key, value in list(metadata.items()):
        if value is None:
            metadata[key] = ""
    
    metadatas_list.append(metadata)   
     
#print("metadatas_list: ", metadatas_list[0])

embeddings_list = songs_with_lyrics_embeddings["embeddings"].to_list()

# ids should be strings
ids = songs_with_lyrics_embeddings["index"].to_list()
ids_list = [str(x) for x in ids]

# Add documents to the songs_lyrics collection
songs_lyrics.add(
    documents=documents_list,
    metadatas=metadatas_list,
    embeddings=embeddings_list,
    ids=ids_list
)
