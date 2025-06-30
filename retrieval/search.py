import chromadb
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.embedder import embed_texts

# Get path to db
db_path = os.path.join(parent_dir, "db", "chroma_db")

# Chromadb client
chroma_client = chromadb.PersistentClient(path=db_path)

def get_context(query_text: str, collection_name: str):
  # Get collection
  lyrics_test = chroma_client.get_collection(name=collection_name)

  query_embedding = embed_texts([query_text])[0].tolist()

  results = lyrics_test.query(
    query_embeddings=[query_embedding],
    n_results=3,
  )

  # Print results
  print("Results from ChromaDB:")
  for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"{i+1}. {metadata['Song Title']} - {metadata['Artist']}")
      
  # Build context
  docs = results['documents'][0]
  metadatas = results['metadatas'][0]

  context = ""
  for doc, meta in zip(docs, metadatas):
    emotions = meta.get("emotion_tags", "")
    context += f"""
    Song: {meta.get("Song Title", "Unknown")} by {meta.get("Artist", "Unknown")}
    Emotions: {emotions}
    Lyrics (excerpt): {doc[:300]}...
    """

  #print("context: ", context)
  
  return context

# Test
# query_text = "songs with narratives about darkness"
# context = get_context(query_text, "songs_lyrics")

# print("context: ", context)



