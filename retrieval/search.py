import chromadb
import os
import sys
from langchain_community.llms.ctransformers import CTransformers

# Get path to db
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

db_path = os.path.join(parent_dir, "db", "chroma_db")

print("db_path: ", db_path)

# Chromadb client
chroma_client = chromadb.PersistentClient(path=db_path)

# Get collection
lyrics_test = chroma_client.get_collection(name="lyrics_test")

# Search by lyrics
from utils.embedder import embed_texts

query_text = "give 5 recommendations of songs when feeling nostalgic"
query_embedding = embed_texts([query_text])[0].tolist() 
#print("query_embedding: ", query_embedding)

results = lyrics_test.query(
    query_embeddings=[query_embedding],
    n_results=2,
)

# print("results: ", results)

# Print results
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

print("context: ", context)

# Generate prompt
prompt = f"""
Context: {context}

Question: {query_text}

Answer:"""


print("prompt: ", prompt)

# test generation
llm = CTransformers(model="TheBloke/Llama-2-7b-GGML", model_type="llama")

generated_response = llm(prompt)

print("generated_response: ", generated_response)

