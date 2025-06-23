import polars as pl
from transformers import pipeline

# Load songs data from JSON file
file_path="../data/archive/all_songs_data.json"
songs_data = pl.read_json(file_path).with_row_index(offset=1)

# Select a subset of the data for testing
test_subset = songs_data[0:5]

print(f"You will process a subset of {len(test_subset)} songs...")

# Define the list of emotion labels
emotion_labels = [
    "joy", "sadness", "love", "anger", "loneliness",
    "nostalgia", "hope", "despair", "desire", "regret",
    "empowerment", "grief", "betrayal", "freedom", "pain"
]

# Load model to classify emotions
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to get emotion tags for a given lyric of a song
def classify_emotions_in_lyrics(text: str) -> list[str]:
    # Limit lyrics length to avoid errors with long inputs
    text = text[:700]
    
    try:
        # Classify emotions in the lyrics
        result = classifier(text, candidate_labels=emotion_labels, multi_label=True)
        #print("result: ", result)
    
        # Get the emotion tags with a score >= 0.3, get enough emotions but not restrictive
        tags = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= 0.3:
                tags.append(label)
    
        #print("tags: ", tags)
        return tags
    except Exception as e:
        print("Error: ", e)
        return []

# Apply emotion classification to each row
print("Applying emotion classification...")

emotion_tags = []
for lyrics in test_subset["Lyrics"]:
    if lyrics is None or lyrics.strip() == "":
        emotion_tags.append([])
    else:
        emotion_tags.append(classify_emotions_in_lyrics(lyrics))
        
print("Completed emotion classification.")

# Add emotion tags to the songs data
test_subset_with_emotions = test_subset.with_columns(pl.Series("emotion_tags", emotion_tags))

# Save the dataset with emotion tags in a new JSON file
print("Saving dataset with emotion tags...")
test_subset_with_emotions.write_json("songs_with_emotions_test_2.json")
print("Dataset saved.")
