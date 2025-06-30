import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from retrieval.search import get_context

from dotenv import load_dotenv
load_dotenv() 

from langchain_openai import OpenAI

prompt_schemas = {
  "song_recommendations": {
    "intro": "You are a music recommendation expert with deep knowledge of songs and their emotional themes.",
    "description": "list of 2-3 recommended songs different from the ones in the context, with brief explanation why each matches the query and the key lyrics of the song."
  },
  "lyrics_analysis": {
    "intro": "You are a music recommendation expert with deep knowledge of songs and their emotional themes.",
    "description": "provide a brief analysis of the emotional themes in the query and the key lyrics of the song."
  }
}

def generate_prompt(query_text: str, collection_name: str, search_type: str):
  # Get context
  context = get_context(query_text, collection_name)
  
  # Get prompt schema
  prompt_schema = prompt_schemas[search_type]
  intro = prompt_schema["intro"]
  description = prompt_schema["description"]
  
  # Generate prompt
  prompt = f"""
  {intro}
  
  Based on the following context of songs and their emotional themes, {description}
  
  Context: {context}
  
  User Query: {query_text}

  Answer:"""


  #print("prompt: ", prompt)
  
  return prompt

llm = OpenAI(
  model_name="gpt-3.5-turbo-instruct",
  temperature=0.7,
  max_tokens=350
)

def generate_response(query_text: str, collection_name: str, search_type: str = "song_recommendations"):
  prompt = generate_prompt(query_text, collection_name, search_type)

  generated_response = llm.invoke(prompt)

  return generated_response
  
# Test
# query_text = "What are the most common themes in the songs?"
# collection_name = "lyrics_test"
# generated_response = generate_response(query_text, collection_name)

# print("generated_response: ", generated_response)