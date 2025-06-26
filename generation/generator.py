import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from retrieval.search import get_context

from dotenv import load_dotenv
load_dotenv() 

from langchain_openai import OpenAI

def generate_prompt(query_text: str, collection_name: str):
  # Get context
  context = get_context(query_text, collection_name)
  
  # Generate prompt
  prompt = f"""
  Context: {context}

  Question: {query_text}

  Answer:"""


  print("prompt: ", prompt)
  
  return prompt

llm = OpenAI(
  model_name="gpt-3.5-turbo-instruct",
  temperature=0.7,
  max_tokens=256
)

def generate_response(query_text: str, collection_name: str):
  prompt = generate_prompt(query_text, collection_name)

  generated_response = llm.invoke(prompt)

  return generated_response
  
# Test
query_text = "What are the most common themes in the songs?"
collection_name = "lyrics_test"
generated_response = generate_response(query_text, collection_name)

print("generated_response: ", generated_response)