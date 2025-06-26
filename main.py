import sys

from generation.generator import generate_response

def main():
  escape = False
  while not escape:
    print("🎧 Song Recommendation Bot 🎧")
    print("Write an emotion or idea to get song recommendations:")
    query_text = input("> ")
    
    if query_text == "exit":
      escape = True
    else:
      print("\n🔍 Searching for songs...\n")
      response = generate_response(query_text, "lyrics_test")

      print("🎵 Recommendations generated:")
      print(response)
  
  
if __name__ == "__main__":
  main()