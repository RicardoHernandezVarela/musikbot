import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mcp.server.fastmcp import FastMCP

from generation.generator import generate_response

# Create an MCP server
# mcp = FastMCP("DemoServer")
mcp = FastMCP("MyServer", host="0.0.0.0", port=8050)

# Tool for song recommendations
@mcp.tool()
def get_songs_recommendations(query_text: str) -> str:
  """Get song recommendations based on a query

  Args:
      query_text: The query text
  """
  
  response = generate_response(query_text, "songs_lyrics")
  
  return response
  
# Tool for lyrics analysis
@mcp.tool()
def get_lyrics_analysis(query_text: str) -> str:
  """Get a lyrics analysis based on a query

  Args:
      query_text: The query text
  """
  
  response = generate_response(query_text, "songs_lyrics", "lyrics_analysis")

  return response
  
# Run the server
if __name__ == "__main__":
    mcp.run(transport="sse")