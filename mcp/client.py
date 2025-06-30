import asyncio
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    # Connect to the server using SSE
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
                
            # Call our get_songs_recommendations tool
            print("\n🔍 Searching for songs...\n")
            result_songs = await session.call_tool("get_songs_recommendations", arguments={"query_text": "songs with narratives about darkness"})
            
            print("🎵 Recommendations generated:")
            #print(result_songs)
            print(f"Result: {result_songs.content[0].text}")
            
            # Call our get_lyrics_analysis tool
            print("\n🔍 Searching for song lyrics...\n")
            result_lyrics = await session.call_tool("get_lyrics_analysis", arguments={"query_text": "what the song like the moon by future islands?"})
            
            print("🎵 Lyrics analysis generated:")
            #print(result_lyrics)
            print(f"Result: {result_lyrics.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())