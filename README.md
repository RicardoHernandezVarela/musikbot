# Music Recommendation Bot

A music recommendation system that suggests songs based on emotions, themes, or lyrics using RAG (Retrieval Augmented Generation) and vector search.

## Features

- Get song recommendations based on emotions or themes
- Analyze lyrics and emotional themes in songs
- CLI interface for direct interaction
- API server with MCP protocol support

## Architecture

- **Vector Database**: ChromaDB for storing song lyrics and metadata
- **Embeddings**: Sentence transformers for semantic search
- **LLM Integration**: OpenAI for generating recommendations
- **API**: FastMCP server for tool integration

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```
4. Prepare the database (if not already created):
   ```
   python db/chroma_client.py
   ```

## Usage

### CLI Interface

```
python main.py
```

### MCP Server

Start the server:
```
python mcp/server.py
```

Connect with the client:
```
python mcp/client.py
```

## Project Structure

- `main.py`: CLI interface
- `data/`: Song dataset and preprocessing
- `embeddings/`: Vector embedding generation
- `db/`: ChromaDB integration
- `retrieval/`: Vector search functionality
- `generation/`: LLM prompt and response generation
- `mcp/`: MCP server and client implementation