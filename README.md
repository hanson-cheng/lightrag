# LightRAG

A lightweight RAG (Retrieval-Augmented Generation) system that combines document search, semantic understanding, and web browsing capabilities.

## Features

- Multiple search modes:
  - Naive (keyword-based) search
  - Semantic search using embeddings
  - Local context search
  - Global context search
  - Hybrid search combining multiple approaches
- Web browsing integration for dynamic content retrieval
- Vector database integration using Supabase
- Flexible LLM integration

## Requirements

- Python 3.13+
- OpenAI API key
- Supabase account and credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/lightrag.git
cd lightrag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
DATABASE_URL=your_database_url
```

## Usage

See `example.py` for a complete usage example. Here's a quick start:

```python
from lightrag.lightrag import LightRAG, QueryParam
from langchain_openai import OpenAI

# Initialize
rag = LightRAG(working_dir="./lightrag_data", llm_model_func=your_llm_func)

# Add documents
rag.insert("Your document content here")

# Search
results = rag.query(
    "Your query here",
    QueryParam(mode="hybrid", max_results=3)
)

# Web search
web_content = await rag.search_web("Your web query here")

# Clean up
rag.close()
```

## Testing

Run the test suite:
```bash
pytest -v tests/
```

## License

MIT License
