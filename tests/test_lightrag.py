import sys
import os
import pytest
from dotenv import load_dotenv
from langchain_openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.lightrag import LightRAG, QueryParam

# Load environment variables
load_dotenv()

def get_llm():
    """Get LLM instance for testing"""
    def llm_func(prompt: str, temperature: float = 0.7) -> str:
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo-instruct",
            temperature=temperature
        )
        return llm.invoke(prompt)
    return llm_func

@pytest.fixture
def rag_instance(tmp_path):
    """Create a LightRAG instance for testing"""
    working_dir = tmp_path / "lightrag_test"
    working_dir.mkdir()
    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=get_llm()
    )

def test_basic_document_insertion(rag_instance):
    """Test basic document insertion and retrieval"""
    content = "This is a test document about artificial intelligence and machine learning."
    
    # Insert document
    doc_id = rag_instance.insert(content)
    assert doc_id is not None
    
    # Retrieve and verify document
    result = rag_instance.supabase.table("documents").select("*").eq("id", doc_id).execute()
    assert len(result.data) == 1
    assert result.data[0]["content"] == content

def test_vector_search(rag_instance):
    """Test vector similarity search"""
    # Insert multiple documents
    docs = [
        "Artificial Intelligence is revolutionizing technology",
        "The weather today is sunny with clear skies",
        "Italian cuisine is known for pasta and pizza"
    ]
    
    for doc in docs:
        rag_instance.insert(doc)
    
    # Test search
    query = "What's new in AI technology?"
    param = QueryParam(mode="semantic", max_results=2)
    results = rag_instance.query(query, param)
    assert results is not None
    assert isinstance(results, str)
    assert len(results) > 0

def test_hybrid_search(rag_instance):
    """Test hybrid search combining keyword and semantic search"""
    docs = [
        "Python is a popular programming language",
        "Java is used for enterprise applications"
    ]
    
    for doc in docs:
        rag_instance.insert(doc)
    
    # Test hybrid search
    query = "programming with Python"
    param = QueryParam(mode="hybrid", max_results=2)
    results = rag_instance.query(query, param)
    assert results is not None
    assert isinstance(results, str)
    assert len(results) > 0

def test_naive_search(rag_instance):
    """Test naive keyword search"""
    content = "This is a unique test document"
    doc_id = rag_instance.insert(content)
    
    # Test naive search
    query = "unique test"
    param = QueryParam(mode="naive", max_results=1)
    results = rag_instance.query(query, param)
    assert results is not None
    assert isinstance(results, str)
    assert len(results) > 0

def test_global_search(rag_instance):
    """Test global context search"""
    docs = [
        "First part of a story about AI",
        "Second part continuing the AI narrative",
        "Third part concluding the AI discussion"
    ]
    
    for doc in docs:
        rag_instance.insert(doc)
    
    # Test global search
    query = "Summarize the AI story"
    param = QueryParam(mode="global", max_results=3)
    results = rag_instance.query(query, param)
    assert results is not None
    assert isinstance(results, str)
    assert len(results) > 0

import pytest
import asyncio

@pytest.mark.asyncio
async def test_web_search(rag_instance, monkeypatch):
    """Test web search and content extraction"""
    # Mock the browser search to return a known result
    async def mock_search_and_extract(*args, **kwargs):
        from lightrag.browser import BrowseResult
        return BrowseResult(
            content="LoRA (Low-Rank Adaptation) is a technique in machine learning that enables efficient fine-tuning of large language models.",
            url="https://example.com",
            title="LoRA Explanation",
            done=True
        )
    
    # Apply the mock
    monkeypatch.setattr(rag_instance.browser, "search_and_extract", mock_search_and_extract)
    
    query = "What is LoRA in machine learning?"
    content = await rag_instance.search_web(query)
    
    assert content is not None
    assert isinstance(content, str)
    assert len(content) > 0
    assert "LoRA" in content
    
    # Test that content was added to database
    result = rag_instance.supabase.table("documents").select("*").execute()
    assert len(result.data) > 0
    assert any("LoRA" in doc["content"] for doc in result.data)

def test_cleanup(rag_instance):
    """Test proper cleanup of resources"""
    rag_instance.close()
