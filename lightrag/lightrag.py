import os
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Union
import json
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import asyncio
from .browser import BrowserManager, BrowseResult
import logging

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class QueryParam:
    mode: str = "hybrid"  # Can be "naive", "local", "global", "semantic", or "hybrid"
    max_results: int = 5
    temperature: float = 0.7

class LightRAG:
    def __init__(self, working_dir: str, llm_model_func: Callable):
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.browser = BrowserManager()
        
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)
        
        # Create tables if they don't exist
        self._init_database()

    def _init_database(self):
        """Initialize the database tables"""
        # Create documents table
        self.supabase.table("documents").select("*").limit(1).execute()
        
    def insert(self, content: str, doc_id: Optional[str] = None) -> str:
        """Insert a document into the RAG system"""
        if doc_id is None:
            # Get the count of existing documents
            result = self.supabase.table("documents").select("id").execute()
            doc_id = f"doc_{len(result.data)}"
        
        # Insert document
        self.supabase.table("documents").insert({
            "id": doc_id,
            "content": content,
            "created_at": "now()"
        }).execute()
        
        return doc_id

    def query(self, query: str, param: Optional[QueryParam] = None) -> str:
        """Query the RAG system using specified mode"""
        if param is None:
            param = QueryParam()

        # Get all documents
        result = self.supabase.table("documents").select("*").execute()
        if not result.data:
            return "No documents available to search."

        # Different search strategies based on mode
        if param.mode == "naive":
            return self._naive_search(query, result.data, param)
        elif param.mode == "semantic":
            return self._semantic_search(query, result.data, param)
        elif param.mode == "local":
            return self._local_search(query, result.data, param)
        elif param.mode == "global":
            return self._global_search(query, result.data, param)
        else:  # hybrid
            return self._hybrid_search(query, result.data, param)

    def _naive_search(self, query: str, documents: List[Dict], param: QueryParam) -> str:
        """Simple keyword-based search"""
        results = []
        for doc in documents:
            if query.lower() in doc["content"].lower():
                results.append(doc["content"])
                if len(results) >= param.max_results:
                    break
        
        if not results:
            return "No relevant documents found."
        
        context = "\n".join(results)
        return self.llm_model_func(
            f"Based on the following context, answer the query: {query}\n\nContext: {context}",
            temperature=param.temperature
        )

    def _local_search(self, query: str, documents: List[Dict], param: QueryParam) -> str:
        """Search focusing on immediate context"""
        results = self._naive_search(query, documents, param)
        return self.llm_model_func(
            f"Analyze the local context and answer: {query}\n\nContext: {results}",
            temperature=param.temperature
        )

    def _global_search(self, query: str, documents: List[Dict], param: QueryParam) -> str:
        """Search considering broader context and relationships"""
        all_content = "\n".join(doc["content"] for doc in documents)
        return self.llm_model_func(
            f"Consider the entire context and answer: {query}\n\nGlobal Context: {all_content}",
            temperature=param.temperature
        )

    def _hybrid_search(self, query: str, documents: List[Dict], param: QueryParam) -> str:
        """Combine local and global search strategies"""
        local_results = self._local_search(query, documents, param)
        global_results = self._global_search(query, documents, param)
        
        return self.llm_model_func(
            f"Combining local and global insights, answer: {query}\n\n"
            f"Local Context: {local_results}\n\nGlobal Context: {global_results}",
            temperature=param.temperature
        )

    def _semantic_search(self, query: str, documents: List[Dict], param: QueryParam) -> str:
        """Semantic search using embeddings"""
        # Use LLM to analyze semantic relevance
        context = "\n".join(doc["content"] for doc in documents)
        prompt = f"""Given the following documents, find the most semantically relevant information for the query: {query}
        
Documents:
{context}

Please provide a detailed answer based on the most relevant information found."""
        
        return self.llm_model_func(prompt, temperature=param.temperature)

    async def search_web(self, query: str) -> Union[str, None]:
        """Search web for information and add to RAG system"""
        try:
            logger.info(f"Starting web search in LightRAG for query: {query}")
            result = await self.browser.search_and_extract(query)
            
            if result.done and not result.error and result.content:
                logger.info("Successfully extracted web content, adding to database")
                # Add extracted content to our database
                self.insert(result.content)
                return result.content
                
            if result.error:
                logger.error(f"Web search failed: {result.error}")
            else:
                logger.warning("Web search returned no content")
            return None
            
        except Exception as e:
            logger.error(f"Error in search_web: {e}")
            return None

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'browser'):
            self.browser.close()
