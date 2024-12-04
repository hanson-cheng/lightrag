import asyncio
from lightrag.lightrag import LightRAG, QueryParam
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm():
    """Get LLM instance"""
    def llm_func(prompt: str, temperature: float = 0.7) -> str:
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo-instruct",
            temperature=temperature
        )
        return llm.invoke(prompt)
    return llm_func

async def main():
    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./lightrag_data",  # This is where your data will be stored
        llm_model_func=get_llm()
    )
    
    try:
        # Example 1: Add some documents to your knowledge base
        print("\n🔍 Adding documents to knowledge base...")
        rag.insert("Python is a high-level programming language known for its simplicity and readability.")
        rag.insert("TensorFlow and PyTorch are popular deep learning frameworks in Python.")
        rag.insert("JavaScript is primarily used for web development and runs in browsers.")
        
        # Example 2: Search using different modes
        print("\n🔍 Searching with different modes...")
        
        # Naive (keyword) search
        query = "What programming languages are mentioned?"
        print(f"\nQuery: {query}")
        result = rag.query(query, QueryParam(mode="naive", max_results=2))
        print(f"Naive Search Result: {result}")
        
        # Semantic search
        query = "Tell me about machine learning tools"
        print(f"\nQuery: {query}")
        result = rag.query(query, QueryParam(mode="semantic", max_results=2))
        print(f"Semantic Search Result: {result}")
        
        # Example 3: Web search and add to knowledge base
        print("\n🔍 Searching the web...")
        query = "What are the key features of Python 3.13?"
        web_content = await rag.search_web(query)
        if web_content:
            print(f"Added web content about Python 3.13 to knowledge base")
            
            # Now search including the new content
            print("\n🔍 Searching with new web content...")
            result = rag.query(
                "What's new in Python 3.13?", 
                QueryParam(mode="hybrid", max_results=3)
            )
            print(f"Search Result: {result}")
        
        # Example 4: Global context search
        print("\n🔍 Global context search...")
        query = "Summarize everything you know about programming languages"
        result = rag.query(query, QueryParam(mode="global", max_results=5))
        print(f"Global Search Result: {result}")

    finally:
        # Clean up
        rag.close()

if __name__ == "__main__":
    asyncio.run(main())
