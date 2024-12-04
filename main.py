import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

# Load environment variables
load_dotenv()

# Initialize browser controller
controller = Controller()

async def extract_info(search_query: str, task_description: str):
    """Extract information using browser agent"""
    agent = Agent(
        task=f"Go to google.com and find information about {task_description}",
        llm=ChatOpenAI(model="gpt-4", timeout=25),
        controller=controller
    )

    max_steps = 20
    for i in range(max_steps):
        print(f'\n📍 Step {i+1}')
        action, result = await agent.step()

        print('Action:', action)
        print('Result:', result)

        if result.done:
            print('\n✅ Task completed successfully!')
            print('Extracted content:', result.extracted_content)
            
            # Save extracted content
            with open('extracted_content.txt', 'w') as file:
                file.write(result.extracted_content)
            print("Extracted content has been saved to extracted_content.txt")
            return result.extracted_content

    return None

async def main():
    # Set up working directory
    WORKING_DIR = "./knowledge_base"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # Initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete
    )

    # Extract information using browser agent
    content = await extract_info(
        "LLM fine-tuning",
        "Find comprehensive information about LLM fine-tuning techniques and approaches"
    )

    if content:
        # Insert extracted content into RAG
        doc_id = rag.insert(content)
        print(f"Content stored with document ID: {doc_id}")

        # Demonstrate different search modes
        query = "What are the main approaches to LLM fine-tuning?"
        
        print("\nNaive Search Results:")
        print(rag.query(query, param=QueryParam(mode="naive")))
        
        print("\nLocal Search Results:")
        print(rag.query(query, param=QueryParam(mode="local")))
        
        print("\nGlobal Search Results:")
        print(rag.query(query, param=QueryParam(mode="global")))
        
        print("\nHybrid Search Results:")
        print(rag.query(query, param=QueryParam(mode="hybrid")))

if __name__ == "__main__":
    asyncio.run(main())
