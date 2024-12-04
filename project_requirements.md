Let’s Start Coding
Before we dive into our application, we will create an ideal environment for the code to work. For this, we need to install the necessary Python libraries required. Firstly, we will start by installing the libraries that support the model. For this, we will do a pip install requirements. Since the demo uses the OpenAI large model, you must first set the OpenAI API Key.

pip install -r requirements.txt
Once installed we import browser_use, langchain_openai, and lightrag.

from browser_use import Agent, Controller
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
Then, we use the Controller to manage and persist the browser state across multiple agents. It allows agents to share a browsing session, maintaining consistency in cookies, sessions, and tabs.

# Persist browser state across agents
controller = Controller()
Let’s initialise the agent to find and extract information about “LoRA LLM” by searching in Google. then we use the chatOpenai model to process and analyze the content which is connected to the controller to maintain the browser state.

# Initialize browser agent
agent = Agent(
    task="Go to google.com and find the article about Lora llm and extract everything about Lora",
    llm=ChatOpenAI(model="gpt-4o", timeout=25, stop=None),
    controller=controller)
Also, we can initialise another agent, but it is optional and depends on how many agents you want to include in your code. They can do different tasks, but you need to manage each agent to a different task

agent = Agent(
    task="Go to google.com and find the article Supervised llm and extract everything about Supervised Fine-Tuning",
    llm=ChatOpenAI(model="gpt-4o", timeout=25, stop=None),
    controller=controller)
Then we define an asynchronous function for the concurrent execution of tasks we set the max step of the agent limit to 20 but feel free to set any number you want At each step, the agent performs an action that Represents what the agent plans to do next and the result Contains the output of the step, including whether the task is complete and any data extracted. If the task is completed, the extracted content is saved to a file named text.txt, and the process terminates.

async def main():
    max_steps = 20
    # Run the agent step by step
    for i in range(max_steps):
        print(f'\n📍 Step {i+1}')
        action, result = await agent1.step()

        print('Action:', action)
        print('Result:', result)

        if result.done:
            print('\n✅ Task completed successfully!')
            print('Extracted content:', result.extracted_content)
            
            # Save extracted content to a text file
            with open('text.txt', 'w') as file:
                file.write(result.extracted_content)
            print("Extracted content has been saved to text.txt")
            
            break

asyncio.run(main())
Now we define the working directory and check if a directory named dickens exists in the current working directory. If it doesn't, the program creates it. This ensures the directory is available for storing files or other resources.

WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
The primary step is configuring the LightRAG instance with the necessary parameters. We initialized with a working directory (./dickens) and a lightweight GPT-4o model (gpt_4o_mini_complete) as the default language model. This setup is efficient for retrieval-augmented tasks, with the flexibility to use a more robust model (gpt_4o_complete) if needed.

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)
we read the contents text.txt from the specified path and insert it into the RAG system using the rag.insert()

with open("C:/Users/mrtar/Desktop/lightrag/text.txt") as f:
    rag.insert(f.read())
We perform a naive search for the query “What is Supervised Fine-Tuning” in the RAG system. In naive search mode, the system looks for documents or entries that directly contain the keywords in the query, without considering any relationships or context around those terms. It is useful for straightforward queries that don’t need complex reasoning. It will return the results based purely on keyword matching.

# Perform naive search
print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="naive")))
Also, we performed a local search for the query “What is Supervised Fine-Tuning?” In local search mode, the system retrieves information about the query and its immediate neighbours (directly related entities). It will provide additional context, focusing on close relationships directly related to “Supervised Fine-Tuning.

Search is more detailed than naive and valuable when you need more context about direct connections or relationships.

# Perform local search
print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="local")))
Now, we use a global search for the query “What is Supervised Fine-Tuning.”The system considers the entire knowledge graph in global search mode, looking at direct and indirect relationships across a broader scope. It examines all possible connections related to “Supervised Fine-Tuning,” not just the immediate ones. It provides a comprehensive overview and is ideal for queries needing a wide-ranging context or a global relationship perspective.

# Perform global search
print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="global")))
Finally, we perform a hybrid search for the query "What is Supervised Fine-Tuning.” Hybrid search mode combines the benefits of both local and global searches. It retrieves information based on direct relationships (like local search) but also considers indirect or global relationships (like global search). It provides a balanced and thorough context, suitable for most scenarios, especially when it is essential to understand the overall and specific context.

# Perform hybrid search
print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="hybrid")))\
Conclusion :
More than just technological advancements, LightRAG and Browser-Use can potentially fundamentally change how we interact with information. They offer more accurate and comprehensive search capabilities, precise answers to complex questions, and responses that always reflect the latest knowledge.

If these goals are realized, they could revolutionize fields such as education, research, and business. LightRAG and Browser-Use represent groundbreaking technology that will open up the next generation of information search and generation. I’m really looking forward to seeing how it develops in the future!

