"""Browser integration for LightRAG"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Do not change the AI model without explicit permission from the user
MODEL_NAME = "gpt-4o"  # Required model per project requirements

@dataclass
class BrowseResult:
    content: str
    url: str
    title: Optional[str] = None
    done: bool = False
    error: Optional[str] = None

class BrowserManager:
    def __init__(self, model_name: str = MODEL_NAME, timeout: int = 25):
        """Initialize browser manager with a controller to maintain state"""
        self.controller = Controller()
        self.model_name = model_name
        self.timeout = timeout
        logger.info(f"Initialized BrowserManager with model {model_name}")
        
    async def search_and_extract(self, query: str, max_steps: int = 20) -> BrowseResult:
        """Search the web and extract relevant information"""
        logger.info(f"Starting web search for query: {query}")
        try:
            agent = Agent(
                task=f"Go to google.com and find information about: {query}",
                llm=ChatOpenAI(
                    model=self.model_name,
                    timeout=self.timeout,
                    stop=None
                ),
                controller=self.controller
            )
            logger.info("Created browser agent")
            
            for i in range(max_steps):
                logger.info(f"Step {i+1}/{max_steps}")
                try:
                    action, result = await agent.step()
                    logger.info(f"Action: {action}")
                    
                    if result.done:
                        logger.info("Search completed successfully")
                        return BrowseResult(
                            content=result.extracted_content,
                            url=result.current_url,
                            title=result.page_title,
                            done=True
                        )
                        
                except Exception as step_error:
                    logger.error(f"Error during step {i+1}: {step_error}")
                    return BrowseResult(
                        content="",
                        url="",
                        error=str(step_error),
                        done=True
                    )
                    
            logger.warning("Max steps reached without completion")
            return BrowseResult(
                content="Max steps reached without completion",
                url="",
                done=True
            )
            
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            return BrowseResult(
                content="",
                url="",
                error=str(e),
                done=True
            )
    
    def close(self):
        """Clean up browser resources"""
        if hasattr(self, 'controller'):
            logger.info("Cleaning up browser resources")
            self.controller = None
