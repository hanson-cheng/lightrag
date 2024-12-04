from langchain_openai import ChatOpenAI
from typing import Optional

def gpt_4o_mini_complete(prompt: str, temperature: Optional[float] = 0.7) -> str:
    """Lightweight GPT-4 completion function"""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=temperature,
        timeout=25
    )
    return llm.predict(prompt)

def gpt_4o_complete(prompt: str, temperature: Optional[float] = 0.7) -> str:
    """Full GPT-4 completion function with longer context"""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=temperature,
        timeout=60
    )
    return llm.predict(prompt)
