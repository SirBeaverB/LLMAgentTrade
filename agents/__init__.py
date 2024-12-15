from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, Any, List

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.llm = ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        self.memory: List[Dict[str, Any]] = []
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the given data and return insights
        
        Args:
            data: Dictionary containing relevant data for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def _create_prompt(self, role: str, content: str) -> List[SystemMessage | HumanMessage]:
        """Create a prompt for the LLM"""
        messages = [
            SystemMessage(content=role),
            HumanMessage(content=content)
        ]
        return self.llm.invoke(messages).content
    
    def save_to_memory(self, interaction: Dict[str, Any]):
        """Save interaction to agent's memory"""
        self.memory.append(interaction)
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Retrieve agent's memory"""
        return self.memory