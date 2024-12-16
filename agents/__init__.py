from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, Any, List
import os
import requests
from config import AVAILABLE_MODELS

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_free_tier = not bool(os.getenv("OPENAI_API_KEY"))
        
        if self.is_free_tier:
            try:
                # Use HuggingFace's hosted models
                model_name = config.get("model", "mistralai/Mistral-7B-Instruct-v0.1")
                
                # Validate model is in free tier list
                if model_name not in AVAILABLE_MODELS["free"]:
                    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Default to Mistral if invalid model
                
                self.llm = HuggingFaceHub(
                    repo_id=model_name,
                    task="text-generation",
                    temperature=config.get("temperature", 0.7),
                    max_length=config.get("max_tokens", 1000),
                    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY", "")
                )
            except Exception as e:
                raise RuntimeError(f"Error initializing HuggingFace model: {str(e)}")
        else:
            # Ensure we're using a premium model
            model_name = config.get("model")
            if model_name not in AVAILABLE_MODELS["premium"]:
                model_name = "gpt-4o-mini"  # Default to gpt-4o-mini if invalid model
            
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000)
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
    
    def _create_prompt(self, role: str, content: str) -> str:
        """Create a prompt for the LLM"""
        if self.is_free_tier:
            # For HuggingFace, combine role and content into a single prompt
            prompt = f"<|system|>{role}</s><|user|>{content}</s><|assistant|>"
            return self.llm.invoke(prompt)
        else:
            # For OpenAI, use the chat format
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