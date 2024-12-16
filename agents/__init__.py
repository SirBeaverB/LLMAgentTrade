from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, Any, List
import os
import subprocess
import requests
from config import AVAILABLE_MODELS

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_free_tier = not bool(os.getenv("OPENAI_API_KEY"))
        
        if self.is_free_tier:
            try:
                # Check if Ollama is running
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    raise ConnectionError("Ollama server is not running")
                
                # Get available models
                available_models = [model["name"] for model in response.json().get("models", [])]
                model_name = config.get("model", "llama2")
                
                # Validate model is in free tier list
                if model_name not in AVAILABLE_MODELS["free"]:
                    model_name = "llama2"  # Default to llama2 if invalid model
                
                # If model isn't available, try to pull it
                if model_name not in available_models:
                    subprocess.run(["ollama", "pull", model_name], check=True)
                
                self.llm = Ollama(
                    model=model_name,
                    temperature=config.get("temperature", 0.7)
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    "Could not connect to Ollama. Please ensure Ollama is installed and running. "
                    "Visit https://ollama.ai to install."
                )
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    f"Failed to pull model {model_name}. Please check if the model name is correct "
                    "and you have sufficient disk space."
                )
            except Exception as e:
                raise RuntimeError(f"Error initializing Ollama: {str(e)}")
        else:
            # Ensure we're using a premium model
            model_name = config.get("model")
            if model_name not in AVAILABLE_MODELS["premium"]:
                model_name = "gpt-4o-mini"
            
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
            # For Ollama, combine role and content into a single prompt
            prompt = f"{role}\n\n{content}"
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