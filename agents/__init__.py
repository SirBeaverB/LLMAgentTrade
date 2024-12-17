from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, Any, List
import os
import requests
from config import AVAILABLE_MODELS
import huggingface_hub

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_free_tier = not bool(os.getenv("OPENAI_API_KEY"))
        
        if self.is_free_tier:
            try:
                # Use HuggingFace's hosted models
                model_name = config.get("model", "gpt2")
                
                # Validate model is in free tier list
                if model_name not in AVAILABLE_MODELS["free"]:
                    model_name = "gpt2"  # Default to gpt2 if invalid
                
                # Get HuggingFace token from config
                hf_token = config.get("huggingface_api_key")
                if not hf_token:
                    raise ValueError("HuggingFace API key not provided in config")
                
                # Initialize HuggingFaceEndpoint with proper configuration
                self.llm = HuggingFaceEndpoint(
                    endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
                    huggingfacehub_api_token=hf_token,
                    task="text-generation",
                    temperature=config.get("temperature", 0.7),
                    max_new_tokens=256,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    return_full_text=False
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
            model_name = self.config.get("model", "gpt2")
            
            # Even more aggressive truncation for content
            max_content_tokens = 150  # Reduced from 250
            content_words = content.split()
            if len(content_words) > max_content_tokens:
                first_part = ' '.join(content_words[:75])
                last_part = ' '.join(content_words[-75:])
                content = f"{first_part}...[truncated]...{last_part}"
            
            # More aggressive role truncation
            role_words = role.split()
            if len(role_words) > 30:  # Reduced from 50
                role = ' '.join(role_words[:30]) + "..."
            
            if "llama" in model_name.lower():
                prompt = f"[INST]{role}[/INST]{content}"  # Minimized format
            elif "t5" in model_name.lower():
                prompt = f"System:{role} Input:{content}"  # Minimized format
            else:
                prompt = f"{role}\nQ:{content}\nA:"  # Most minimal format
            
            try:
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                raise RuntimeError(f"Error generating response: {str(e)}")
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