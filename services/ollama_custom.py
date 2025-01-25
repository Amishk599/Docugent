import os
import requests
from typing import List, Optional, ClassVar
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain.schema import (
    ChatMessage,
    AIMessage,
    ChatResult
)

# implement a custom chat model
class ChatLocalOllamaMistral(BaseChatModel):
    """
    A custom chat model that interfaces with a locally installed Ollama Mistral model
    """
    model_name: str = Field(default="mistral", alias="model")
    host: str = os.getenv('OLLAMA_HOST')
    
    @property
    def _llm_type(self) -> str:
        return "local-ollama-mistral"
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,) -> ChatResult:
        """
        Process a list of message and generate a response
        """
        formatted_messages = self._format_messages(messages)
        response = self._send_request(formatted_messages)

        # form ai message for the received response
        ai_message = AIMessage(
            content=response.get("response", ""),
            additional_kwargs={
                "model_name": response.get("model", ""),
            }
        )

        # return as ChatResult
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)]
        )
    
    def _send_request(self, prompt: str) -> str:
        """
        Sends a request to the local Ollama Mistral API
        """
        if not self.host:
            raise ValueError("The 'host' value is not set. Please set the OLLAMA_HOST environment variable.")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        # TODO : Add retries
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
            )
            # print(f"response from API: {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed with error: {e}")
            raise RuntimeError("Failed to generate response")
        
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """
        Convert LangChain messages into a single string format
        """
        formatted_messages = ""
        for message in messages:
            if isinstance(message, AIMessage):
                formatted_messages += f"AI: {message.content}\n"
            else:
                formatted_messages += f"User: {message.content}\n"
        return formatted_messages.strip()