import os
import json
import requests
from typing import List, Optional, ClassVar, Generator, Iterator
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain.schema import (
    ChatMessage,
    AIMessage,
    ChatResult,
)
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.language_models.chat_models import generate_from_stream

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
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the model.
        """
        formatted_messages = self._format_messages(messages)
        stream_generator = self._send_request(formatted_messages, stream=True)
        for chunk in stream_generator:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk),
            )

    def _generate(
            self, 
            messages: List[BaseMessage], 
            stop: Optional[List[str]] = None,
        ) -> ChatResult:
        """
        Process a list of message and generate a response
        """
        formatted_messages = self._format_messages(messages)
        response_generator = self._send_request(formatted_messages, stream=False)
        response_text = next(response_generator)

        # form ai message for the received response
        ai_message = AIMessage(
            content=response_text,
        )

        # return as ChatResult
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)]
        )
    
    def _send_request(self, prompt: str, stream: bool = False) -> Generator[str, None, None]:
        """
        Sends a request to the local Ollama Mistral API
        """
        if not self.host:
            raise ValueError("The 'host' value is not set. Please set the OLLAMA_HOST environment variable.")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=stream,
            )
            response.raise_for_status()
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            # parse the json response and extract the response text
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get("response", "")
                            # ignore empty chunk
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError as e:
                            print(f"JSON decoding error: {e}")
            else:
                try:
                    json_data = response.json()
                    result = json_data.get("response", "")
                    yield result
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error (non-stream): {e}")
                    raise RuntimeError("Failed to parse JSON response from server.")
            
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
    