import os
import json
from typing import Optional, List, Dict, Any, Callable

import fire
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from PIL import Image
from llm_wrapper.utils import pil_image_to_base64


class ChatGPTConfig:
    def __init__(
        self,
        deployment_id: str,
        azure_endpoint: str,
        api_version: str,
        api_key: Optional[str] = None,
        instruction: str = "You are a helpful assistant.",
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[list] = None,
        seed: int = 42,
        tools: Optional[List[Dict[str, Any]]] = None,
        available_functions: Optional[Dict[str, Callable]] = None,
        max_retry: int = 10,
    ):
        self.deployment_id = deployment_id
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self._api_key = api_key
        self.instruction = instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.seed = seed
        self.tools = tools
        self.available_functions = available_functions
        self.max_retry = max_retry

    @property
    def settings(self) -> dict:
        base_settings = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "seed": self.seed,
            "n": 1,  # TODO: Add support for multiple completions
        }
        if self.tools:
            base_settings.update({"tools": self.tools, "tool_choice": "auto"})
        return base_settings

    def __repr__(self):
        return f"<ChatGPTConfig deployment_id={self.deployment_id}>"


class ChatGPT:
    def __init__(self, config: ChatGPTConfig):
        self.config = config
        self.client = openai.AzureOpenAI(
            api_key=self.config._api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.azure_endpoint,
        )
        self.messages = [
            {"role": "system", "content": self.config.instruction},
        ]
        self.usages = []
        self.retry_counter = 0

    def __call__(self, content: str, image: Optional[Image.Image] = None) -> str:
        if image:
            base64_image = pil_image_to_base64(image)
            image_url = f"data:image/jpeg;base64,{base64_image}"
            content = [
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "detail": "high",
                },
            ]

        self.messages.append({"role": "user", "content": content})
        response = self.generate_response()
        self.usages.append(response.usage.model_dump())
        return response.choices[0].message.content

    def reset(self):
        self.messages = [
            {"role": "system", "content": self.config.instruction},
        ]
        self.usages = []

    def generate_response(self):
        _response = self._generate_by_azure_openai(self.messages)
        response_message = _response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", [])
        self.messages.append(response_message)
        if tool_calls and self.config.available_functions is not None:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # obtain available function
                function_to_call = self.config.available_functions.get(function_name)
                if not function_to_call:
                    continue

                if function_args:
                    function_response = function_to_call(**function_args)
                else:
                    function_response = function_to_call()

                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                _response = self._generate_by_azure_openai(self.messages)

        return _response

    @retry(
        wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3)
    )
    def _generate_by_azure_openai(self, messages):
        """
        Call Azure OpenAI's ChatCompletion API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=messages,
                **self.config.settings,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise


def test_response(
    model_id: Optional[str] = "gpt-4o",
    azure_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    tool: bool = False,
    cot: bool = False,
    image: bool = False,
    **kwargs,
) -> None:
    config = ChatGPTConfig(
        deployment_id=os.getenv("MODEL_ID", model_id),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_Dev", azure_endpoint),
        api_key=os.getenv("OPENAI_API_KEY_Dev", api_key),
        api_version="2024-09-01-preview",
    )

    if config.azure_endpoint is None:
        raise ValueError("Please set the environment variable AZURE_OPENAI_ENDPOINT")
    if config.api_key is None:
        raise ValueError("Please set the environment variable OPENAI_API_KEY")

    # test_tool / test_cot / normal
    query = "What time is it now?"
    chatgpt = ChatGPT(config)
    if tool:
        test_tool(config, query)
    elif cot:
        test_cot(chatgpt)
    elif image:
        test_image(chatgpt)
    else:
        print("Question:", query)
        response = chatgpt(query)
        print(response)


def test_tool(config: ChatGPTConfig, query: str):
    """
    test for function calling
    """

    def get_current_time():
        """Get the current time in ISO 8601 format."""
        from datetime import datetime

        return datetime.now().isoformat()

    func_name = get_current_time.__name__
    tools = [
        {
            "type": "function",
            "function": {
                "name": func_name,
                "description": "Get the current time in ISO 8601 format.",
            },
        },
    ]

    available_functions = {
        func_name: get_current_time,
    }

    # update config and create an instance of ChatGPT
    config.tools = tools
    config.available_functions = available_functions

    chatgpt = ChatGPT(config)
    print("Question:", query)
    response = chatgpt(query)
    print(response)


def test_cot(chatgpt: ChatGPT):
    """
    Test for CoT(Chain of Thoughts)
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "How can I solve 8x + 7 = -23?"},
    ]
    completion = chatgpt.chain_of_thoughts(messages)
    print(completion.choices[0].message.parsed)


def test_image(chatgpt: ChatGPT):
    """
    Test for image
    """
    query = "What is this?"
    from llm_wrapper.utils import fetch_image_from_web

    image = fetch_image_from_web(
        "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"
    )
    return chatgpt(query, image=image)


if __name__ == "__main__":
    fire.Fire(test_response)
