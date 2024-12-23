import os
import json
from typing import Optional

import fire
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


class ChatGPT:
    def __init__(
        self,
        deployment_id: str,
        azure_endpoint: str,
        api_version: str,
        instruction="You are a helpful assistant.",
        api_key=None,
        tools=None,
        available_functions=None,
        temperature=0.5,
        max_tokens=None,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        seed=42,
    ):
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.instruction = instruction
        self.messages = [
            {"role": "system", "content": self.instruction},
        ]
        self.usages = []
        self.deployment_id = deployment_id
        self.retry_counter = 0
        self.max_retry = 10
        self.settings = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "seed": seed,
            "n": 1,  # TODO: Number of completions to generate for each prompt
        }
        self.tools = tools
        self.available_functions = available_functions
        if self.tools:
            self.tool_choice = "auto"
            self.settings.update({"tools": self.tools, "tool_choice": self.tool_choice})

    def generate_response(self):
        _response = self.generate_by_azure_openai(self.messages)
        response_message = _response.choices[0].message
        tool_calls = response_message.tool_calls
        self.messages.append(response_message)
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
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
                _response = self.generate_by_azure_openai(self.messages)
        return _response

    def __call__(self, content):
        self.messages.append({"role": "user", "content": content})
        _response = self.generate_response()
        self.usages.append(_response.usage.model_dump())
        return _response.choices[0].message.content

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3)
    )
    def generate_by_azure_openai(
        self,
        messages,
    ):
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=messages,
                **self.settings,
            )
            return response

        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def reset(self):
        self.messages = [
            {"role": "system", "content": self.instruction},
        ]
        self.usages = []


def test_response(
    query: str,
    azure_endpoint: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    tool: bool = False,
    **kwargs,
) -> None:
    """Generate Response

    Parameters
    ----------
    query: str
        Your question for chatgpt.
    region: str
        Choose the region for the deployment.
    model_id: str
    tool: bool
    kwargs

    Returns
    -------

    """
    config = {
        "deployment_id": os.getenv("MODEL_ID", model_id),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", azure_endpoint),
        "api_key": os.getenv("OPENAI_API_KEY", api_key),
        "api_version": "2024-09-01-preview",
    }
    if config["azure_endpoint"] is None:
        raise ValueError("Please set the environment variable AZURE_ENDPOINT")
    if config["api_key"] is None:
        raise ValueError("Please set the environment variable OPENAI_API_KEY")

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
    if tool:
        chatgpt = ChatGPT(
            **config,
            tools=tools,
            available_functions=available_functions,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
    else:
        chatgpt = ChatGPT(**config)

    print("Question:", query)
    response = chatgpt(query)
    print(response)


if __name__ == "__main__":
    fire.Fire(test_response)
