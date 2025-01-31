# Agent Recipes:
# based on "https://www.agentrecipes.com/"
import os
from typing import List, Tuple, Literal, Dict

from pydantic import BaseModel, Field

from llm_wrapper import ChatGPT, ChatGPTConfig
from llm_wrapper.utils import run_chatgpt, JSON_llm


def serial_chain_workflow(input_query: str, prompt_chain: List[str]) -> List[str]:
    """Run a serial chain of LLM calls to address the `input_query`
    using a list of prompts specified in `prompt_chain`.
    """
    response_chain = []
    response = input_query
    print(f"Input:\n{response}\n")
    for i, prompt in enumerate(prompt_chain):
        print(f"Step {i+1}")
        response = run_chatgpt(f"{prompt}\nInput:\n{response}", model="gpt-4o")
        response_chain.append(response)
        print(f"{response}\n")
    return response_chain


def serial_chain_workflow_with_history(
    chatgpt: ChatGPT, input_query: str, prompt_chain: List[str]
) -> Tuple[List[str], ChatGPT]:
    """
    Execute a chain of prompts in series using the same ChatGPT instance,
    preserving conversation history.

    Unlike `serial_chain_workflow`, this function does not reset
    the conversation at each step. The LLM retains context from previous
    prompts and responses, making it possible to maintain a coherent dialogue.

    Args:
        chatgpt (ChatGPT): An instance of ChatGPT that keeps track of the conversation.
        input_query (str): The initial query or message to start the chain.
        prompt_chain (List[str]): A list of prompt strings to be used in sequence.

    Returns:
        Tuple[List[str], ChatGPT]:
            - A list of responses from each step in the prompt chain.
            - The updated ChatGPT instance with the conversation history.
    """
    _response_chain = []
    response = input_query
    for i, prompt in enumerate(prompt_chain):
        print(f"Step {i+1}")
        response = chatgpt(f"{prompt}\nInput:\n{response}" if i == 0 else prompt)
        _response_chain.append(response)
        print(f"{response}\n")
    return _response_chain, chatgpt


def test_serial_chain_workflow(is_chat_history=False):
    question = "Sally earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

    prompt_chain = [
        """Given the math problem, ONLY extract any relevant numerical information and how it can be used.""",
        """Given the numerical information extracted, ONLY express the steps you would take to solve the problem.""",
        """Given the steps, express the final answer to the problem.""",
    ]

    if is_chat_history:
        config = ChatGPTConfig(
            deployment_id="gpt-4o",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_Dev"),
            api_key=os.getenv("OPENAI_API_KEY_Dev"),
            api_version="2024-09-01-preview",
        )
        chatgpt = ChatGPT(config)
        responses, chatgpt = serial_chain_workflow_with_history(
            chatgpt, question, prompt_chain
        )
    else:
        responses = serial_chain_workflow(question, prompt_chain)

    final_answer = responses[-1]
    print(f"Final answer: {final_answer}")
    # print("Test passed!", chatgpt.messages)


def router_workflow(input_query: str, routes: Dict[str, str]) -> str:
    """Given a `input_query` and a dictionary of `routes` containing options and details for each.
    Selects the best model for the task and return the response from the model.
    """
    ROUTER_PROMPT = """Given a user prompt/query: {user_query}, select the best option out of the following routes:
    {routes}. Answer only in JSON format."""

    # Create a schema from the routes dictionary
    class Schema(BaseModel):
        route: Literal[tuple(routes.keys())]
        reason: str = Field(
            description="Short one-liner explanation why this route was selected for the task in the prompt/query."
        )

    # Call LLM to select route
    selected_route = JSON_llm(
        ROUTER_PROMPT.format(user_query=input_query, routes=routes), Schema
    )
    print(
        f"Selected route:{selected_route['route']}\nReason: {selected_route['reason']}\n"
    )

    # Use LLM on selected route.
    # Could also have different prompts that need to be used for each route.
    response = run_chatgpt(user_prompt=input_query, model=selected_route["route"])
    print(f"Response: {response}\n")

    return response


def test_routing():
    prompt_list = [
        "Produce python snippet to check to see if a number is prime or not.",
        "Plan and provide a short itenary for a 2 week vacation in Europe.",
        "Write a short story about a dragon and a knight.",
    ]

    # model_routes = {
    #     "Qwen/Qwen2.5-Coder-32B-Instruct": "Best model choice for code generation tasks.",
    #     "Gryphe/MythoMax-L2-13b": "Best model choice for story-telling, role-playing and fantasy tasks.",
    #     "Qwen/QwQ-32B-Preview": "Best model for reasoning, planning and multi-step tasks",
    # }

    # This is a test routes dictionary
    model_routes = {
        "gpt-4o": "Best model choice for code generation tasks.",
        "gpt-4o-mini": "Best model choice for story-telling, role-playing and fantasy tasks.",
    }
    for i, prompt in enumerate(prompt_list):
        print(f"Task {i + 1}: {prompt}\n")
        print(20 * "==")
        router_workflow(prompt, model_routes)


def test():
    test_serial_chain_workflow()
    # test_routing()


if __name__ == "__main__":
    test()
