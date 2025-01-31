import os
import base64
import requests

from PIL import Image
from io import BytesIO
import json

from pydantic import BaseModel

import openai

_client = None  # Global variable to store the OpenAI client


def get_client(azure_endpoint=None, api_key=None, api_version=None):
    if azure_endpoint is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_version is None:
        api_version = "2024-09-01-preview"

    global _client
    if _client is None:
        if not azure_endpoint or not api_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY is not set.")
        _client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def run_chatgpt(
    model: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant",
    **kwargs,
) -> str:
    """Run a single chatgpt call to address the `user_prompt`."""
    client = get_client(**kwargs)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        **kwargs,
    )
    return response.choices[0].message.content


def JSON_llm(model, user_prompt: str, schema, system_prompt: str = None, **kwargs):
    client = get_client(**kwargs)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    extract = client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        response_format=schema,
        **kwargs,
    )
    return json.loads(extract.choices[0].message.content)


def fetch_image_from_web(url: str) -> Image.Image:
    """
    Fetch an image from a given URL and return it as a PIL Image object.

    Parameters
    ----------
    url : str
        URL of the image to fetch.

    Returns
    -------
    PIL.Image.Image
        The image as a PIL Image object.

    Raises
    ------
    ValueError
        If the URL does not point to a valid image.
    """
    try:
        # Send a GET request to fetch the image
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Convert the content of the response to a PIL Image
        image = Image.open(BytesIO(response.content))
        return image

    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL: {e}")


def image_file_to_base64(image_path: str) -> str:
    """
    Encode an image from a file path into a Base64 string.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    str
        Base64-encoded string of the image.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    IOError
        If there is an issue reading the file.

    Examples
    --------
    >>> encoded = image_file_to_base64("path/to/image.png")
    >>> print(encoded)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading file {image_path}: {e}")


def pil_image_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL Image object into a Base64 string.

    Parameters
    ----------
    image : PIL.Image.Image
        A PIL Image object to encode.

    Returns
    -------
    str
        Base64-encoded string of the image.

    Raises
    ------
    ValueError
        If the image cannot be saved to a buffer.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open("path/to/image.png")
    >>> encoded = pil_image_to_base64(img)
    >>> print(encoded)
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except ValueError as e:
        raise ValueError(f"Error encoding image to Base64: {e}")


def chain_of_thoughts(chatgpt, messages):
    """
    Chain of Thoughts from "https://platform.openai.com/docs/guides/structured-outputs"
    """

    class Step(BaseModel):
        explanation: str
        output: str

    class Reasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    response = chatgpt.client.beta.chat.completions.parse(
        model=chatgpt.config.deployment_id,
        messages=messages,
        response_format=Reasoning,
    )
    return response


def make_tool(func):
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",  # Use the docstring for the description.
        },
    }
