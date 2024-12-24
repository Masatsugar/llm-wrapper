import os
import base64
import requests

from PIL import Image
from io import BytesIO


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
