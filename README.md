# LLM Wrapper
A simple wrapper for AzureOpenAI's ChatGPT API.


```
pip install git+https://github.com/Masatsugar/llm-wrapper.git
```

## Usage

- Basic usage


```python
from llm_wrapper import ChatGPT, ChatGPTConfig

config = ChatGPTConfig(
    deployment_id=<MODEL ID>,
    azure_endpoint=<Azure Endpoint>,
    api_key=<Your API KEY>,
    api_version="2024-09-01-preview",
)
chatgpt = ChatGPT(config)
response= chatgpt("Hello, how are you?")
print(response)
```

- Use tool

```python
from llm_wrapper.utils import make_tool

# 1. First, define your function.
def get_current_time():
    """Get the current time in ISO 8601 format."""
    from datetime import datetime
    return datetime.now().isoformat()

# 2. Manage all the functions you want to register in one place.
my_functions = [get_current_time]

# 3. Automatically generate both tools and available_functions from my_functions.
tools = [make_tool(f) for f in my_functions]
available_functions = {f.__name__: f for f in my_functions}

# 5. Update config and create the ChatGPT instance.
config.tools = tools
config.available_functions = available_functions
chatgpt = ChatGPT(config)

# 6. Run a query.
response = chatgpt("What time is it now?")
print(response)
```