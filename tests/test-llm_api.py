from openai import OpenAI
import os
from dotenv import load_dotenv
from pprint import pprint


def print_structure(data, indent=0):
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}dict \\")
        for key, value in data.items():
            print(f"{prefix}  {key}: ", end="")
            if isinstance(value, (dict, list)):
                print()
                print_structure(value, indent + 2)
            else:
                print(f"{type(value).__name__}")
        print(f"{prefix}\\")
    elif isinstance(data, list):
        print(f"{prefix}list [")
        if len(data) > 0:
            # Показать структуру первого элемента для примера
            print_structure(data[0], indent + 1)
            if len(data) > 1:
                print(f"{prefix}  ... ({len(data)} items total)")
        else:
            print(f"{prefix}  (empty)")
        print(f"{prefix}]")
    else:
        print(f"{prefix}{type(data).__name__}")


load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar",
    messages=[
        {"role": "system", "content": "Будь точным и кратким."},
        {"role": "user", "content": "Сколько звёзд в нашей галактике?"}
    ]
)

pprint(response.choices[0])
