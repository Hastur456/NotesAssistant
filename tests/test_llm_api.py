from openai import OpenAI
import os
from dotenv import load_dotenv
from pprint import pprint
from langchain.messages import AIMessage, HumanMessage
from langchain_amvera import AmveraLLM

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AGENT.tools import FileOperationTools

load_dotenv()


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

tools = FileOperationTools("./tests/testnotes").create_tools()
amvera_llm = AmveraLLM(
        model="llama8b", 
        api_token=os.getenv("AMVERA_API_TOKEN"), 
        temperature=0.4,
        max_tokens=1000,
        tools=tools
    )

tools = FileOperationTools("./tests/testnotes").create_tools()
llm_with_tools = amvera_llm.bind_tools(tools=tools)

print(hasattr(amvera_llm, 'bind_tools'))
print(amvera_llm.__class__.__mro__)  # посмотрите иерархию классов)

response = amvera_llm.invoke([HumanMessage(content="Используя инструменты которые я тебе дал выведи структуру директроии ./tests/testnotes.")])
response_with_tools = llm_with_tools.invoke([HumanMessage(content="Выведи структуру директроии.")])


print("Исполльзование инструмента из tools:")
print(tools[-1].invoke({"root_path": "./tests/testnotes"}))
print()

print("========== Ответ нейросети ==========")
print(response)
print("========== Ответ нейросети ==========")
print(response.content)
print("=====================================")
print("========== Ответ нейросети с инструментами==========")
print(response_with_tools)
print("========== Ответ нейросети с инструментами ==========")
print(response_with_tools.content)
