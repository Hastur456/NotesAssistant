from langchain_amvera import AmveraLLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = AmveraLLM(model="llama70b", api_token=os.getenv("AMVERA_API_TOKEN"))

response = llm.invoke("Объясни принципы работы нейросетей простым языком")
print(response.content)