from langchain_amvera import AmveraLLM
from dotenv import load_dotenv
import os

load_dotenv()

amvera_llm = AmveraLLM(
        model="llama8b", 
        api_token=os.getenv("AMVERA_API_TOKEN"), 
        temperature=0.4,
        max_tokens=1000
    )
