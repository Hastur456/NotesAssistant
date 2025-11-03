from openai import OpenAI
import os
from dotenv import load_dotenv


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

print(response.choices[0].message.content)
