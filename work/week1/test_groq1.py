'''
This script demonstrates how to use the Groq API to get a chat completion response.'''
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "user",
            "content": "What is Groq?",
        }
        ]
    )
print(response.choices[0].message.content)
print('Works!!')