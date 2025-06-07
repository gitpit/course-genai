'''
first chatbot app
'''
from dotenv import load_dotenv
import gradio  as gr
from groq import Groq
import os

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def chat(message, history):
    history = [{"role":msg["role"], "content": msg["content"]} for msg in history]
    messages = (
        [{"role": "system", "content": "system message"}] +
        history +
        [{"role": "user", "content": message}]
    )
    stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response

gr.ChatInterface(fn=chat,type="messages").launch()