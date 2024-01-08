import gradio as gr
import os
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

def process(filepath):
    audio = open(filepath, "rb")
    key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key
    transcript = openai.Audio.transcribe("whisper-1", audio)

    llm = OpenAI(
        temperature=1,
        openai_api_key=key)

    return llm(transcript["text"])

demo = gr.Interface(
    fn=process,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=["text"],
)

demo.launch()
