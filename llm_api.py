import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from dsp import LM

load_dotenv()
def get_llm(model_name = "deepseek-chat"):
    if (model_name == "deepseek-chat"):
        return ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=os.getenv('DEEPSEEK_API_KEY'), 
            openai_api_base='https://api.deepseek.com',
            max_tokens=4096
        )