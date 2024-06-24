import dspy
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

class LLM_API:
    def __init__(self) -> None:
        load_dotenv()
        self.turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', api_key = os.getenv('OPENAI_API_KEY'))
        
        self.client = OpenAI(api_key = os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

        
        
    def generate_text(self, prompt, model = "gpt3.5"):
        if model == "gpt3.5":
            response = self.turbo(prompt)
            return response[0]
        elif model == "deepseek":
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
    
  