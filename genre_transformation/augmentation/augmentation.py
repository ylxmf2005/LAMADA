import os
import random
import json
import sys
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_api import get_llm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dotenv import load_dotenv

load_dotenv()

# One independent statement(s) JSON list with the format like [\"statement\"]. The length of the JSON list is 1. Each statement is a brief opinion relevant to the text. Your response should begin with '[' and end with ']'.

# Are these statements sufficiently meaningful and specific and can serve as a central argument in an argumentative essay (shoubld be a point to prove, not a fact that has been defined)?

# Please revise the statement to present a clear, independent(not mention external text directly like \"The text ...\"), debatable, and well-defined thesis suitable for an argumentative essay."

# Are the essay's arguments well-supported by evidence from the text and presented with a coherent structure?

class AugmentationTransformer:
    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()
        self.generate_statement_prompt = PromptTemplate.from_template
        (
        '''
        {text}
        As a professional argumentative essay writer, your task is to generate a clear, independent, debatable, and well-defined thesis statement based on the provided text. The thesis statement should be a point to prove, not a fact that has been defined. The thesis statement must be strong enough to serve as the central argument in an argumentative essay. You don't need to write the entire essay, just provide the thesis statement.                             
        '''
        )

    def transform_dataset(self, dataset):
        output_list = []
        for data in dataset:
            pass
        return output_list

    def save_results(self, output_list, save_path="result/genre_transformation/augmentation.json"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output_list, f, indent=4)

if __name__ == "__main__":
    loader = JSONLoader(
        file_path='./data/wikibooks.jsonl',
        jq_schema='.text',
        text_content=False,
        json_lines=True
    )
    dataset = loader.load()
    random.shuffle(dataset)
   
    print(dataset[0], dataset[1], sep = '\n')
    dataset = dataset[:10] # For testing
    
    summarizer = AugmentationTransformer()

    output_list = summarizer.transform_dataset(dataset)
    summarizer.save_results(output_list)