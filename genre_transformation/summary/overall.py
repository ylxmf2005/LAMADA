import os
import json
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_api import get_llm

load_dotenv()

class OverallSummaryTransformer:
    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()

        self.overall_summary_prompt = PromptTemplate.from_template("""{text}
        As a professional summarizer, create a concise and comprehensive summary of the provided text, while adhering to these guidelines:
        1. Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
        2. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
        3. Ensure that the summary is self-contained and does not require the reader to refer back to the original text for context.""")

        self.overall_summary_chain = self.overall_summary_prompt | self.llm | self.parser

    def transform_dataset(self, dataset):
        output_list = []
        for data in dataset:
            summary = self.overall_summary_chain.invoke({"text": data.page_content})
            output_list.append({"original_text": data.page_content, "transformed_text": summary, "type": "overall_summary", "tag": []})

        return output_list

    def save_results(self, output_list, save_path="result/genre_transformation/overall_summary.json"):
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

    summarizer = OverallSummaryTransformer()

    output_list = summarizer.transform_dataset(dataset[:10])  # For testing
    summarizer.save_results(output_list)
