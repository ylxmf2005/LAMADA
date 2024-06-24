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

class SummaryTransformer:
    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()

        self.overall_summary_prompt = PromptTemplate.from_template("""{text}
As a professional summarizer, create a concise and comprehensive summary of the provided text, while adhering to these guidelines:
1. Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
2. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
3. Ensure that the summary is self-contained and does not require the reader to refer back to the original text for context.""")

        self.different_perspectives_prompt = PromptTemplate.from_template("""{text}
As a professional summarizer, summarize this text from 2~5 different directions(can be different perspectives, aspects, components, etc.). Each direction you pick should be content-rich and reflect specific insights or themes found in the original text that are different from the other directions. Avoid generic direction like content overview.
The summaries should be an ordered list (each point is a direction), insightful, and tailored to the text's nuances and themes.
Exclude and avoid using "The text", "The article", "The summary", "the view", "the direction", etc. Instead, use concept/entity in the original text to start each summary and make each summary self-contained.""")

        self.overall_summary_chain = self.overall_summary_prompt | self.llm | self.parser
        self.different_perspectives_chain = self.different_perspectives_prompt | self.llm | self.parser

    def transform_dataset(self, dataset):
        output_list = []
        for data in dataset:
            summary = self.overall_summary_chain.invoke({"text": data.page_content})
            output_list.append({"original_text": data.page_content, "transformed_text": summary, "type": "overall_summary", "tag": []})

            summary = self.different_perspectives_chain.invoke({"text": data.page_content})
            output_list.append({"original_text": data.page_content, "transformed_text": summary, "type": "different_perspectives", "tag": []})

        return output_list

    def save_results(self, output_list, save_path="result/genre_transformation/summary.json"):
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
    
    summarizer = SummaryTransformer()

    output_list = summarizer.transform_dataset(dataset)
    summarizer.save_results(output_list)