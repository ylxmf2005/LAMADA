import os
import json
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_api import get_llm

load_dotenv()

class DifferentPerspectivesTransformer:
    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()

        self.different_perspectives_prompt = PromptTemplate.from_template("""{text}
        As a professional summarizer, summarize this text from 2~5 different directions (can be different perspectives, aspects, components, etc.). Each direction you pick should be content-rich and reflect specific insights or themes found in the original text that are different from the other directions. Avoid generic direction like content overview.
        The summaries should be an ordered list (each point is a direction), insightful, and tailored to the text's nuances and themes.
        Exclude and avoid using "The text", "The article", "The summary", "the view", "the direction", etc. Instead, use concept/entity in the original text to start each summary and make each summary self-contained.""")

        self.different_perspectives_chain = self.different_perspectives_prompt | self.llm | self.parser

    def transform_dataset(self, dataset):
        output_list = []
        for data in dataset:
            summary = self.different_perspectives_chain.invoke({"text": data.page_content})
            output_list.append({"original_text": data.page_content, "transformed_text": summary, "type": "different_perspectives", "tag": []})

        return output_list

    def save_results(self, output_list, save_path="result/genre_transformation/different_perspectives_summary.json"):
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

    summarizer = DifferentPerspectivesTransformer()

    output_list = summarizer.transform_dataset(dataset[:10])  # For testing
    summarizer.save_results(output_list)
