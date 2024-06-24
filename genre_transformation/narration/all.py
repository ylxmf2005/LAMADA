# TODO: Add memory function
# TODO: Add assertion / feedback loop
import os
import re
import random
import Levenshtein
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_api import get_llm

class FuzzyDict:
    """
    A dictionary that can get the value by a key with a fuzzy match.
    """
    def __init__(self, dict):
        self.dict = dict
        
    def fuzzy_get(self, key, threshold=0.3):
        if key in self.dict:
            return key
        minn_distance   = 1000
        minn_key = None
        for k in self.dict:
            # print(f"key: {key}, k: {k}")
            edit_distance = Levenshtein.distance(k, key)
            if edit_distance < minn_distance and edit_distance / len(k) < threshold:
                minn_distance = edit_distance
                minn_key = k
        
        if minn_key:
            return minn_key
        else:
            return key

    def value_add(self, key, D=1):
        key = self.fuzzy_get(key)
        if key not in self.dict:
            self.dict[key] = D
        else:
            self.dict[key] += D

class NarrativeTransformer:
    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()
        
        self.can_be_modified_template = PromptTemplate.from_template("""
        Narrative types: Diary, Epistolary style, Prose, Novel.
        {text}
        Is this text meaningful (has enough content) and feasible to be rewritten as a narrative? (yes/no)
        """)
        
        self.template1 = PromptTemplate.from_template("""
        {text}
        As a professional narrative writing expert, you know:
        1. Types of narrative:
           - Diary
           - Blog
           - Epistolary style 
           - Prose
           - Novel
        2. Types of main characters:
           - Fictional person
           - Author themselves
           - Real people
           - Anthropomorphized animals/objects/concepts

        Your role is to guide another AI in transforming any given text into a narrative form. You won't be writing the narrative itself, but rather completing the following three tasks, which another AI will reference to generate the narrative:
        1. Summarize the content of the given text.
        2. Choose the type of narrative, Person(first, second, third), and the type of main characters.
        3. Based on the given text, analyze how to transform the text into the (type_of_narrative, person, type_of_main_characters).

        Output in a code block enclosed by triple backticks in the following format:
        ```
        1. {{summary}}
        2. Type of narrative: {{type_of_narrative}}; Type of main characters: {{type_of_main_characters}}.
        3. Analysis: {{analysis}}
        ```
        Do not provide any other information except the guidance of the three questions above.

        ---
        Memory area:
        Here is the number of times you have chosen specific type_of_narrative:
        {narrative_dict}, and the number of times you have chosen specific type_of_main_characters: {character_dict}.
        You need to diversify the selection of type_of_narrative to ensure an even distribution, in order to generate a diverse range of narratives.
        """)
        
        self.template2 = PromptTemplate.from_template("""
        {text}
        As a professional narrative writing expert, your task is to transform any given text, which could be of any type, into type_of_narrative. The requirements are:
        1. Avoid unnecessary filler content.
        2. Integrate sufficient background information about the given text in the narrative.
        3. Pay attention to causal logic in the narrative.
        4. Refer to the following suggestions to complete this task:
        {suggestions}
        """)
        
        self.narrative_dict = FuzzyDict({"Diary": 0, "Blog": 0, "Epistolary style": 0, "Prose": 0, "Novel": 0})
        self.character_dict = FuzzyDict({"Fictional person": 0, "Author themselves": 0, "Real people": 0, "Anthropomorphized animals/objects/concepts": 0})
        
        self.can_be_modified_chain = self.can_be_modified_template | self.llm | self.parser
        self.chain1 = self.template1 | self.llm | self.parser
        self.chain2 = self.template2 | self.llm | self.parser

    def yes_in_string(self, s):
        return "yes" in s.lower()

    def can_be_modified(self, text):
        response = self.can_be_modified_chain.invoke({"text": text})
        assert response is not None
        return self.yes_in_string(response)
    
    def extract_type_and_character(self, text):
        narrative_pattern = r"Type of narrative: ([^;]+);"
        character_pattern = r"Type of main characters: ([^.]+)\."
        narrative_match = re.search(narrative_pattern, text)
        character_match = re.search(character_pattern, text)
        
        type_of_narrative = narrative_match.group(1) if narrative_match else None
        type_of_main_characters = character_match.group(1) if character_match else None
        
        return (type_of_narrative, type_of_main_characters)
    
    def transform_dataset(self, dataset):
        output_list = []
        
        dataset = [data for data in dataset if self.can_be_modified(data.page_content)]
        
        for data in dataset:
            response = self.chain1.invoke({"text": data.page_content, "narrative_dict": self.narrative_dict.dict, "character_dict": self.character_dict.dict})
            
            # print("response:", response)
            
            type_of_narrative, type_of_main_characters = self.extract_type_and_character(response)
            
            # print("type_of_narrative:", type_of_narrative, "type_of_main_characters:", type_of_main_characters)

            self.narrative_dict.value_add(type_of_narrative)
            self.character_dict.value_add(type_of_main_characters)
            
            suggestions = {"type_of_narrative": type_of_narrative, "type_of_main_characters": type_of_main_characters}
            response = self.chain2.invoke({"text": data.page_content, "suggestions": suggestions})
            
            output_list.append({"original_text": data.page_content, "transformed_text": response, "type": "narration", "tag": []})
        
        return output_list

    def save_results(self, output_list, save_dir="result/genre_transformation"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(f"{save_dir}/narration.json", "w") as f:
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

    transformer = NarrativeTransformer()
    output_list = transformer.transform_dataset(dataset)
    transformer.save_results(output_list)         