'''
# prompt05.py <this is a sequential design of langraph
# This code demonstrates how to use LangChain with Groq's LLM to process text by classifying it, extracting entities, and summarizing it.
# It includes a `TextProcessor` class that defines a state graph for processing text, with nodes for classification, entity extraction, and summarization.
# The code uses the `ChatGroq` class to interact with the Groq model and the `PromptTemplate` class to create prompts for each processing step.
# The `process` method of the `TextProcessor` class takes a text input, invokes the state graph, and prints the results of classification, entity extraction, and summarization.
# It also includes a sample text to demonstrate the functionality.
'''


import os
import gradio as gr
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

class TextProcessor:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.graph = self.create_graph()

    def classification_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Classify the following text into one of the categories: News, Blog, Research, or Other. Do not write anything else except one of these words.\n\nText:{text}\n\nCategory:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        classification = self.llm.invoke([message]).content.strip()
        return {"classification": classification}
    
    def entity_extraction_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        entities = self.llm.invoke([message]).content.strip().split(", ")
        return {"entities": entities}
        
    def summarization_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        summary = self.llm.invoke([message]).content.strip()
        return {"summary": summary}

    def create_graph(self):
        graph = StateGraph(State)
        graph.add_node("classification_node", self.classification_node)
        graph.add_node("entity_extraction", self.entity_extraction_node)
        graph.add_node("summarization", self.summarization_node)
        
        graph.set_entry_point("classification_node")
        graph.add_edge("classification_node", "entity_extraction")
        graph.add_edge("entity_extraction", "summarization")
        graph.add_edge("summarization", END)
        return graph.compile()
     
    def process(self, text):
        state_input = {"text": text}
        result = self.graph.invoke(state_input)
        print("Classification:", result["classification"])
        print("\nEntities:", result["entities"])
        print("\nSummary:", result["summary"])
        return result

if __name__ == "__main__":
    processor = TextProcessor()
    sample_text = """
    The World Health Organization (WHO) has published a new report on climate change and its impact on global health systems. According to the findings, rising temperatures are leading to increased spread of infectious diseases, particularly in tropical regions.

    The report also highlights that extreme weather events such as floods, hurricanes, and droughts are causing significant disruptions to healthcare infrastructure in vulnerable communities. Researchers from Oxford University and the CDC collaborated on the study, which analyzed data from 45 countries over a period of 15 years.

    WHO Director-General emphasized that healthcare systems need to become more resilient and adaptable to climate-related challenges, recommending that countries allocate at least 5% of their health budgets to climate adaptation measures by 2030.
    """
    result = processor.process(sample_text)