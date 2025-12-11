'''
# prompt06.py prompt routing with langgraph
# This method creates a state graph for processing text, with nodes for classification, entity extraction, and summarization.

**Important Notes:
 - It works with venv3.11
 - Not working -- from langchain.prompts import PromptTemplate; use below
 - from langchain_core.prompts import PromptTemplate which works
 - not working with llama model - "llama3-70b-8192" which is deprecated
https://docs.langchain.com/oss/python/migrate/langchain-v1
'''

from typing import Literal, Optional, TypedDict, Dict, List
from pydantic import BaseModel, Field
import os
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

load_dotenv()

class QueryInput(BaseModel):
    query: str = Field(description="The user's original query")


class ClassificationResult(BaseModel): #what is this?
    """Result of the query classification."""
    category: Literal["physics", "math", "nature", "general"] = Field(  
        description="The classified category of the query"
    )
    confidence: float = Field(
        description="Confidence score for the classification", 
        ge=0.0, 
        le=1.0
    )


class Response(BaseModel):
    """Response to the user's query."""
    answer: str = Field(description="The answer to the user's query")
    category_used: Literal["physics", "math", "nature", "general"] = Field(
        description="The category used to generate the answer"
    )


class State(TypedDict):

    input: QueryInput
    classification: Optional[ClassificationResult]
    response: Optional[Response]


class QueryRouter:

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model_name=model_name)
    
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier. Given a query, determine which of the following categories it belongs to: 
            physics, math, nature, or general. Respond with ONLY the category name followed by a confidence score between 0 and 1.
            
            Example format: 
            physics:0.85
            
            Make your classification as accurate as possible."""),
            ("human", "{query}")
        ])
        
        self.physics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a physics expert. Provide accurate, detailed answers to physics questions.
            Focus on explaining physical laws, theories, and phenomena in a clear and educational manner.
            Use scientific terminology appropriately and include relevant formulas when needed."""),
            ("human", "{query}")
        ])
        
        self.math_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematics expert. Provide accurate, detailed answers to math questions.
            Focus on explaining mathematical concepts, proofs, and problem-solving techniques.
            Show step-by-step solutions when appropriate and use precise mathematical notation."""),
            ("human", "{query}")
        ])
        
        self.nature_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a natural sciences expert. Provide accurate, detailed answers about the natural world.
            Focus on explaining biological systems, ecological relationships, geological processes, and other natural phenomena.
            Use scientific terminology appropriately and emphasize the interconnectedness of natural systems."""),
            ("human", "{query}")
        ])
        
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant. Provide helpful, accurate information on general topics.
            Aim to be informative and educational while keeping explanations accessible to a general audience.
            Draw from a broad knowledge base to give well-rounded answers."""),
            ("human", "{query}")
        ])
    
        self.setup_graph()

    def classify_query(self, state: State) -> State:
        query = state["input"].query
        chain = self.classification_prompt | self.llm
        result = chain.invoke({"query": query})
        
        result_text = result.content.strip()
        category, confidence_str = result_text.split(":")
        category = category.strip().lower()
        confidence = float(confidence_str)
        
        return {
            **state,
            "classification": ClassificationResult(category=category, confidence=confidence)
        }
    
    def route_to_physics(self, state: State) -> State:
        query = state["input"].query
        chain = self.physics_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="physics"
            )
        }
    
    def route_to_math(self, state: State) -> State:
        query = state["input"].query
        chain = self.math_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="math"
            )
        }
    
    def route_to_nature(self, state: State) -> State:
        query = state["input"].query
        chain = self.nature_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="nature"
            )
        }
    
    def route_to_general(self, state: State) -> State:
        query = state["input"].query
        chain = self.general_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="general"
            )
        }
    
    def decide_route(self, state: State) -> Literal["physics", "math", "nature", "general"]:
        return state["classification"].category
    
    def setup_graph(self):
        self.graph = StateGraph(State)
        
        self.graph.add_node("classification_node", self.classify_query)
        self.graph.add_node("physics_node", self.route_to_physics)
        self.graph.add_node("math_node", self.route_to_math)
        self.graph.add_node("nature_node", self.route_to_nature)
        self.graph.add_node("general_node", self.route_to_general)
        
        self.graph.add_edge(START, "classification_node")
        self.graph.add_conditional_edges(
            "classification_node",
            self.decide_route,
            {
                "physics": "physics_node",
                "math": "math_node",
                "nature": "nature_node",
                "general": "general_node"
            }
        )
        
        self.graph.add_edge("physics_node", END)
        self.graph.add_edge("math_node", END)
        self.graph.add_edge("nature_node", END)
        self.graph.add_edge("general_node", END)
        
        self.graph_instance = self.graph.compile()
    
    def process_query(self, query: str) -> Dict:
        initial_state = {"input": QueryInput(query=query), "classification": None, "response": None}
        result = self.graph_instance.invoke(initial_state)
        return {
            "query": query,
            "category": result["classification"].category,
            "confidence": result["classification"].confidence,
            "answer": result["response"].answer
        }


if __name__ == "__main__":    
    #router = QueryRouter(model_name="llama3-70b-8192")
    router = QueryRouter(model_name="llama-3.3-70b-versatile")
    response = router.process_query("What is the theory of relativity")
    print("Query:", response["query"])
    print("It works!!")