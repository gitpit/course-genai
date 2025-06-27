'''
prompt03.py
# generate multiple reasoning paths
# This code demonstrates how to use LangChain with Groq's LLM to generate multiple reasoning paths for a problem,
# aggregate the results, and perform a self-consistency check.
# 1. Generate multiple reasoning paths for the problem.
# 2. Aggregate the results from these paths.
# 3. Perform a self-consistency check on the aggregated result.
'''
# Self consistency

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
from collections import Counter

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3) 

def generate_multiple_paths(problem, num_paths=3):
    """
    Generate multiple reasoning paths for a given problem.
    
    Args:
    problem (str): The problem statement.
    num_paths (int): Number of reasoning paths to generate.
    
    Returns:
    list: A list of generated reasoning paths.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "path_number"],
        template="""Solve the following problem using a unique approach. This is reasoning path {path_number}.
        Problem: {problem}
        Reasoning path {path_number}:"""
    )

    paths = []
    for i in range(num_paths):
        chain = prompt_template | llm
        response = chain.invoke({"problem": problem, "path_number": i+1}).content
        paths.append(response)
    
    return paths

# problem = "A ball is thrown upwards with an initial velocity of 20 m/s. How high will it go?"
# paths = generate_multiple_paths(problem)


def aggregate_results(paths):
    """
    Aggregate results from multiple reasoning paths.
    
    Args:
    paths (list): List of reasoning paths.
    
    Returns:
    str: The most consistent answer.
    """
    prompt_template = PromptTemplate(
        input_variables=["paths"],
        template="""Analyze the following reasoning paths and determine the most consistent answer. If there are discrepancies, explain why and provide the most likely correct answer.
        Reasoning paths:
        {paths}
        
        Most consistent answer:"""
    )

    chain = prompt_template | llm
    response = chain.invoke({"paths": "\n".join(paths)}).content
    return response

# aggregated_result = aggregate_results(paths)
# print("Aggregated Result:\n", aggregated_result)

def self_consistency_check(problem, aggregated_result):
    """
    Perform a self-consistency check on the aggregated result.
    
    Args:
    problem (str): The original problem statement.
    aggregated_result (str): The aggregated result to check.
    
    Returns:
    str: An evaluation of the result's consistency and reliability.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "result"],
        template="""Evaluate the consistency and reliability of the following result for the given problem.
        Problem: {problem}
        Result: {result}
        
        Evaluation (consider factors like logical consistency, adherence to known facts, and potential biases):"""
    )

    chain = prompt_template | llm
    response = chain.invoke({"problem": problem, "result": aggregated_result}).content
    return response


# consistency_evaluation = self_consistency_check(problem, aggregated_result)
# print("Self-Consistency Evaluation:\n", consistency_evaluation)


def solve_problem(problem):
    """
    Solve a problem using multiple reasoning paths, aggregation, and self-consistency check.
    
    Args:
    problem (str): The problem statement.
    
    Returns:
    tuple: (aggregated_result, consistency_evaluation)
    """
    paths = generate_multiple_paths(problem)
    aggregated_result = aggregate_results(paths)
    consistency_evaluation = self_consistency_check(problem, aggregated_result)
    return aggregated_result, consistency_evaluation


problems = [
    "What is the capital of France?",
    "Explain the concept of supply and demand in economics.",
    "If a train travels at 60 km/h, how long will it take to cover 180 km?"
]

for problem in problems:
    print(f"Problem: {problem}")
    result, evaluation = solve_problem(problem)
    print("Aggregated Result:\n", result)
    print("\nConsistency Evaluation:\n", evaluation)
    print("\n" + "-"*50 + "\n")
