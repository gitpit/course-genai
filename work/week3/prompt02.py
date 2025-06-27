'''
#prompt02.py
# Chain of Thought and Logical Reasoning with LangChain and Groq
# This code demonstrates how to use LangChain with Groq's LLM to solve mathematical problems
# and logical reasoning puzzles.
# It includes a standard prompt for direct answers, a chain of thought (cot) prompt for step-by-step reasoning,
# and an advanced cot prompt for complex calculations.
'''
# cot = chain of thought

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 


standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}."
)

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question step by step: {question}"
)

standard_chain = standard_prompt | llm #what is this pipe operator?
# The pipe operator (`|`) is used to create a chain of operations in LangChain,
# where the output of one operation (in this case, the prompt) is passed as input
standard_prompt_response = standard_chain.invoke("How the F16 fighter jet works?").content
print(standard_prompt_response)

cot_chain = cot_prompt | llm
question = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

cot_chain_response = cot_chain.invoke(question).content
print(cot_chain_response)

# standard_response = standard_chain.invoke(question).content
# cot_response = cot_chain.invoke(question).content

advanced_cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Solve the following problem step by step. For each step:
1. State what you're going to calculate
2. Write the formula you'll use (if applicable)
3. Perform the calculation
4. Explain the result

Question: {question}

Solution:"""
)
advanced_cot_chain = advanced_cot_prompt | llm

complex_question = "A car travels 150 km at 60 km/h, then another 100 km at 50 km/h. What is the average speed for the entire journey?"
advanced_cot_response = advanced_cot_chain.invoke(complex_question).content
exit()

logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""Analyze the following logical puzzle thoroughly. Follow these steps in your analysis:

List the Facts:

Summarize all the given information and statements clearly.
Identify all the characters or elements involved.
Identify Possible Roles or Conditions:

Determine all possible roles, behaviors, or states applicable to the characters or elements (e.g., truth-teller, liar, alternator).
Note the Constraints:

Outline any rules, constraints, or relationships specified in the puzzle.
Generate Possible Scenarios:

Systematically consider all possible combinations of roles or conditions for the characters or elements.
Ensure that all permutations are accounted for.
Test Each Scenario:

For each possible scenario:
Assume the roles or conditions you've assigned.
Analyze each statement based on these assumptions.
Check for consistency or contradictions within the scenario.
Eliminate Inconsistent Scenarios:

Discard any scenarios that lead to contradictions or violate the constraints.
Keep track of the reasoning for eliminating each scenario.
Conclude the Solution:

Identify the scenario(s) that remain consistent after testing.
Summarize the findings.
Provide a Clear Answer:

State definitively the role or condition of each character or element.
Explain why this is the only possible solution based on your analysis.
Scenario:

{scenario}

Analysis:""")