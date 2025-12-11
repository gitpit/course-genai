'''
prompt01.py
# This code demonstrates how to use the LangChain library with Groq's ChatGroq model for few-shot learning tasks --`few_shot_sentiment_classification`,
#  `multi_task_few_shot` and `in_context_learning`.
# It includes functions for sentiment classification, multi-task learning, and in-context learning with examples.
# The `multi_task_few_shot` function performs a specified task on the input text, such as language detection or sentiment analysis.
# The `in_context_learning` function allows for in-context learning by providing a task description, examples, and input text.
# The code uses the `PromptTemplate` class to create prompts for each task and invokes the ChatGroq model to get the results.

**Important Notes:
 - It works with venv3.11
 - Not working -- from langchain.prompts import PromptTemplate; use below
 - from langchain_core.prompts import PromptTemplate which works
 '''

from dotenv import load_dotenv
from langchain_groq import ChatGroq # ChatGroq is a class from the LangChain library that interfaces with Groq's LLMs, allowing users to generate text based on prompts.
#from langchain.prompts import PromptTemplate # PromptTemplate is a class from the LangChain library that helps in creating structured prompts for language models.
from langchain_core.prompts import PromptTemplate

load_dotenv()

'''
Below Creates a client connection to Groq's cloud API
Specifies which model you want to use on Groq's servers
Sets the temperature parameter for generation
'''
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def few_shot_sentiment_classification(input_text):
    ''' The `few_shot_sentiment_classification` function classifies the sentiment of a given text using a few-shot prompt.
    '''
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text"],
        template=""" #what is this three quotes in python?
        Classify the sentiment as Positive, Negative, or Neutral.

        Examples:
        Text: I love this product! It's amazing.
        Sentiment: Positive

        Text: This movie was terrible. I hated it.
        Sentiment: Negative

        Text: The weather today is okay.
        Sentiment: Neutral

        Now, classify the following:
        Text: {input_text}
        Sentiment:
        """
    )

    chain = few_shot_prompt | llm
    result = chain.invoke(input_text).content

    result = result.strip()
    if ':' in result:
        result = result.split(':')[1].strip()

    return result

def multi_task_few_shot(input_text, task):
    '''
    The `multi_task_few_shot` function performs a specified task on the input text using a few-shot prompt. It takes 
    input as `input_text` and `task
    '''
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text", "task"],
        template="""
        Perform the specified task on the given text.

        Examples:
        Text: I love this product! It's amazing.
        Task: sentiment
        Result: Positive

        Text: Bonjour, comment allez-vous?
        Task: language
        Result: French

        Now, perform the following task:
        Text: {input_text}
        Task: {task}
        Result:
        """
    )

    chain = few_shot_prompt | llm
    return chain.invoke({"input_text": input_text, "task": task}).content

def in_context_learning(task_description, examples, input_text):
    example_text = "".join([f"Input: {e['input']}\nOutput: {e['output']}\n\n" for e in examples])

    in_context_prompt = PromptTemplate(
        input_variables=["task_description", "examples", "input_text"],
        template="""
        Task: {task_description}

        Examples:
        {examples}

        Now, perform the task on the following input:
        Input: {input_text}
        Output:
        """
    )

    chain = in_context_prompt | llm
    return chain.invoke({"task_description": task_description, "examples": example_text, "input_text": input_text}).content

test_text = "I can't believe how great this new restaurant is!"
result = few_shot_sentiment_classification(test_text)
print(result)

task_desc = "Convert the given text to pig latin."
examples = [
    {"input": "hello", "output": "ellohay"},
    {"input": "apple", "output": "appleay"}
]
test_input = "python"
result = in_context_learning(task_desc, examples, test_input)
print(result)
multi_task_few_shot_input = "Bonjour, comment allez-vous?"
multi_task_few_shot_task = "language"
result = multi_task_few_shot(multi_task_few_shot_input, multi_task_few_shot_task)
print(result)

print("It works!!")