from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os 
load_dotenv()


@tool
def add_numbers(a:int, b:int):
    """
    It will return the sum of the two numbers.
    Args:
        a: Number one
        b: Number two
    """
    return a+b

@tool
def multiply_numbers(a:int, b:int):
    """
    It will return the sum of the two numbers.
    Args:
        a: Number one
        b: Number two
    """
    return a*b

# res = add_numbers.invoke({'a':3,'b':4})
# print(res)

llm = ChatGroq(model="qwen/qwen3-32b")

agent= create_agent(
    model=llm,
    tools=[add_numbers, multiply_numbers],
    system_prompt='You are a math teacher, and always use tool for calculations.'
)

response = agent.invoke({'messages':[{"role":"user", "content":"What is 2+2*100?"}]})

print(response['messages'])