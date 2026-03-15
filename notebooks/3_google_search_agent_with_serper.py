from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv()


search = GoogleSerperAPIWrapper()

model = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(
    model=model,
    tools=[search.run],
    system_prompt='You are an agent, can search any question on google.',
    checkpointer=InMemorySaver()
)

question1 = "Who is the prime minister of India?"
question2 = "What is his age ?"

response = agent.invoke(
    {'messages':[{'role':'user', 'content':question1}]},
    {"configurable": {"thread_id": "abc123"}},
)

print('res1: ', response['messages'][-1].content)

response = agent.invoke(
    {'messages':[{'role':'user', 'content':question2}]},
    {"configurable": {"thread_id": "abc123"}},
)

print('res2: ', response['messages'][-1].content)