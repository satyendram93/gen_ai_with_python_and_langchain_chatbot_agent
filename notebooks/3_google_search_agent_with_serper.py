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











# ##########Using Langgrap
# from langchain_community.utilities import GoogleSerperAPIWrapper
# from langchain_groq import ChatGroq
# from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent  # Updated import
# from langgraph.checkpoint.memory import InMemorySaver
# from dotenv import load_dotenv
# import os

# load_dotenv()

# search = GoogleSerperAPIWrapper()  # Needs SERPER_API_KEY
# model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# agent = create_agent(model=model, tools=[search.run], checkpointer=InMemorySaver())
# # agent = create_agent(
# #     model=model,
# #     tools=[search.run],
# #     system_prompt='You are an agent, can search any question on google.',
# #     checkpointer=InMemorySaver()
# # )
# config = {"configurable": {"thread_id": "test1"}}

# # Test 1
# resp1 = agent.invoke({"messages": [{"role": "user", "content": "Who is the prime minister of India?"}]}, config)
# print("1:", resp1['messages'][-1].content)  # Narendra Modi [web:23]

# # Test 2 (memory active)
# # resp2 = agent.invoke({"messages": [{"role": "user", "content": "What is his age?"}]}, config)
# # print("2:", resp2['messages'][-1].content)  # ~75 [web:21]