from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


search = GoogleSerperAPIWrapper()
memory = InMemorySaver()

model = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt='You are an agent, can search any question on google.',
)

while True:
    query = input('User: ')
    if query.lower() == 'quite':
        print('Good Bye')
        break 

    response = agent.invoke(
                    {'messages':[{'role':'user', 'content':query}]},
                    {'configurable': {'thread_id': 'abc123'}},
                )

    print('AI: ', response['messages'][-1].content)
