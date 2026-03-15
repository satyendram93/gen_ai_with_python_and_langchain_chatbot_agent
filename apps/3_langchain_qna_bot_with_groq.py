### Requirements_for_this_app = [ LLM, Tools - Google search tool, Agent, Memory, streaming, Web Interface]

from dotenv import load_dotenv
import os

load_dotenv()  # Call after imports, before using env vars

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


#This llm is the model of Chat Groq
llm = ChatGroq(model='openai/gpt-oss-20b')

#search is the objects of google serper API Wrapper
search = GoogleSerperAPIWrapper()

#this tools used to create google search agent
tools = [search.run]

if 'memory' not in st.session_state:
    #This session is used to hold the memory
    st.session_state.memory = InMemorySaver()

    #It will store the history of user and ai
    st.session_state.history = []

#Now ready to create agent using llm with google search serper
agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=st.session_state.memory,
        system_prompt='You are an amazing agent and can search on google as well.',
    )

##Building Web Interface
st.subheader('QuickAnswer - Answers at the speed of thought')

#update the history of user and ai on each message
for message in st.session_state.history:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

query = st.chat_input('Ask anythings ?')

if query:
    st.chat_message('user').markdown(query)

    #update the history of user
    st.session_state.history.append({'role':'user', 'content':query})

    #now agent will receive and user's input and return response
    response = agent.invoke(
                        {'messages':[{'role':'user', 'content':query}]},
                        {'configurable':{'thread_id':'abc123'}},
                    )
    
    #answers will accepts the response
    answers = response['messages'][-1].content 

    st.chat_message('ai').markdown(answers)

    #update the history ai
    st.session_state.history.append({'role':'ai', 'content':answers})

    

