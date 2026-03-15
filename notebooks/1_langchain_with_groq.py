from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os 
load_dotenv()

key = os.getenv('GROQ_API_KEY')


llm = ChatGroq(model='qwen/qwen3-32b', streaming=True)

question="Can you explain me GEN-AI?"

response = llm.stream(question)
for i in response:
    print(i.content, end="")