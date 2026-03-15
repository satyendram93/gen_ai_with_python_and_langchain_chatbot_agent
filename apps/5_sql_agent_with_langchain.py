###Requirements: db,llm,tools,create_agent, system_prompt

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit   


db = SQLDatabase.from_uri("sqlite:///my_task.db")

db.run("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
# print('DB created successfully.')

model = ChatGroq(model='openai/gpt-oss-20b')
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
memory = InMemorySaver()

system_prompt = """
            You are a powerful Task Management SQL Agent for the `my_task` database. You handle full CRUD operations on the `tasks` table.

            TASK RULES:
            1. LIMIT SELECT query to 10 results max with ORDER BY created_at DESC
            2. after CREATE/UPDATE/DELECT, confirm with SELECT query
            3. if the user results a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.

            CRUD OPERATIONS:
            CREATE: INSERT INTO tasks (title, description, status)
            READ: SELECT * FROM tasks WHERE ... LIMIT 10
            UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
            DELETE: DELETE FROM tasks WHERE id=? OR title=?

            Table Schema: id, title, description, status(pending/progress/completed), created_at.

        """

agent = create_agent(
    model = model,
    tools = tools, 
    checkpointer = memory,
    system_prompt = system_prompt
)

while True:
    query = input('User: ')

    response = agent.invoke(
                        {'messages':[{'role':'user', 'content':query}]},
                            {'configurable':{'thread_id':'1'}}
                    )
    
    result = response['messages'][-1].content 

    print("Ai: ", result)