from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
# Get API_KEY
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load and test GEMINI
gemini = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    api_key = GEMINI_API_KEY
)
#response = gemini.invoke("How does Donald Trump view Malaysia?")
#print(response)


"-------------------------------------"

class ResearchResponse(BaseModel):
  topic:str
  summary:str
  sources:list[str]
  tools_used:list[str]
  think:str
  action:list[str]
  observation:str

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt =  ChatPromptTemplate.from_messages(
    [
     (
         "system",
         """
         You are a research assistant that will help generate a research paper.
         Answer the user query and use necessary tools. Think step by step. Explain your reasoning.
         State and list your possible actions. Reflect on your observations after executing tools.  
         Wrap the output in this format and provide no other text \n{format_instructions}\n
         """,
     ),
     ("placeholder","{chat_history}"),
     ("human","{query}"),
     ("placeholder","{agent_scratchpad}"),   
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool,wiki_tool,save_tool]
agent = create_tool_calling_agent(
    llm=gemini,
    prompt = prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

query = input("What can I help you research about?")
raw_response = agent_executor.invoke({"query":query})

#print(raw_response)

"""
# Debug
def get_dict_keys(dict_x):
  if type(dict_x) == dict:  
    for key in dict_x:
      print(key)
  else:
    print("variable is not of type dictionary")

#get_dict_keys(raw_response)
"""

try:
  structured_response = parser.parse(raw_response.get("output"))
  print(structured_response)
  #print(structured_response.topic)
except Exception as e:
  print("Error parsing response",e,"Raw response",raw_response)



"""
#Debug
output_text = "This is the content you want to save."

# Save to a text file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(output_text)

print("File saved as output.txt")"
"""
