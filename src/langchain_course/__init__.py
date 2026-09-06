from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 10 job postings for an AI Engineer using LangChain in Hyderabad, "
                                                            "India and list their details in a table format with the following columns: Job Title, Company Name, Location, Salary, and Job Description.")})
    print(result)

if __name__ == "__main__":
    main()