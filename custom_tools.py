import os
import pandas as pd
from langchain.agents import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from langchain_openai import OpenAI
# from main import data_input

url = os.getenv("CSV_URL")
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
data_input = pd.read_csv(dwn_url)


# def searchFromInternet(query: str):
#     retriever = TavilySearchAPIRetriever(k=10, search_depth="advanced")
#     result = retriever.invoke(query)
#     return result


# def searchFromInternetUsingGoogle(query: str):
#     # https://python.langchain.com/docs/integrations/tools/google_serper
#     search = GoogleSerperAPIWrapper(type="news")
#     results = search.results(query)
#     return results

def default_tools():
    answer = """
    """
    return answer

def chart_generator(query : str):
    langchain_llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    df = SmartDataframe(data_input, config={"llm": langchain_llm})

    response = df.chat(query)

    return response



def initialize_tools():
    search_tavily = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search_tavily)

    tools = [
        Tool(
            name="searchFromInternet",
            func=tavily_tool.run,
            description="useful to get additional information from internet."
        ),

        Tool(
            name="ChartGenerator",
            func=chart_generator,
            description="useful to get chart from user question."
        )
    ]

    return tools


