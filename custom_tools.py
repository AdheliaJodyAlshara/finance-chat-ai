import tempfile
import shutil
import re
import pandas as pd
from typing import List, Tuple
from langchain.agents import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
# from typing import Annotated
from config import df, llm

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

#def get data
def get_data(user_question : str) -> Tuple[pd.DataFrame, str]:
    prompt = f"""I have a STAR AM's pandas DataFrame 'df' with the following schema:
        - year: The year field indicates the fiscal year for which the trial balance data is being pulled. 
        - month: The month field specifies the particular month for which the trial balance numbers are being reported. This value is in integer data type.
        - business_unit : the business unit's name ("STARAM").
        - business_metric : this specifies the sub-categories into which the chart of accounts are mapped. the mapping relationship is one to many where one business_metrics value contains many chart of accounts from description column. ('manpower cost','technology cost','selling & marketing expenses','general and administrative expenses','revenue - operational', 'direct cost').
        - description : refers to the section of the trial balance report that lists the names of the accounts from the chart of accounts. This column provides a clear and organized way to identify each account involved in the financial reporting process, making it easier to understand the source of the financial data.
        - mtd_value: refers to a section of the trial balance report that displays the cumulative balances of all accounts from the first day of the month up to the last day within that month.
        - ytd_value: refers to a section of the trial balance report that displays the cumulative balances of all accounts from the beginning of the fiscal year up to a specific month. 

        EBITDA is calculated based on below formula in business_metrics Value: EBITDA = Total Revenue - Direct Cost - Manpower Cost - Selling & Marketing Expense - General and Administrative Expense - Technology Cost.

    Write code to filter my DataFrame based on the user's question. Return Markdown for a Python code snippet and nothing else. Always store the final filtered df in `filtered_df` variable and no need to show/print the `filtered_df` variable in the end of the code. Please always show the column names in the result of `filtered_df`. Always include 'year' and 'month' columns in the result of `filtered_df`. You must always do aggregation to get distinct data.

    User Question: {user_question}"""

    ai_msg = llm.invoke(input=prompt)

    code = ai_msg.content.replace("```python", "").replace("```", "").strip()
    # print("Extracted Python Code:\n", code)
    print(code)

    # Create a dictionary to capture local variables
    local_vars = {}
    
    # Execute the generated code within the local_vars dictionary
    exec(code, globals(), local_vars)

    # Retrieve filtered_df from local_vars
    filtered_df = local_vars.get('filtered_df')
    if filtered_df is None:
        raise ValueError("filtered_df was not created by the executed code.")
    
    # filtered_df_str = filtered_df.to_markdown(index=False)
    filtered_df_str = filtered_df.to_string(index=False)

    # print(len(filtered_df))
    # return code, format_scientific_notation(filtered_df_str)
    return filtered_df, filtered_df_str

# def chart_generator(user_question : Annotated[str, "The original user question without paraphrasing for creating chart"]) -> str:
def chart_generator(user_question : str) -> str:
    user_question += """\n\nAlways show in numeric number instead in scientific number format, Add the data label, and Please always order in ascending"""
    filtered_df, _ = get_data(user_question=user_question)
    sdf = SmartDataframe(filtered_df, config={"llm": llm, "save_charts": True})
    chart_path = sdf.chat(user_question)

    # file_ = open(chart_path, "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()

    response = f"""The chart has been generated.
    
    Please strictly add this HTML command in your response to show the chart image to the user:
    <img src="{chart_path}" alt="chart image">"""
    return response


def initialize_tools() -> List[Tool]:
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
            description="useful to create chart based on user question. Please address the entire user question directly without any paraphrasing."
        )
    ]

    return tools


