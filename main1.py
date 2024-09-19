import streamlit as st
import pandas as pd
import os
import re
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from custom_tools import initialize_tools, data_input
from callbacks import StreamHandler, stream_data
from config import llm

# Set the page layout to wide
st.set_page_config(layout="wide")

# url = os.getenv("CSV_URL")
# file_id=url.split('/')[-2]
# dwn_url='https://drive.google.com/uc?id=' + file_id
# data_input = pd.read_csv(dwn_url)


# Initialize tools
tools = initialize_tools()

# Define the prefix and suffix for the prompt
prefix = '''You are the best Data Analyst that can help me Financial Analyst in summarizing findings, drawing insights, and making recommendations from a financial data.
You should not make things up and only answer all questions related to the data and finance.

STAR AM Company Background:
PT Surya Timur Alam Raya Asset Management (STAR AM) is a company engaged in asset management in Indonesia. The company has a primary focus on investment fund management and investment advisory, both for individuals and business entities. STAR AM is dedicated to always providing quality investment solutions, which allows STAR AM to assert its position as one of the investment managers with a high level of development in the mutual fund industry in Indonesia.

- Manpower Cost: This business metric includes all expenses related to employee compensation. It encompasses salaries, wages, benefits (like health insurance and retirement contributions), bonuses, and other related costs. This also includes payroll taxes, training, and any other expenses directly tied to staffing.
- Selling & Marketing Expenses: These are costs incurred to promote and sell products or services. They include advertising, promotional activities, and other related activities aimed at increasing sales and market presence.
- General and Administrative Expenses: These are the overhead costs necessary for the overall operation of the business but not directly tied to production or sales. This business metric includes expenses like office rent, utilities, office supplies, legal and accounting fees, insurance, and executive salaries, and other related general & admin activities. These are essential to keeping the company running smoothly but do not contribute directly to revenue generation.
- Technology Cost: This business metric encompasses expenses related to the acquisition, maintenance, and operation of technology systems. It includes costs for software, hardware, IT support, data storage, cloud services, cybersecurity, and any other technology-related expenses that enable the business to operate efficiently and securely.
- Revenue - Operational: refers to the income generated from the company's core business activities related to managing client assets. This subcategory of operational revenue typically includes Management Fees: Fees charged to clients for managing their portfolios or investment funds. These fees are usually calculated as a percentage of the assets under management (AUM).

Here the steps for you to summarize and give insight about the STAR AM finance report data:
- Step 1 : Step by step analyze provided finance data trends in mtd_value or ytd_value column as user request within years over months from each business metric and description of chart of account. You don't need to use the tools for this step.
- Step 2 : Enhance your analysis from Step 1 by gathering additional insights from the internet to strengthen your summary. You are limited to a maximum of three internet searches. If you find the information you need before reaching three searches, you can proceed to the next step without completing all three searches. But if the question involves creating a chart, you can use the Chart Generator tool by passing the user question without paraphrasing for creating chart.
- Step 3 : Summarize the findings and provide insights based on your analysis from Step 1 and the additional information from Step 2. Ensure your final answer integrates the data trends with insights from the internet searches or chart generator.
- Step 4 : In the final output, You should include all reference data & links to back up your research; You should include all reference data.

If the question is a follow-up question or does not relate to the provided finance data, then here the steps for you:
- Step 1 : Get the information from the internet to get answer from the user question. REMEMBER YOU ARE ONLY PERMITTED TO SEARCH FROM THE INTERNET 3 TIMES OR LESS! If you feel enough with your research from the internet less than 3 times, you can immediately move on to the next step.
- Step 2 : From step 1, provide the final answer. In the final output, You should include all reference data & links to back up your research; You should include all reference data.
'''

data_string = f'''Finance Report Data of STAR AM Business Unit: 
```
{data_input.to_string()}
```

'''

suffix = data_string + '''Your past conversation with human:
```
{chat_history}
```

Begin!

Question: {human_input}
Thought: {agent_scratchpad}
'''

# Define the format instructions for the agent's output
format_instructions = """Strictly use the following format and it must be in consecutive order without any punctuation:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be one of these tool names only without the parameters [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated only 3 times)
Thought: I now know the final answer
Final Answer: If the question directly involves analyzing the finance report data provided. 

Provide your final answer using the following output format for each business metrics:
<Business Metrics>
- Summarization: <Your Summarization as a paragraph> 
- Insight: <Your Insight as a paragraph>


If the question is a follow-up or does not relate to the provided finance data:
Final Answer: <Directly provide the summarized answer without the detailed format>
"""

# Create the prompt for the ZeroShotAgent
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    format_instructions=format_instructions,
    input_variables=["chat_history", "human_input", "agent_scratchpad"],

)

memory = ConversationBufferMemory(memory_key="chat_history")

# stream_handler = StreamHandler(st.empty())

# Initialize the language model chain with a chat model
llm_chain = LLMChain(
    llm=llm,
    #     ChatOpenAI(
    #     model_name="gpt-4o",
    #     temperature=0
    #     # streaming=True,
    #     # callbacks=[stream_handler]
    # ),
    prompt=prompt
)

# Create the ZeroShotAgent with the language model chain
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# Initialize the AgentExecutor with the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)

if __name__ == "__main__":
    # Setup the Streamlit interface
    st.title('Q&A AI Finance')

    # Initialize session state for maintaining conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "<img" in message['content']:
                image_path = re.search(r'src="([^"]*)"', message['content']).group(1)
                new_message = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', message['content'])
                st.markdown(new_message)
                st.image(image_path)
            else:
                st.markdown(message['content'])

    # User inputs their question
    user_question = st.chat_input("Enter your question about the finance data...")

    # Button to process the question
    if user_question:
        # Append user's question to the session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run the agent with the formulated prompt
        with st.spinner('Processing...'):
            try:
                response = agent_executor.run(human_input=user_question)
            except:
                response = "I'm sorry I can't process your query right now. Please try again."

        # Append AI response to the session state
        try:
            with st.chat_message("assistant"):
                if "<img" in response:
                    image_path = re.search(r'src="([^"]*)"', response).group(1)
                    new_response = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', response)
                    st.write_stream(stream_data(new_response))
                    st.image(image_path)
                else:
                    st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except:
            with st.chat_message("assistant"):
                response = "I'm sorry I can't process your query right now. Please try again."
                st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
