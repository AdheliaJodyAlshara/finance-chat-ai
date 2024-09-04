import streamlit as st
import re

from agent import agent_with_chat_history
from callbacks import stream_data

if __name__ == "__main__":
    # Set the page layout to wide
    st.set_page_config(layout="wide")

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
                response = agent_with_chat_history.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": "1"}},
                )
                assistant_answer = response.get("output")
            except:
                assistant_answer = "I'm sorry I can't process your query right now. Please try again."
        
        # Append AI response to the session state
        try:
            with st.chat_message("assistant"):
                if "<img" in assistant_answer:
                    image_path = re.search(r'src="([^"]*)"', assistant_answer).group(1)
                    new_assistant_answer = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', assistant_answer)
                    st.write_stream(stream_data(new_assistant_answer))
                    st.image(image_path)
                else:
                    st.write_stream(stream_data(assistant_answer))
            st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
        except:
            with st.chat_message("assistant"):
                assistant_answer = "I'm sorry I can't process your query right now. Please try again."
                st.write_stream(stream_data(assistant_answer))
            st.session_state.messages.append({"role": "assistant", "content": assistant_answer})