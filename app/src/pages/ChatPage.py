import streamlit as st
from streamlit_chat import message

from ..services.chat import ChatCompletion
from ..services.data_processor import DataProcessor
from ..services.database import CosmosDB
from ..services.embeddings import EmbeddingsGenerator

class ChatPage:
    chat_history = []
    cosmos_db = CosmosDB()
    embeddings_generator = EmbeddingsGenerator()
    chat_completion = ChatCompletion(cosmos_db, embeddings_generator)
    data_processor = DataProcessor(cosmos_db, embeddings_generator)

    def __init__(self):
        pass

    def write(self):
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []

        text_input_style = f"""
        <style>
            .stTextInput {{
            z-index: 1;
            position: fixed;
            bottom: 3rem;
            }}
        </style>
        """
        st.markdown(text_input_style, unsafe_allow_html=True)

        user_input = st.text_input("Prompt here: ", key="input", label_visibility="collapsed")

        if user_input:
            output = self.chat_completion.chat_completion(user_input)
            # output = "PLACEHOLDER OUTPUT PLACEHOLDER OUTPUT PLACEHOLDER OUTPUTPLACEHOLDER OUTPUTPLACEHOLDER OUTPUTPLACEHOLDER OUTPUT"

            # Append the user input and response to session state
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        # Display the chat history in reverse order
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                # user and bot messages are displayed
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

