import asyncio
import logging

import streamlit as st
from src.services.chat import ChatCompletion
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from streamlit_chat import message

logger = logging.getLogger("papermaid")


class ChatPage:
    """
    Manages the chat interface and interactions for the PaperMaid application.

    This class handles the Streamlit-based user interface for the chat application, including:
    - Initializing necessary services and components
    - Managing file uploads and processing
    - Handling user input and generating responses
    - Displaying chat history and messages
    """

    chat_history = []
    cosmos_db = CosmosDB()
    embeddings_generator = LangchainEmbeddingsGenerator()
    data_processor = DataProcessor(cosmos_db, embeddings_generator)
    chat_completion = ChatCompletion(cosmos_db, embeddings_generator, data_processor)

    def __init__(self):
        """Initialize the ChatPage instance."""
        pass

    def write(self):
        """
        Render the chat interface and handle user interactions.

        This method sets up the Streamlit interface, manages file uploads,
        processes user input, generates responses, and displays the chat history.
        """
        message(
            "Welcome to PaperMaid! Ask me anything about your research.", is_user=False
        )

        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "uploaded_files" not in st.session_state:
            st.session_state["uploaded_files"] = []

        style = f"""
  """
        st.markdown(style, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf"],
            accept_multiple_files=True,
            key="fileUploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_files = [
                file
                for file in uploaded_files
                if file not in st.session_state["uploaded_files"]
            ]
            if new_files:
                st.session_state["uploaded_files"].extend(new_files)
                for file in new_files:
                    st.write(f"Processing file: {file.name}")
                    asyncio.run(self.chat_completion.process_and_store_file(file))
                st.success("Successful")

        user_input = st.text_input(
            "Prompt here: ", key="input", label_visibility="collapsed"
        )
        if user_input:
            output = asyncio.run(self.chat_completion.chat_completion(user_input))

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


def main():
    """Initialize and run the ChatPage application."""
    chat_page = ChatPage()
    chat_page.write()


if __name__ == "__main__":
    main()
