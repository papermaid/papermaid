import asyncio
import logging

import streamlit as st
from src.services.chat import ChatCompletion
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.graph import KnowledgeGraphManager
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from streamlit.components.v1 import html
from streamlit_chat import message
from src.services.web_search import BingSearchClient

logger = logging.getLogger("papermaid")


class ChatPage:
    """
    Manages the chat interface and user interactions.

    This class provides functionality for:
    - Initializing and managing the chat session state
    - Processing uploaded files
    - Handling user input and generating responses
    - Rendering the chat interface
    """

    def __init__(self):
        """
        Initialize the ChatPage with necessary services and session state.
        """
        self.cosmos_db = CosmosDB()
        self.embeddings_generator = LangchainEmbeddingsGenerator()
        self.data_processor = DataProcessor(self.cosmos_db, self.embeddings_generator)
        self.knowledge_graph_manager = KnowledgeGraphManager(self.data_processor)
        self.bing_search_client = BingSearchClient()
        self.chat_completion = ChatCompletion(
            self.cosmos_db,
            self.embeddings_generator,
            self.data_processor,
            self.knowledge_graph_manager,
            self.bing_search_client,
        )

        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "file_contents" not in st.session_state:
            st.session_state["file_contents"] = []
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = []
        if "user_input" not in st.session_state:
            st.session_state["user_input"] = ""
        if "Internet Access_results" not in st.session_state:
            st.session_state["Internet Access_results"] = []

    async def process_files(self, files):
        """
        Process multiple files and return their contents as chunks.

        :param files: A list of uploaded file objects.
        :return: A list of processed file contents.
        """
        tasks = [self.chat_completion.process_file(file) for file in files]
        return await asyncio.gather(*tasks)

    def handle_input(self, use_graph=False, *args, **kwargs):
        """
        Handle user input, generate a response, and update the chat history.
        """
        if st.session_state["user_input"]:
            user_input = st.session_state["user_input"]
            search_results = self.bing_search_client.search(user_input)
            if search_results:
                urls = self.bing_search_client.format_url(search_results)
                snippets = self.bing_search_client.format_snippet(search_results)
                st.session_state["Internet Access_results"] = list(zip(urls, snippets))

            output = asyncio.run(
                self.chat_completion.chat_completion(
                    user_input,
                    st.session_state["file_contents"],
                    st.session_state["Internet Access_results"],
                )
            )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)
            st.session_state["user_input"] = ""

            if use_graph and self.knowledge_graph_manager.save_graph():
                with open("nx.html", "r") as f:
                    graph_html = f.read()
                    html(graph_html, height=750)

    def write(self, use_graph=False):
        """
        Render the chat interface and handle user interactions.
        """
        st.title("PaperMaid Chat")

        message(
            "Welcome to PaperMaid! Ask me anything about your research.", is_user=False
        )

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
                if file not in st.session_state["processed_files"]
            ]
            if new_files:
                st.write(f"Processing {len(new_files)} new file(s)...")
                new_contents = asyncio.run(self.process_files(new_files))
                st.session_state["file_contents"].extend(new_contents)
                st.session_state["processed_files"].extend(new_files)
                st.success(f"Successfully processed {len(new_files)} file(s)")

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))

            if st.session_state["Internet Access_results"]:
                st.write("Related web search results:")
                for result in st.session_state["Internet Access_results"][:3]:
                    if isinstance(result, tuple) and len(result) >= 1:
                        url = result[0]
                        st.markdown(f"- [{url}]({url})")
                    elif isinstance(result, str):
                        st.markdown(f"- [{result}]({result})")
                    else:
                        st.write(f"Unexpected result format: {result}")

                st.write("---")
            else:
                st.write("No web search results available.")
        st.text_input(
            "Ask a question:",
            key="user_input",
            on_change=lambda: self.handle_input(use_graph=use_graph),
        )


def main():
    chat_page = ChatPage()
    chat_page.write()


if __name__ == "__main__":
    main()
