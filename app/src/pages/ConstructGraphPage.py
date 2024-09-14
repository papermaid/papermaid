import asyncio
import logging

import streamlit as st
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.graph import KnowledgeGraphManager
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator

logger = logging.getLogger("papermaid")


class ConstructGraphPage:
    def __init__(self):
        self.cosmos_db = CosmosDB()
        self.embeddings_generator = LangchainEmbeddingsGenerator()
        self.data_processor = DataProcessor(self.cosmos_db, self.embeddings_generator)
        self.knowledge_graph_manager = KnowledgeGraphManager(self.data_processor)

        if "processed_files_2" not in st.session_state:
            st.session_state["processed_files_2"] = []
        if "user_input_2" not in st.session_state:
            st.session_state["user_input_2"] = ""
        if "generated_2" not in st.session_state:
            st.session_state["generated_2"] = []

    async def process_files(self, files, openai_model):
        """
        Process multiple files and return their contents as chunks.

        :param files: A list of uploaded file objects.
        :return: A list of processed file contents.
        """
        tasks = [self.knowledge_graph_manager.construct_graph(file, openai_model) for file in files]
        return await asyncio.gather(*tasks)

    def handle_input(self, openai_model):
        """
        Handle user input, generate a response, and update the chat history.
        """
        if st.session_state["user_input_2"]:
            user_input = st.session_state["user_input_2"]
            output = asyncio.run(
                self.knowledge_graph_manager.construct_graph_from_topic(user_input, openai_model)
            )
            if output:
                st.session_state["generated_2"].append(
                    "Constructing graph knowledge from topic: " + user_input + " Successful")
            else:
                st.session_state["generated_2"].append(
                    "Fail to construct graph knowledge from topic: " + user_input)


    def write(self, openai_model: str, use_graph: bool):
        logger.info("Rendering ConstructGraphPage...")
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
                if file not in st.session_state["processed_files_2"]
            ]
            if new_files:
                st.write(f"Processing {len(new_files)} new file(s)...")
                asyncio.run(self.process_files(new_files, openai_model))
                st.session_state["processed_files_2"].extend(new_files)
                st.success(f"Successfully processed {len(new_files)} file(s)")

        st.text_input(
            "Enter a topic to construct graph knowledge",
            key="user_input",
            on_change=lambda: self.handle_input(openai_model),
            placeholder="Enter a topic to construct graph knowledge",
        )
        st.text("Topic example: Machine Learning, Attention (Machine Learning), etc.")

        if st.session_state["generated_2"]:
            st.text("\n".join(st.session_state["generated_2"]))
