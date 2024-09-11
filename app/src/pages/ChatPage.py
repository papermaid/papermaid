import asyncio
import logging
import os
import tempfile

import streamlit as st
from src.services.chat import ChatCompletion
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from streamlit_chat import message

logger = logging.getLogger("papermaid")


class ChatPage:
    chat_history = []
    cosmos_db = CosmosDB()
    embeddings_generator = LangchainEmbeddingsGenerator()
    chat_completion = ChatCompletion(cosmos_db, embeddings_generator)
    data_processor = DataProcessor(cosmos_db, embeddings_generator)

    def __init__(self):
        pass

    def process_and_store_file(self, file):
        asyncio.run(self._process_and_store_file_async(file))

    async def _process_and_store_file_async(self, file):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            data = await self.data_processor.process_pdfs(temp_dir)
            vector_property = "vector"
            data_with_vectors = await self.data_processor.generate_vectors(
                data, vector_property
            )
            await self.data_processor.insert_data(data_with_vectors)

    def chat_completion_sync(self, user_input):
        return asyncio.run(self.chat_completion.chat_completion(user_input))

    def write(self):
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
                    self.process_and_store_file(file)
                st.success("Successful")

        user_input = st.text_input(
            "Prompt here: ", key="input", label_visibility="collapsed"
        )
        if user_input:
            output = self.chat_completion_sync(user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


def main():
    chat_page = ChatPage()
    chat_page.write()


if __name__ == "__main__":
    main()
