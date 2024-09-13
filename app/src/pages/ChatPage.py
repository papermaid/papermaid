import asyncio
import logging

from src.services.chat import ChatCompletion
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator

logger = logging.getLogger("papermaid")

import streamlit as st
from streamlit_chat import message


class ChatPage:
    chat_history = []
    cosmos_db = CosmosDB()
    embeddings_generator = LangchainEmbeddingsGenerator()
    data_processor = DataProcessor(cosmos_db, embeddings_generator)
    chat_completion = ChatCompletion(cosmos_db, embeddings_generator, data_processor)

    def __init__(self):
        pass

    async def process_files(self, files):
        """Process multiple files and return their contents as chunks."""
        tasks = [self.chat_completion.process_file(file) for file in files]
        return await asyncio.gather(*tasks)

    def write(self):
        message(
            "Welcome to PaperMaid! Ask me anything about your research.", is_user=False
        )

        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "file_contents" not in st.session_state:
            st.session_state["file_contents"] = []

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
                if file not in st.session_state.get("processed_files", [])
            ]
            if new_files:
                st.write(f"Processing {len(new_files)} new file(s)...")
                new_contents = asyncio.run(self.process_files(new_files))
                st.session_state["file_contents"].extend(new_contents)
                if "processed_files" not in st.session_state:
                    st.session_state["processed_files"] = []
                st.session_state["processed_files"].extend(new_files)
                st.success(f"Successfully processed {len(new_files)} file(s)")

        user_input = st.text_input(
            "Prompt here: ", key="input", label_visibility="collapsed"
        )
        if user_input:
            output = asyncio.run(
                self.chat_completion.chat_completion(
                    user_input, st.session_state["file_contents"]
                )
            )

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
