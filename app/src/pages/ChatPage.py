import streamlit as st
from src.services.chat import ChatCompletion
from src.services.data_processor import DataProcessor
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from streamlit_chat import message


class ChatPage:
    chat_history = []
    cosmos_db = CosmosDB()
    embeddings_generator = LangchainEmbeddingsGenerator()  # Change this line
    chat_completion = ChatCompletion(cosmos_db, embeddings_generator)
    data_processor = DataProcessor(cosmos_db, embeddings_generator)

    def __init__(self):
        pass

    async def upload_files(self, folder_path):
        data = self.data_processor.process_pdfs(folder_path)

    def write(self):
        message("Welcome to PaperMaid! Ask me anything about your research.",
                is_user=False)

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = []

        style = f"""
        <style>
            .stFileUploader {{
                z-index: 1;
                position: fixed;
                bottom: 6rem;
            }}
            .stTextInput {{
                z-index: 1;
                position: fixed;
                bottom: 3rem;
            }}
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)

        uploaded_files = st.file_uploader("Upload files", type=["pdf"],
                                          accept_multiple_files=True,
                                          key="fileUploader",
                                          label_visibility="collapsed")

        if uploaded_files:
            st.session_state['uploaded_files'] = uploaded_files
            # Process file tong nee
            for file in uploaded_files:
                st.write(f"Uploaded file: {file.name}")
                st.write(f"File type: {file.type}")
                st.write(f"File size: {file.size} bytes")
                st.write(f"File content: {file.getvalue()}")

        user_input = st.text_input("Prompt here: ", key="input",
                                   label_visibility="collapsed")

        if user_input:
            output = self.chat_completion.chat_completion(user_input)
            # output = "PLACEHOLDER OUTPUT PLACEHOLDER OUTPUT PLACEHOLDER OUTPUTPLACEHOLDER OUTPUTPLACEHOLDER OUTPUTPLACEHOLDER OUTPUT"

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True,
                        key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
