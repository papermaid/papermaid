import streamlit as st


class SettingPage:
    def __init__(self, state):
        self.state = state

    def write(self):
        st.title("Chat Page")
        st.chat_message("user")
