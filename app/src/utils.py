import streamlit as st
from abc import ABC, abstractmethod

class Page(ABC):
    @abstractmethod
    def write(self):
        pass


def add_custom_css():
    st.markdown(
        """
        <style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p 
        {
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )