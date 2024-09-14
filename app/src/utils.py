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
        div[class*="stRadio"]> div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            display: none;
        }

        div[class*="stRadio"]> div[role="radiogroup"] > label[data-baseweb="radio"]  {
            display: block;
            padding: 10px;
            margin-bottom: 4px;
            background-color: #1c1c1d;
            border-radius: 8px;
            border: 2px solid #2b2b2c;
            cursor: pointer;
            transition: background-color 0.3s ease;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
            font-size: 24px
            
        }

        div[class*="stRadio"] > div[role="radiogroup"] > label:hover {
            background-color: #2e2e2f;
        }

        [data-testid=stSidebar] {
            background-color: #1c1c1d;
            border-style: solid;
            border-color: #2b2b2c;
        }

        [data-testid="stAppViewContainer"] > .main {
            background-color: #161617;
        }

        [data-testid="stHeader"] {
            background-color: #161617;
        }
        </style>
        """,
        # 191919
        unsafe_allow_html=True,
    )
