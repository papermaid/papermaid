import logging
import logging_config
import streamlit as st

logging_config.setup_logging()

from src.pages import PAGE_MAP
from src.utils import add_custom_css


add_custom_css()

def main():
    current_page = st.sidebar.radio(label=":rainbow[**Pages**]",
                                    options=list(PAGE_MAP))
    PAGE_MAP[current_page]().write()

if __name__ == "__main__":
    main()
