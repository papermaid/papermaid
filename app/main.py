import logging
import logging_config
import streamlit as st

logging_config.setup_logging()

from src.pages import PAGE_MAP
from src.utils import add_custom_css


add_custom_css()

def main():
    with st.sidebar:
        st.title(":blue[Papermaid]👩🏻‍🏫")
        st.subheader("Read and understand any papers with :blue[Papermaid].")
        st.divider()

        current_page = st.sidebar.radio(options=list(PAGE_MAP), label="Navigation", label_visibility="collapsed")
    
        st.divider()
        st.subheader("Settings")
        openai_model = st.selectbox(
            "Choose OpenAI Model:",
            options=["gpt-3.5-turbo", "gpt-4o"],
            index=0,  # Default to gpt-3.5-turbo
        )
        use_graph = st.checkbox("Use Graph?", value=False)

    PAGE_MAP[current_page]().write(openai_model, use_graph)

if __name__ == "__main__":
    main()
