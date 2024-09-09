import streamlit as st
from streamlit_chat import message
from decouple import config
from openai import OpenAI


# Initialize values
response = None
prompt_tokens = 0
completion_tokens = 0
total_tokens_used = 0
cost_of_response = 0

OPENAI_ORGANIZATION_ID=config('OPENAI_ORGANIZATION_ID')
OPENAI_PROJECT_ID=config('OPENAI_PROJECT_ID')
API_KEY=config('OPENAI_SECRET_KEY')


client = OpenAI(
    api_key=API_KEY,
    organization=OPENAI_ORGANIZATION_ID,
    project=OPENAI_PROJECT_ID,
)

class ChatPage:
    def __init__(self, state):
        self.state = state

    def make_request(question_input: str):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{question_input}"}],
    )
        print(response)
        return response

    def write(self):
        st.header("PAPERMAID")
        st.markdown("""---""")

        question_input = st.text_input("Your question")

        message("My message") 
        message("Hello bot!", is_user=True)