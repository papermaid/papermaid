import time

import gradio as gr

from chat import ChatCompletion
from data_processor import DataProcessor
from database import CosmosDB
from embeddings import EmbeddingsGenerator


class App:
    def __init__(self):
        self.chat_history = []
        self.cosmos_db = CosmosDB()
        self.embeddings_generator = EmbeddingsGenerator()
        self.chat_completion = ChatCompletion(self.cosmos_db,
                                              self.embeddings_generator)
        self.data_processor = DataProcessor(self.cosmos_db,
                                            self.embeddings_generator)

    def user(self, user_message, chat_history):
        start_time = time.time()
        response_payload = self.chat_completion.chat_completion(user_message)
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)

        details = f"\n (Time: {elapsed_time}ms)"
        chat_history.append([user_message, response_payload + details])
        return gr.update(value=""), chat_history

    def create_interface(self):
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(label="Cosmic Movie Assistant")
            msg = gr.Textbox(
                label="Ask me about movies in the Cosmic Movie Database!")
            clear = gr.Button("Clear")

            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False)
            clear.click(lambda: None, None, chatbot, queue=False)
        return demo

    def run(self):
        demo = self.create_interface()
        demo.launch(debug=True)


if __name__ == "__main__":
    app = App()
    app.run()
