from openai import OpenAI

import config


class EmbeddingsGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_KEY)

    def generate_embeddings(self, text):
        print("Generating embeddings")
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        print("Done generating embeddings, dumping model")
        embeddings = response.model_dump()
        print("Done embedding generation")
        return embeddings['data'][0]['embedding']
