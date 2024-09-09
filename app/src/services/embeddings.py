import logging
from openai import OpenAI

import config

logger = logging.getLogger('papermaid')

class EmbeddingsGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_KEY)

    def generate_embeddings(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=config.OPENAI_EMBEDDINGS_MODEL,
            input=text,
            encoding_format="float"
        )
        embeddings = response.model_dump()
        logger.info("Done embedding generation")
        return embeddings['data'][0]['embedding']

