import asyncio
import logging

import config
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("papermaid")


class LangchainEmbeddingsGenerator:
    """
    Generates embeddings for text using OpenAI's embedding model via Langchain.

    This class provides functionality for:
    - Splitting text into chunks
    - Generating embeddings for each chunk
    - Calculating the average embedding for the entire text
    """

    def __init__(self):
        """
        Initialize the LangchainEmbeddingsGenerator with OpenAI embeddings and text splitter.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def generate_embeddings(self, text: str):
        """
        Generate embeddings for the given text.

        This method splits the text into chunks, generates embeddings for each chunk,
        and then calculates the average embedding.

        :param text: The input text to generate embeddings for.
        :return: The average embedding vector for the entire text.
        """
        docs = [Document(page_content=text)]
        chunks = self.text_splitter.split_documents(docs)
        embeddings = []
        for chunk in chunks:
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, chunk.page_content
            )
            embeddings.append(embedding)

        avg_embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        logger.info("Done embedding generation")
        return avg_embedding
