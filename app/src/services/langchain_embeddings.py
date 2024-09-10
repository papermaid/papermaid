import logging

import config
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

logger = logging.getLogger('papermaid')


class LangchainEmbeddingsGenerator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def generate_embeddings(self, text: str):
        docs = [Document(page_content=text)]
        chunks = self.text_splitter.split_documents(docs)

        embeddings = []
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk.page_content)
            embeddings.append(embedding)

        avg_embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        logger.info("Done embedding generation")
        return avg_embedding
