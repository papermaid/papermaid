from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import config


class LangchainEmbeddingsGenerator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def generate_embeddings(self, text):
        docs = [Document(page_content=text)]
        chunks = self.text_splitter.split_documents(docs)

        embeddings = []
        for chunk in chunks:
            embedding = await self.embeddings.aembed_query(chunk.page_content)
            embeddings.append(embedding)

        avg_embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        print("Done embedding generation")
        return avg_embedding
