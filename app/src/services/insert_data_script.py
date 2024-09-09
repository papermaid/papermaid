import asyncio

from langchain_embeddings import LangchainEmbeddingsGenerator

from data_processor import DataProcessor
from database import CosmosDB


async def main():
    cosmos_db = CosmosDB()
    langchain_embeddings_generator = LangchainEmbeddingsGenerator()
    data_processor = DataProcessor(cosmos_db, langchain_embeddings_generator)

    pdf_folder_path = "../../../assets/papers"
    vector_property = "vector"

    await data_processor.process_and_insert_pdfs(pdf_folder_path,
                                                 vector_property)


if __name__ == "__main__":
    asyncio.run(main())
