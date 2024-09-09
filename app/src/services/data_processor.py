import asyncio
import logging
import os
import time
import uuid

from PyPDF2 import PdfReader

from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator

logger = logging.getLogger('papermaid')

class DataProcessor:
    def __init__(self, cosmos_db: CosmosDB, langchain_embeddings_generator: LangchainEmbeddingsGenerator):
        self.cosmos_db = cosmos_db
        self.langchain_embeddings_generator = langchain_embeddings_generator

    async def process_and_insert_pdfs(self, folder_path, vector_property):
        data = await self.process_pdfs(folder_path)
        data_with_vectors = await self.generate_vectors(data, vector_property)
        await self.insert_data(data_with_vectors)

    async def process_pdfs(self, folder_path: str) -> list[dict]:
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        data = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            text_content = await self.extract_text_from_pdf(pdf_path)

            item = {
                'id': str(uuid.uuid4()),
                'filename': pdf_file,
                'content': text_content,
                'partitionKey': pdf_file
            }
            data.append(item)
        logger.info(f"Processed {len(data)} PDF files")
        return data

    @staticmethod
    async def extract_text_from_pdf(pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    async def generate_vectors(self, items: dict, vector_property):
        for item in items:
            vectorArray = await self.langchain_embeddings_generator.generate_embeddings(
                item['content'])
            item[vector_property] = vectorArray
        logger.info("Done generating vectors")
        return items

    async def insert_data(self, data):
        start_time = time.time()
        counter = 0
        tasks = []
        max_concurrency = 5
        semaphore = asyncio.Semaphore(max_concurrency)

        async def upsert_object(obj):
            nonlocal counter
            async with semaphore:
                await asyncio.get_event_loop().run_in_executor(None,
                                                               self.cosmos_db.upsert_item,
                                                               obj)
                counter += 1
                if counter % 100 == 0:
                    logger.debug(
                        f"Sent {counter} documents for insertion into collection.")

        for obj in data:
            tasks.append(asyncio.create_task(upsert_object(obj)))

        await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"All {counter} documents inserted!")
        logger.debug(f"Time taken: {duration:.2f} seconds ({duration:.3f} milliseconds)")