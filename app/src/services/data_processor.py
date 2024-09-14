import asyncio
import logging
import os
import time
from typing import List
import uuid

import tiktoken
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
import config
import tempfile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


logger = logging.getLogger("papermaid")


class DataProcessor:
    """
    Handles processing of PDF files, text extraction, vector generation, and data insertion.

    This class provides functionality for:
    - Processing PDF files in a specified folder
    - Extracting text content from PDF files
    - Generating embedding vectors for text content
    - Inserting processed data into a Cosmos DB collection
    - Splitting content into chunks
    """

    def __init__(
        self,
        cosmos_db: CosmosDB,
        langchain_embeddings_generator: LangchainEmbeddingsGenerator,
    ):
        """
        Initialize the DataProcessor with necessary services.

        :param cosmos_db: An instance of CosmosDB for database operations.
        :param langchain_embeddings_generator: An instance of LangchainEmbeddingsGenerator for generating embeddings.
        """
        self.cosmos_db = cosmos_db
        self.langchain_embeddings_generator = langchain_embeddings_generator
        self.tokenizer = tiktoken.encoding_for_model(config.OPENAI_16k_MODEL)

    async def process_pdfs(self, folder_path: str) -> list[dict]:
        """
        Process all PDF files in the specified folder and extract text content.

        :param folder_path: The path to the folder containing the PDF files.
        :return: A list of dictionaries containing metadata and extracted text content.
        """
        logger.info(f"Processing PDF files in {folder_path}")
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        data = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            text_content = await self.extract_text_from_pdf(pdf_path)
            if not text_content:
                continue
            item = {
                "id": str(uuid.uuid4()),
                "filename": pdf_file,
                "content": text_content,
                "partitionKey": pdf_file,
            }
            data.append(item)
        logger.info("Done processing PDF files")
        return data

    @staticmethod
    async def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text content from a single PDF file.

        :param pdf_path: The path to the PDF file.
        :return: Extracted text content as a string.
        """
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                if len(reader.pages) == 0:
                    logger.warning(f"PDF file {pdf_path} has no pages.")
                    return ""

                text = ""
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        else:
                            logger.warning(f"Empty page found in {pdf_path}")
                    except Exception as e:
                        logger.error(
                            f"Error extracting text from page in {pdf_path}: {str(e)}"
                        )

                if not text:
                    logger.warning(f"No text could be extracted from {pdf_path}")
                return text
        except PdfReadError as e:
            logger.error(f"Error reading PDF file {pdf_path}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path}: {str(e)}")
            return ""

    async def generate_vectors(self, items: list[dict], vector_property: str):
        """
        Generate embedding vectors for the content of each item in the provided list.

        :param items: A list of dictionaries where each dictionary represents an item containing text content to be embedded.
        :param vector_property: The key under which the generated embedding vectors will be stored in each item dictionary.
        :return: The list of items with the generated vectors added.
        """
        for item in items:
            vectorArray = await self.langchain_embeddings_generator.generate_embeddings(
                item["content"]
            )
            item[vector_property] = vectorArray
        logger.info("Done generating vectors")
        return items

    async def insert_data(self, data):
        """
        Insert a list of data items into a Cosmos DB collection.

        This method handles concurrent insertions to optimize performance.

        :param data: A list of data items to be inserted into the database.
        """
        start_time = time.time()
        counter = 0
        tasks = []
        max_concurrency = 5
        semaphore = asyncio.Semaphore(max_concurrency)

        async def upsert_object(obj):
            nonlocal counter
            async with semaphore:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.cosmos_db.upsert_item, obj
                )
                counter += 1
                if counter % 100 == 0:
                    logger.debug(
                        f"Sent {counter} documents for insertion into collection."
                    )

        for obj in data:
            tasks.append(asyncio.create_task(upsert_object(obj)))

        await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"All {counter} documents inserted!")
        logger.debug(
            f"Time taken: {duration:.2f} seconds ({duration:.3f} milliseconds)"
        )

    def split_content(self, content: str, max_tokens: int = 12000) -> list[str]:
        """
        Split the content into chunks of approximately max_tokens.

        :param content: The text content to be split.
        :param max_tokens: The maximum number of tokens per chunk.
        :return: A list of content chunks.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for line in content.split("\n"):
            line_length = len(self.tokenizer.encode(line))
            if current_length + line_length > max_tokens:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def pdf_to_document(self, uploaded_file) -> list[Document]:
        """
        Convert an uploaded PDF file to a list of Langchain Document objects.

        :param uploaded_file: The UploadedFile object from Streamlit.
        :return: A list of Langchain Document objects.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            pages: List[Document] = loader.load_and_split()

            os.unlink(temp_file_path)

            return pages
        except Exception as e:
            logger.error(
                f"Error processing PDF file {uploaded_file.name}: {str(e)}")
            return []
