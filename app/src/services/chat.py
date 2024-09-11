import json
import logging
from typing import Any

import tempfile
import os

import config
from openai import OpenAI
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from src.services.data_processor import DataProcessor

logger = logging.getLogger("papermaid")


class ChatCompletion:
    """
    Manages chat completions using OpenAI, vector search, and document processing.

    This class handles various aspects of chat completion, including:
    - Initializing connections to OpenAI, CosmosDB, and other necessary services
    - Performing vector searches on stored documents
    - Retrieving chat history
    - Generating chat completions based on user input and context
    - Processing and storing uploaded files for future reference
    """

    def __init__(
        self,
        cosmos_db: CosmosDB,
        embeddings_generator: LangchainEmbeddingsGenerator,
        data_processor: DataProcessor,
    ) -> None:
        """Initialize the ChatCompletion instance with necessary services."""
        self.client = OpenAI(api_key=config.OPENAI_KEY)
        self.cosmos_db = cosmos_db
        self.embeddings_generator = embeddings_generator
        self.data_processor = data_processor

    def vector_search(
        self, vectors: list[float], similarity_score=0.02, num_results=5
    ) -> list[dict[str, Any]]:
        """
        Search the Cosmos DB for the most similar vectors to the given vectors.

        :param vectors: The vectors to search for.
        :param similarity_score: The minimum similarity score to return.
        :param num_results: The number of results to return.
        :return: The most similar vectors to the given vectors
        """
        results = self.cosmos_db.query_items(
            query="""
    SELECT TOP @num_results c.overview, VectorDistance(c.vector, @embedding) as SimilarityScore 
    FROM c
    WHERE VectorDistance(c.vector, @embedding) > @similarity_score
    ORDER BY VectorDistance(c.vector, @embedding)
    """,
            parameters=[
                {"name": "@embedding", "value": vectors},
                {"name": "@num_results", "value": num_results},
                {"name": "@similarity_score", "value": similarity_score},
            ],
        )

        formatted_results = [
            {"SimilarityScore": result.pop("SimilarityScore"), "document": result}
            for result in results
        ]
        return formatted_results

    def get_chat_history(self, completions=3) -> list:
        """Retrieve the most recent chat history from Cosmos DB."""
        results = self.cosmos_db.query_items(
            query="""
    SELECT TOP @completions *
    FROM c
    ORDER BY c._ts DESC
    """,
            parameters=[{"name": "@completions", "value": completions}],
        )
        logger.info("Done getting chat history")
        return list(results)

    def generate_completion(
        self, user_prompt: str, vector_search_results: list, chat_history: list[dict]
    ) -> dict[str, Any]:
        """
        Generate a chat completion based on user input and context.

        :param user_prompt: The user prompt to complete.
        :param vector_search_results: The vector search results from the PDF content.
        :param chat_history: The chat history.
        :return: The dictionary representation of the model's response.
        """
        system_prompt = """
        You are an advanced AI assistant specializing in academic research analysis. Your primary functions are to summarize scientific papers and identify relationships among papers or conferences based on PDF files provided by users.
        Your capabilities include:
        1. Summarizing individual papers, highlighting key findings, methodologies, and conclusions.
        2. Identifying common themes, methodologies, or research directions across multiple papers.
        3. Analyzing trends in conference proceedings over time.
        4. Comparing and contrasting different papers on similar topics.
        5. Identifying potential collaborations or research gaps based on the analyzed papers.
        Guidelines:
        - Provide concise yet comprehensive summaries of individual papers.
        - When analyzing multiple papers or conferences, focus on identifying relationships, trends, and patterns.
        - Use academic language and maintain a professional tone.
        - If asked about specific details not present in the provided information, politely state that the information is not available in the given context.
        - When appropriate, structure your responses with clear headings or bullet points for better readability.
        - If requested, provide a list of key papers or authors that appear influential based on your analysis.

        Remember to base your responses solely on the information extracted from the PDF files and provided in the vector search results. Do not make assumptions or include external information not present in the given context.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"{chat.get('prompt', '')} {chat.get('completion', '')}",
                }
                for chat in chat_history
            ]
        )
        messages.append({"role": "user", "content": user_prompt})
        messages.extend(
            [
                {"role": "system", "content": json.dumps(result["document"])}
                for result in vector_search_results
            ]
        )

        logger.debug("Messages going to OpenAI: %s", messages)

        response = self.client.chat.completions.create(
            model=config.OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.1,
        )
        logger.debug("Done generating completions in generate_completion")
        return response.model_dump()

    async def chat_completion(self, user_input: str) -> str:
        """Generate a chat completion based on user input."""
        logger.info("Starting completion: %s", user_input)
        user_embeddings = await self.embeddings_generator.generate_embeddings(
            user_input
        )
        search_results = self.vector_search(user_embeddings)
        chat_history = self.get_chat_history(3)
        completions_results = self.generate_completion(
            user_input, search_results, chat_history
        )
        completions_results = completions_results["choices"][0]["message"]["content"]
        logger.info("Done generating completions: %s", completions_results)
        return completions_results

    async def process_and_store_file(self, file):
        """Process and store an uploaded file for future reference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            data = await self.data_processor.process_pdfs(temp_dir)
            vector_property = "vector"
            data_with_vectors = await self.data_processor.generate_vectors(
                data, vector_property
            )
            await self.data_processor.insert_data(data_with_vectors)
