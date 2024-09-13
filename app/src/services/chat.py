import logging
from typing import Any, List
import tempfile
import os
import asyncio
import tiktoken

import config
from openai import OpenAI
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator
from src.services.data_processor import DataProcessor

logger = logging.getLogger("papermaid")


class ChatCompletion:
    def __init__(
        self,
        cosmos_db: CosmosDB,
        embeddings_generator: LangchainEmbeddingsGenerator,
        data_processor: DataProcessor,
    ) -> None:
        self.client = OpenAI(api_key=config.OPENAI_KEY)
        self.cosmos_db = cosmos_db
        self.embeddings_generator = embeddings_generator
        self.data_processor = data_processor
        self.tokenizer = tiktoken.encoding_for_model(config.OPENAI_16k_MODEL)

    async def process_file(self, file):
        """Extract content from the uploaded file and split into chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            content = await self.data_processor.extract_text_from_pdf(file_path)
        return self.split_content(content)

    def split_content(self, content: str, max_tokens: int = 12000) -> List[str]:
        """Split the content into chunks of approximately max_tokens."""
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

    async def summarize_content(self, content: str, max_tokens: int = 2000) -> str:
        """Summarize the given content to fit within the specified token limit."""
        summary_prompt = f"Summarize the following content in about {max_tokens} tokens, focusing on the most important information:\n\n{content}"

        response = self.client.chat.completions.create(
            model=config.OPENAI_16k_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes academic content.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content

    async def chat_completion(
        self, user_input: str, file_contents: List[List[str]]
    ) -> str:
        """Generate a chat completion based on user input and summarized file contents."""
        logger.info("Starting completion: %s", user_input)

        summarized_contents = []
        for file_chunks in file_contents:
            file_summaries = await asyncio.gather(
                *[self.summarize_content(chunk) for chunk in file_chunks]
            )
            summarized_contents.append("\n".join(file_summaries))

        combined_input = f"{user_input}\n\nSummarized File Contents:\n"
        for i, summary in enumerate(summarized_contents, 1):
            combined_input += f"(Title).\nSummary:\n{summary}\n\n"

        user_embeddings = await self.embeddings_generator.generate_embeddings(
            combined_input
        )
        search_results = self.vector_search(user_embeddings)
        chat_history = self.get_chat_history(3)
        completions_results = self.generate_completion(
            combined_input, search_results, chat_history
        )
        completions_results = completions_results["choices"][0]["message"]["content"]
        logger.info("Done generating completions: %s", completions_results)
        return completions_results

    def vector_search(
        self, vectors: list[float], similarity_score=0.8, num_results=3
    ) -> list[dict[str, Any]]:
        results = self.cosmos_db.query_items(
            query="""
        SELECT TOP @num_results c.filename, c.content, VectorDistance(c.vector, @embedding) as SimilarityScore 
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

      Remember to base your responses on the information extracted from the PDF files, provided in the user prompt, and the vector search results. Do not make assumptions or include external information not present in the given context.
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

        context = "Relevant information from the database:\n"
        for result in vector_search_results:
            context += f"File: {result['document']['filename']}\n"
            context += f"Content: {result['document']['content'][:500]}...\n\n"
        messages.append({"role": "system", "content": context})

        logger.debug("Messages going to OpenAI: %s", messages)

        response = self.client.chat.completions.create(
            model=config.OPENAI_16k_MODEL,
            messages=messages,
            temperature=0.1,
        )
        logger.debug("Done generating completions in generate_completion")
        return response.model_dump()
