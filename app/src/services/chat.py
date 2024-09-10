import json
import logging
from typing import Any

import config
from openai import OpenAI
from src.services.database import CosmosDB
from src.services.langchain_embeddings import LangchainEmbeddingsGenerator

logger = logging.getLogger('papermaid')


class ChatCompletion:
    def __init__(self, cosmos_db: CosmosDB,
                 embeddings_generator: LangchainEmbeddingsGenerator) -> None:
        self.client = OpenAI(api_key=config.OPENAI_KEY)
        self.cosmos_db = cosmos_db
        self.embeddings_generator = embeddings_generator

    def vector_search(self, vectors: list[float], similarity_score=0.02,
                      num_results=5) -> list[dict[str, Any]]:
        """
        Search the Cosmos DB for the most similar vectors to the given vectors.

        :param vectors: The vectors to search for.
        :param similarity_score: The minimum similarity score to return.
        :param num_results: The number of results to return.
        :return: The most similar vectors to the given vectors
        """
        results = self.cosmos_db.query_items(
            query='''
        SELECT TOP @num_results c.overview, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector, @embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector, @embedding)
        ''',
            parameters=[
                {"name": "@embedding", "value": vectors},
                {"name": "@num_results", "value": num_results},
                {"name": "@similarity_score", "value": similarity_score}
            ]
        )

        formatted_results = [
            {'SimilarityScore': result.pop('SimilarityScore'),
             'document': result}
            for result in results
        ]
        return formatted_results

    def get_chat_history(self, completions=3) -> list:
        """
        Get the chat history from the Cosmos DB.

        :param completions: The number of completions to return.
        :return: The chat history.
        """
        results = self.cosmos_db.query_items(
            query='''
        SELECT TOP @completions *
        FROM c
        ORDER BY c._ts DESC
        ''',
            parameters=[{"name": "@completions", "value": completions}]
        )
        logger.info("Done getting chat history")
        return list(results)

    def generate_completion(self, user_prompt: str,
                            vector_search_results: list,
                            chat_history: list[dict]) -> dict[str, Any]:
        """
        Get dictionary representation of the model.

        :param user_prompt: The user prompt to complete.
        :param vector_search_results: The vector search results.
        :param chat_history: The chat history.
        :return: The dictionary representation of the model.
        """
        system_prompt = '''
    You are an intelligent assistant for movies. You are designed to provide helpful answers to user questions about movies in your database.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
        - Only answer questions related to the information provided below. Provide at least 3 candidate movie answers in a list.
        - Write two lines of whitespace between each answer in the list.
    '''

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(
            [{'role': 'user',
              'content': f"{chat.get('prompt', '')} {chat.get('completion', '')}"}
             for chat in chat_history])
        messages.append({'role': 'user', 'content': user_prompt})
        messages.extend(
            [{'role': 'system', 'content': json.dumps(result['document'])} for
             result in vector_search_results])

        logger.debug("Messages going to OpenAI: %s", messages)

        response = self.client.chat.completions.create(
            model=config.OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.1
        )
        logger.debug("Done generating completions in generate_completion")
        return response.model_dump()

    def chat_completion(self, user_input: str) -> str:
        logger.info("Starting completion: %s", user_input)
        user_embeddings = self.embeddings_generator.generate_embeddings(
            user_input)
        search_results = self.vector_search(user_embeddings)
        chat_history = self.get_chat_history(3)
        completions_results = self.generate_completion(user_input,
                                                       search_results,
                                                       chat_history)
        completions_results = completions_results['choices'][0]['message'][
            'content']
        logger.info("Done generating completions: %s", completions_results)
        return completions_results
