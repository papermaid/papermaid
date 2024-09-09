import json

from openai import OpenAI

import config


class ChatCompletion:
    def __init__(self, cosmos_db, embeddings_generator):
        self.client = OpenAI(api_key=config.OPENAI_KEY)
        self.cosmos_db = cosmos_db
        self.embeddings_generator = embeddings_generator

    def vector_search(self, vectors, similarity_score=0.02, num_results=5):
        print("Starting vector search")
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
        print("Done vector search")
        return formatted_results

    def get_chat_history(self, completions=3):
        print("Getting chat history")
        results = self.cosmos_db.query_items(
            query='''
        SELECT TOP @completions *
        FROM c
        ORDER BY c._ts DESC
        ''',
            parameters=[{"name": "@completions", "value": completions}]
        )
        print("Done getting chat history")
        return list(results)

    def generate_completion(self, user_prompt, vector_search_results,
                            chat_history):
        system_prompt = '''
    You are an intelligent assistant for movies. You are designed to provide helpful answers to user questions about movies in your database.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
        - Only answer questions related to the information provided below. Provide at least 3 candidate movie answers in a list.
        - Write two lines of whitespace between each answer in the list.
    '''
        print("Generating completions")

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(
            [{'role': 'user',
              'content': f"{chat['prompt']} {chat['completion']}"}
             for chat in chat_history])
        messages.append({'role': 'user', 'content': user_prompt})
        messages.extend(
            [{'role': 'system', 'content': json.dumps(result['document'])} for
             result in vector_search_results])

        print("Messages going to OpenAI:", messages)

        response = self.client.chat.completions.create(
            model=config.OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.1
        )
        print("Done generating completions")
        return response.model_dump()

    def chat_completion(self, user_input):
        print("Starting completion")
        user_embeddings = self.embeddings_generator.generate_embeddings(
            user_input)
        print("Searching vectors")
        search_results = self.vector_search(user_embeddings)
        print("Getting Chat History")
        chat_history = self.get_chat_history(3)
        print("Generating completions")
        completions_results = self.generate_completion(user_input,
                                                       search_results,
                                                       chat_history)
        return completions_results['choices'][0]['message']['content']

