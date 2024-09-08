# Import required libraries
import asyncio
import json
import os
import time
import zipfile

import gradio as gr
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load configuration
load_dotenv(".env")

# Cosmos DB configuration
cosmos_conn = os.getenv('cosmos_uri')
cosmos_key = os.getenv('cosmos_key')
cosmos_database = os.getenv('cosmos_database_name')
cosmos_collection = os.getenv('cosmos_collection_name')
cosmos_vector_property = os.getenv('cosmos_vector_property_name')

# OpenAI configuration
openai_endpoint = os.getenv('openai_endpoint')
openai_key = os.getenv('openai_key')
openai_api_version = os.getenv('openai_api_version')
openai_embeddings_deployment = os.getenv('openai_embeddings_deployment')
openai_embeddings_dimensions = int(os.getenv('openai_embeddings_dimensions'))
openai_completions_deployment = os.getenv('openai_completions_deployment')

# Initialize clients
cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)
openai_client = AzureOpenAI(azure_endpoint=openai_endpoint, api_key=openai_key,
                            api_version=openai_api_version)
gpt_client = OpenAI(api_key=openai_key)
# Create database and containers
db = cosmos_client.create_database_if_not_exists(cosmos_database)

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/" + cosmos_vector_property,
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": openai_embeddings_dimensions
        },
    ]
}

indexing_policy = {
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [
        {"path": "/\"_etag\"/?"},
        {"path": "/" + cosmos_vector_property + "/*"},
    ],
    "vectorIndexes": [
        {
            "path": "/" + cosmos_vector_property,
            "type": "quantizedFlat"
        }
    ]
}

try:
    movies_container = db.create_container_if_not_exists(
        id=cosmos_collection,
        partition_key=PartitionKey(path='/partitionKey'),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
        offer_throughput=1000
    )
    print(f'Container with id \'{movies_container.id}\' created')

except exceptions.CosmosHttpResponseError as e:
    print(f"Error creating containers: {e}")
    raise


# @retry(wait=wait_random_exponential(min=2, max=300),
#        stop=stop_after_attempt(20))
def generate_embeddings(text):
    print("Generating embeddings")
    response = gpt_client.embeddings.create(
          model="text-embedding-ada-002",
          input=text,
            dimensions=openai_embeddings_dimensions,
          encoding_format="float"
        )
    print("Done generating embeddings, dumping model")
    embeddings = response.model_dump()
    print("Done embedding generation")
    return embeddings['data'][0]['embedding']


# Load and process data
with zipfile.ZipFile("/Users/krittinsetdhavanich/Downloads/papermaid/DataSet/Movies/MovieLens 4489 256D.zip", 'r') as zip_ref:
    zip_ref.extractall("../DataSet/Movies/")

with open('../DataSet/Movies/MovieLens-4489-256D.json', 'r') as d:
    data = json.load(d)

print(f"Loaded {len(data)} items from the dataset")


async def generate_vectors(items, vector_property):
    for item in items:
        vectorArray = await generate_embeddings(item['overview'])
        item[vector_property] = vectorArray
    print("Done generating vectors")
    return items


async def insert_data():
    start_time = time.time()
    counter = 0
    tasks = []
    max_concurrency = 5
    print("Starting document load, please wait...")
    semaphore = asyncio.Semaphore(max_concurrency)
    print("Starting document load, please wait...")

    def upsert_item_sync(obj):
        movies_container.upsert_item(body=obj)

    async def upsert_object(obj):
        nonlocal counter
        async with semaphore:
            await asyncio.get_event_loop().run_in_executor(None,
                                                           upsert_item_sync,
                                                           obj)
            counter += 1
            if counter % 100 == 0:
                print(
                    f"Sent {counter} documents for insertion into collection.")

    for obj in data:
        tasks.append(asyncio.create_task(upsert_object(obj)))

    await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    print(f"All {counter} documents inserted!")
    print(f"Time taken: {duration:.2f} seconds ({duration:.3f} milliseconds)")


# Vector search function
def vector_search(container, vectors, similarity_score=0.02, num_results=5):
    print("Starting vector search")
    results = container.query_items(
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
        ],
        enable_cross_partition_query=True,
        populate_query_metrics=True
    )

    formatted_results = [
        {'SimilarityScore': result.pop('SimilarityScore'), 'document': result}
        for result in results
    ]
    print("Done vector search")
    return formatted_results


# Get chat history
def get_chat_history(container, completions=3):
    print("Getting chat history")
    results = container.query_items(
        query='''
    SELECT TOP @completions *
    FROM c
    ORDER BY c._ts DESC
    ''',
        parameters=[{"name": "@completions", "value": completions}],
        enable_cross_partition_query=True
    )
    print("Done getting chat history")
    return list(results)


# Generate completion
def generate_completion(user_prompt, vector_search_results, chat_history):
    system_prompt = '''
You are an intelligent assistant for movies. You are designed to provide helpful answers to user questions about movies in your database.
You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
    - Only answer questions related to the information provided below. Provide at least 3 candidate movie answers in a list.
    - Write two lines of whitespace between each answer in the list.
'''
    print("Generating completions")

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(
        [{'role': 'user', 'content': f"{chat['prompt']} {chat['completion']}"}
         for chat in chat_history])
    messages.append({'role': 'user', 'content': user_prompt})
    messages.extend(
        [{'role': 'system', 'content': json.dumps(result['document'])} for
         result in vector_search_results])

    print("Messages going to OpenAI:", messages)

    response = openai_client.chat.completions.create(
        model=openai_completions_deployment,
        messages=messages,
        temperature=0.1
    )
    print("Done generating completions")
    return response.model_dump()


# Chat completion function
def chat_completion(movies_container, user_input):
    print("Starting completion")
    user_embeddings = generate_embeddings(user_input)
    print("Searching vectors")
    search_results = vector_search(movies_container, user_embeddings)
    print("Getting Chat History")
    chat_history = get_chat_history(movies_container, 3)
    print("Generating completions")
    completions_results = generate_completion(user_input, search_results,
                                              chat_history)
    return completions_results['choices'][0]['message']['content']


# Gradio interface
chat_history = []

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Cosmic Movie Assistant")
    msg = gr.Textbox(label="Ask me about movies in the Cosmic Movie Database!")
    clear = gr.Button("Clear")


    def user(user_message, chat_history):
        start_time = time.time()
        response_payload = chat_completion(movies_container, user_message)
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)

        details = f"\n (Time: {elapsed_time}ms)"
        chat_history.append([user_message, response_payload + details])
        return gr.update(value=""), chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(debug=True)
