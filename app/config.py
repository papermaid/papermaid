import os
from dotenv import load_dotenv
from decouple import config

load_dotenv("/Users/krittinsetdhavanich/Downloads/papermaid/.env")


# Cosmos DB configuration
COSMOS_CONN = os.getenv('cosmos_uri')
COSMOS_KEY = os.getenv('cosmos_key')
COSMOS_DATABASE = os.getenv('cosmos_database_name')
COSMOS_COLLECTION = os.getenv('cosmos_collection_name')
COSMOS_VECTOR_PROPERTY = os.getenv('cosmos_vector_property_name')

# OpenAI configuration
OPENAI_ENDPOINT = os.getenv('openai_endpoint')
OPENAI_KEY = os.getenv('openai_key')
OPENAI_API_VERSION = os.getenv('openai_api_version')
OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv('openai_embeddings_deployment')
OPENAI_EMBEDDINGS_DIMENSIONS = os.getenv('openai_embeddings_dimensions')
OPENAI_COMPLETIONS_DEPLOYMENT = os.getenv('openai_completions_deployment')