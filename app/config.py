import os
from dotenv import load_dotenv
from decouple import config

load_dotenv(".env")


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

# Model configuration
OPENAI_EMBEDDINGS_MODEL = "text-embedding-ada-002"

# Neo4j configuration
NEO4J_URl = os.getenv('neo4j_urL')
NEO4J_USERNAME = os.getenv('neo4j_username')
NEO4J_PASSWORD = os.getenv('neo4j_password')