import logging

from azure.cosmos import container, CosmosClient, PartitionKey, exceptions

import config

logger = logging.getLogger("papermaid")


class CosmosDB:
    """
    Manages interactions with Azure Cosmos DB.

    This class provides functionality for:
    - Initializing a connection to Cosmos DB
    - Creating and managing containers
    - Upserting items into the database
    - Querying items from the database
    """

    def __init__(self):
        """
        Initialize the CosmosDB instance and set up the database and container.
        """
        self.client = CosmosClient(url=config.COSMOS_CONN, credential=config.COSMOS_KEY)
        self.db = self.client.create_database_if_not_exists(config.COSMOS_DATABASE)
        self.container = self.create_container()

    def create_container(self) -> container.ContainerProxy | None:
        """
        Create a Cosmos DB container with specified indexing and vector embedding policies.

        :return: The created container proxy.
        :raises: CosmosHttpResponseError if the container cannot be created.
        """
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/" + config.COSMOS_VECTOR_PROPERTY,
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": config.OPENAI_EMBEDDINGS_DIMENSIONS,
                },
            ]
        }

        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": '/"_etag"/?'},
                {"path": "/" + config.COSMOS_VECTOR_PROPERTY + "/*"},
            ],
            "vectorIndexes": [
                {"path": "/" + config.COSMOS_VECTOR_PROPERTY, "type": "quantizedFlat"}
            ],
        }

        try:
            container = self.db.create_container_if_not_exists(
                id=config.COSMOS_COLLECTION,
                partition_key=PartitionKey(path="/partitionKey"),
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
                offer_throughput=1000,
            )
            logger.info(f"Container with id '{container.id}' created")
            return container
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error creating containers: {e}")
            raise

    def upsert_item(self, item):
        """
        Insert or update the specified item in the Cosmos DB container.

        :param item: The item to be upserted.
        :return: The response from the upsert operation.
        """
        return self.container.upsert_item(body=item)

    def query_items(
        self,
        query,
        parameters,
        enable_cross_partition_query=True,
        populate_query_metrics=True,
    ):
        """
        Query items from the Cosmos DB container based on the provided query.

        :param query: The SQL query string.
        :param parameters: The parameters for the query.
        :param enable_cross_partition_query: Whether to enable cross-partition query.
        :param populate_query_metrics: Whether to populate query metrics.
        :return: An iterable of query results.
        """
        return self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=enable_cross_partition_query,
            populate_query_metrics=populate_query_metrics,
        )
