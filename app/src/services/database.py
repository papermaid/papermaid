import logging

from azure.cosmos import container, CosmosClient, PartitionKey, exceptions

import config


logger = logging.getLogger('papermaid')

class CosmosDB:
    def __init__(self):
        self.client = CosmosClient(url=config.COSMOS_CONN,
                                   credential=config.COSMOS_KEY)
        self.db = self.client.create_database_if_not_exists(
            config.COSMOS_DATABASE)
        self.container = self.create_container()

    def create_container(self) -> container.ContainerProxy | None:
        """
        Create a Cosmos DB container.

        :return: The created container proxy.
        :raises: CosmosHttpResponseError if the container cannot be created.
        """
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/" + config.COSMOS_VECTOR_PROPERTY,
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": config.OPENAI_EMBEDDINGS_DIMENSIONS
                },
            ]
        }

        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/\"_etag\"/?"},
                {"path": "/" + config.COSMOS_VECTOR_PROPERTY + "/*"},
            ],
            "vectorIndexes": [
                {
                    "path": "/" + config.COSMOS_VECTOR_PROPERTY,
                    "type": "quantizedFlat"
                }
            ]
        }

        try:
            container = self.db.create_container_if_not_exists(
                id=config.COSMOS_COLLECTION,
                partition_key=PartitionKey(path='/partitionKey'),
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
                offer_throughput=1000
            )
            logger.info(f'Container with id \'{container.id}\' created')
            return container
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Error creating containers: {e}")
            raise

    def upsert_item(self, item):
        """
        Insert or update the specified item.

        If the item already exists in the container, it is replaced.
        If the item does not already exist, it is inserted.
        """
        return self.container.upsert_item(body=item)

    def query_items(self, query, parameters, enable_cross_partition_query=True,
                    populate_query_metrics=True):
        """
        Return all results matching the given query.

        You can use any value for the container name in the FROM clause, but often the container name is used.
        In the examples below, the container name is "products,"
        and is aliased as "p" for easier referencing in the WHERE clause.
        """
        return self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=enable_cross_partition_query,
            populate_query_metrics=populate_query_metrics
        )
