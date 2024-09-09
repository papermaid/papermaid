from azure.cosmos import CosmosClient, PartitionKey, exceptions

import config


class CosmosDB:
    def __init__(self):
        self.client = CosmosClient(url=config.COSMOS_CONN,
                                   credential=config.COSMOS_KEY)
        self.db = self.client.create_database_if_not_exists(
            config.COSMOS_DATABASE)
        self.container = self.create_container()

    def create_container(self):
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
            print(f'Container with id \'{container.id}\' created')
            return container
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error creating containers: {e}")
            raise

    def upsert_item(self, item):
        return self.container.upsert_item(body=item)

    def query_items(self, query, parameters, enable_cross_partition_query=True,
                    populate_query_metrics=True):
        return self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=enable_cross_partition_query,
            populate_query_metrics=populate_query_metrics
        )
