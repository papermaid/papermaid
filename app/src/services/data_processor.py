import asyncio
import json
import time
import zipfile


class DataProcessor:
    def __init__(self, cosmos_db, embeddings_generator):
        self.cosmos_db = cosmos_db
        self.embeddings_generator = embeddings_generator

    @staticmethod
    async def load_and_process_data(zip_file_path, json_file_name):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("../DataSet/Movies/")

        with open(f'../DataSet/Movies/{json_file_name}', 'r') as d:
            data = json.load(d)

        print(f"Loaded {len(data)} items from the dataset")
        return data

    async def generate_vectors(self, items, vector_property):
        for item in items:
            vectorArray = await self.embeddings_generator.generate_embeddings(
                item['overview'])
            item[vector_property] = vectorArray
        print("Done generating vectors")
        return items

    async def insert_data(self, data):
        start_time = time.time()
        counter = 0
        tasks = []
        max_concurrency = 5
        print("Starting document load, please wait...")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def upsert_object(obj):
            nonlocal counter
            async with semaphore:
                await asyncio.get_event_loop().run_in_executor(None,
                                                               self.cosmos_db.upsert_item,
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
        print(
            f"Time taken: {duration:.2f} seconds ({duration:.3f} milliseconds)")
