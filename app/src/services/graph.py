import logging
from neo4j import Driver, GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import networkx as nx
from pyvis.network import Network
# from streamlit.components.v1 import html

from src.services.data_processor import DataProcessor

import config

logger = logging.getLogger("papermaid")

class Entities(BaseModel):
    """Identifying key knowledge entities within a research paper."""
    names: list[str] = Field(
        ...,
        description="Key terms, concepts, theories, or methodologies discussed in the research paper",
    )


class KnownledgeGraphManager:
    def __init__(self, data_processor: DataProcessor):
        self.graph = Neo4jGraph(config.NEO4J_URl, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        self.driver: Driver = GraphDatabase.driver(config.NEO4J_URl, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=config.OPENAI_KEY)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.vector_index: Neo4jVector = Neo4jVector.from_existing_graph(OpenAIEmbeddings(api_key=config.OPENAI_KEY),
                                                            search_type="hybrid",
                                                            node_label="Document",
                                                            text_node_properties=["text"],
                                                            embedding_node_property="embedding"
                                                            )
        self.data_processor = data_processor

    def construct_graph(self, file_path: str):
        """Construct a knowledge graph from a pdf file."""
        try:
            logger.info(f"Constructing graph")
            documents = self.data_processor.pdf_to_document(file_path)
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True,
            )
            logger.info(f"Graph constructed")
        except Exception as e:
            logger.error(f"Error constructing graph: {str(e)}")

    def save_graph(self) -> bool:
        try:
            logger.info("Saving graph...")
            cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
            results = self.driver.session().run(cypher)

            G = nx.MultiDiGraph()
            for record in results:
                source = record["s"]
                target = record["t"]
                rel = record["r"]

                # node properties + edge type as label
                source_label = str(dict(source)["id"])
                target_label = str(dict(target)["id"])
                
                G.add_node(source.element_id, 
                        label=source_label,
                        title=f"Properties: {dict(source)}",
                        size=30,  # Node size
                        font={'size': 16, 'color': 'white'})
                G.add_node(target.element_id, 
                        label=target_label,
                        title=f"Properties: {dict(target)}",
                        size=30,  # Node size
                        font={'size': 16, 'color': 'white'})
                G.add_edge(source.element_id, target.element_id, 
                        title=f"Type: {rel.type}",
                        label=rel.type,
                        width=1,
                        color='#aaaaaa')

                nt = Network(height='750px', width='100%', directed=True, bgcolor="#222222", font_color="white")
                nt.from_nx(G)

                # Appearance
                for node in nt.nodes:
                    node['font'] = {'size': 16, 'color': 'white'}
                    node['borderWidth'] = 2
                    node['color'] = '#1f78b4'

                for edge in nt.edges:
                    edge['font'] = {'size': 14, 'color': 'lightgrey'}
                    edge['width'] = 2

                nt.force_atlas_2based()
                output_file = 'nx.html'
                # nt.show(output_file, notebook=False)
                nt.save_graph(output_file)
                logger.info(f"Graph saved to {output_file}")
                return True
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            return False
        # with open(output_file, 'r') as f:
        #     graph_html = f.read()
        # html(graph_html, height=750)


    def __generate_full_text_query(self, input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str):
        """Extract key entities from a given question."""
        logger.info("Start retrieval of structured data")
        result = ""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting key terms, concepts, theories, or methodologies discussed in the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ])
        entity_chain = prompt | self.llm.with_structured_output(Entities)
        entities = entity_chain.invoke({"question": question})

        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.__generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        logger.info("Finish retrieval of structured data")
        return result
    
    def retriever(self, question: str):
        """Retrieve information from the structured and unstructured data sources."""
        print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data: {structured_data} Unstructured data: {"#Document ". join(unstructured_data)}"""
        return final_data


if __name__ == "__main__":
    kg = KnownledgeGraphManager()
    kg.save_graph()