"""
Knowledge graph construction and management for AI-FS-KG-Gen pipeline
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import json
from pathlib import Path
from neo4j import GraphDatabase
import py2neo
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import get_kg_config, FOOD_SAFETY_ENTITIES, FOOD_SAFETY_RELATIONS
from utils.logger import get_logger
from utils.helpers import normalize_entity, generate_hash, safe_filename

logger = get_logger(__name__)

class KnowledgeGraphBuilder:
    """
    Knowledge graph builder for food safety domain
    """
    
    def __init__(self, backend: str = "networkx", config: Optional[Dict] = None):
        """
        Initialize knowledge graph builder
        
        Args:
            backend: Backend to use (networkx, neo4j, rdf)
            config: Optional configuration overrides
        """
        self.backend = backend
        self.config = config or get_kg_config()
        
        self._init_backend()
        
        # Statistics
        self.stats = {
            "entities": 0,
            "relations": 0,
            "entity_types": {},
            "relation_types": {}
        }
        
        logger.info(f"KnowledgeGraphBuilder initialized with {backend} backend")
    
    def _init_backend(self):
        """Initialize the knowledge graph backend"""
        if self.backend == "networkx":
            self.graph = nx.MultiDiGraph()
            
        elif self.backend == "neo4j":
            try:
                self.driver = GraphDatabase.driver(
                    self.config["database_url"],
                    auth=(self.config["username"], self.config["password"])
                )
                self.graph = py2neo.Graph(
                    host=self.config["database_url"].split("//")[1].split(":")[0],
                    user=self.config["username"],
                    password=self.config["password"]
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                logger.info("Falling back to NetworkX backend")
                self.backend = "networkx"
                self.graph = nx.MultiDiGraph()
                
        elif self.backend == "rdf":
            self.graph = Graph()
            # Define namespaces
            self.fs_ns = Namespace("http://foodsafety.org/ontology/")
            self.graph.bind("fs", self.fs_ns)
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def add_entity(self, entity: str, entity_type: str, properties: Optional[Dict] = None) -> str:
        """
        Add an entity to the knowledge graph
        
        Args:
            entity: Entity name
            entity_type: Type of entity
            properties: Additional properties
        
        Returns:
            Entity ID
        """
        entity_normalized = normalize_entity(entity)
        entity_id = self._generate_entity_id(entity_normalized, entity_type)
        
        properties = properties or {}
        properties.update({
            "name": entity,
            "normalized_name": entity_normalized,
            "type": entity_type,
            "id": entity_id
        })
        
        if self.backend == "networkx":
            self.graph.add_node(entity_id, **properties)
            
        elif self.backend == "neo4j":
            try:
                # Create node with label as entity type
                node = py2neo.Node(entity_type, **properties)
                self.graph.create(node)
            except Exception as e:
                logger.warning(f"Failed to create Neo4j node: {e}")
                
        elif self.backend == "rdf":
            entity_uri = self.fs_ns[entity_id]
            self.graph.add((entity_uri, RDF.type, self.fs_ns[entity_type]))
            self.graph.add((entity_uri, RDFS.label, Literal(entity)))
            
            for prop, value in properties.items():
                if prop not in ["name", "type", "id"]:
                    self.graph.add((entity_uri, self.fs_ns[prop], Literal(value)))
        
        # Update statistics
        self.stats["entities"] += 1
        if entity_type not in self.stats["entity_types"]:
            self.stats["entity_types"][entity_type] = 0
        self.stats["entity_types"][entity_type] += 1
        
        return entity_id
    
    def add_relation(self, subject: str, predicate: str, obj: str, 
                    properties: Optional[Dict] = None) -> str:
        """
        Add a relation to the knowledge graph
        
        Args:
            subject: Subject entity
            predicate: Relation type
            obj: Object entity
            properties: Additional properties
        
        Returns:
            Relation ID
        """
        # Normalize entities
        subject_norm = normalize_entity(subject)
        obj_norm = normalize_entity(obj)
        
        # Generate entity IDs
        subject_id = self._find_or_create_entity(subject_norm)
        obj_id = self._find_or_create_entity(obj_norm)
        
        properties = properties or {}
        relation_id = f"{subject_id}_{predicate}_{obj_id}_{generate_hash(str(properties))[:8]}"
        
        properties.update({
            "relation_type": predicate,
            "id": relation_id
        })
        
        if self.backend == "networkx":
            self.graph.add_edge(subject_id, obj_id, key=relation_id, **properties)
            
        elif self.backend == "neo4j":
            try:
                # Find or create nodes
                subject_node = self.graph.nodes.match(normalized_name=subject_norm).first()
                obj_node = self.graph.nodes.match(normalized_name=obj_norm).first()
                
                if subject_node and obj_node:
                    # Create relationship
                    rel = py2neo.Relationship(subject_node, predicate.upper(), obj_node, **properties)
                    self.graph.create(rel)
            except Exception as e:
                logger.warning(f"Failed to create Neo4j relationship: {e}")
                
        elif self.backend == "rdf":
            subject_uri = self.fs_ns[subject_id]
            obj_uri = self.fs_ns[obj_id]
            predicate_uri = self.fs_ns[predicate]
            
            self.graph.add((subject_uri, predicate_uri, obj_uri))
        
        # Update statistics
        self.stats["relations"] += 1
        if predicate not in self.stats["relation_types"]:
            self.stats["relation_types"][predicate] = 0
        self.stats["relation_types"][predicate] += 1
        
        return relation_id
    
    def _generate_entity_id(self, entity: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        clean_entity = entity.replace(" ", "_").replace("-", "_").lower()
        return f"{entity_type}_{clean_entity}_{generate_hash(entity)[:8]}"
    
    def _find_or_create_entity(self, entity: str, entity_type: str = "unknown") -> str:
        """Find existing entity or create new one"""
        entity_id = self._generate_entity_id(entity, entity_type)
        
        if self.backend == "networkx":
            if entity_id not in self.graph.nodes:
                self.add_entity(entity, entity_type)
        elif self.backend == "neo4j":
            try:
                existing = self.graph.nodes.match(normalized_name=entity).first()
                if not existing:
                    self.add_entity(entity, entity_type)
            except Exception as e:
                logger.warning(f"Error checking Neo4j entity: {e}")
        
        return entity_id
    
    def build_from_extractions(self, entities: Dict[str, List[Dict]], 
                              relations: List[Dict]) -> Dict[str, Any]:
        """
        Build knowledge graph from extracted entities and relations
        
        Args:
            entities: Extracted entities by type
            relations: Extracted relations
        
        Returns:
            Build statistics
        """
        logger.info("Building knowledge graph from extractions")
        
        # Add entities
        entity_map = {}
        for entity_type, entity_list in entities.items():
            for entity_info in entity_list:
                entity_text = entity_info.get("text", "")
                confidence = entity_info.get("confidence", 0.0)
                
                if confidence > 0.5:  # Filter low confidence entities
                    properties = {
                        "confidence": confidence,
                        "source": entity_info.get("label", "unknown")
                    }
                    
                    entity_id = self.add_entity(entity_text, entity_type, properties)
                    entity_map[entity_text] = entity_id
        
        # Add relations
        for relation in relations:
            subject = relation.get("subject", "")
            predicate = relation.get("predicate", "")
            obj = relation.get("object", "")
            confidence = relation.get("confidence", 0.0)
            
            if confidence > 0.5 and subject and predicate and obj:
                properties = {
                    "confidence": confidence,
                    "source": relation.get("source", "unknown")
                }
                
                self.add_relation(subject, predicate, obj, properties)
        
        build_stats = {
            "entities_added": len(entity_map),
            "relations_added": len([r for r in relations if r.get("confidence", 0) > 0.5]),
            "total_entities": self.stats["entities"],
            "total_relations": self.stats["relations"]
        }
        
        logger.info(f"Built knowledge graph: {build_stats}")
        return build_stats
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities from the knowledge graph
        
        Args:
            entity_type: Filter by entity type
            limit: Maximum number of results
        
        Returns:
            List of entities
        """
        results = []
        
        if self.backend == "networkx":
            nodes = list(self.graph.nodes(data=True))
            
            if entity_type:
                nodes = [(n, data) for n, data in nodes if data.get("type") == entity_type]
            
            for node_id, data in nodes[:limit]:
                results.append({
                    "id": node_id,
                    "name": data.get("name", ""),
                    "type": data.get("type", ""),
                    "properties": data
                })
                
        elif self.backend == "neo4j":
            try:
                if entity_type:
                    nodes = list(self.graph.nodes.match(entity_type).limit(limit))
                else:
                    nodes = list(self.graph.nodes.limit(limit))
                
                for node in nodes:
                    results.append({
                        "id": node.identity,
                        "name": node.get("name", ""),
                        "type": list(node.labels)[0] if node.labels else "",
                        "properties": dict(node)
                    })
            except Exception as e:
                logger.warning(f"Neo4j query failed: {e}")
                
        elif self.backend == "rdf":
            query = """
            SELECT ?entity ?name ?type WHERE {
                ?entity rdf:type ?type .
                ?entity rdfs:label ?name .
            }
            """
            
            if entity_type:
                query = f"""
                SELECT ?entity ?name WHERE {{
                    ?entity rdf:type fs:{entity_type} .
                    ?entity rdfs:label ?name .
                }}
                """
            
            for row in self.graph.query(query)[:limit]:
                results.append({
                    "id": str(row.entity),
                    "name": str(row.name),
                    "type": entity_type or str(row.type),
                    "properties": {}
                })
        
        return results
    
    def query_relations(self, subject: Optional[str] = None, 
                       predicate: Optional[str] = None,
                       obj: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relations from the knowledge graph
        
        Args:
            subject: Filter by subject
            predicate: Filter by predicate
            obj: Filter by object
            limit: Maximum number of results
        
        Returns:
            List of relations
        """
        results = []
        
        if self.backend == "networkx":
            edges = list(self.graph.edges(data=True, keys=True))
            
            for subj_id, obj_id, key, data in edges[:limit]:
                # Get node names
                subj_name = self.graph.nodes[subj_id].get("name", subj_id)
                obj_name = self.graph.nodes[obj_id].get("name", obj_id)
                
                results.append({
                    "subject": subj_name,
                    "predicate": data.get("relation_type", ""),
                    "object": obj_name,
                    "properties": data
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if self.backend == "networkx":
            self.stats.update({
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            })
        
        return self.stats
    
    def export_graph(self, output_path: str, format: str = "json") -> None:
        """
        Export knowledge graph to file
        
        Args:
            output_path: Output file path
            format: Export format (json, gexf, rdf)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._export_json(output_path)
        elif format == "gexf" and self.backend == "networkx":
            nx.write_gexf(self.graph, output_path)
        elif format == "rdf" and self.backend == "rdf":
            self.graph.serialize(destination=output_path, format="ttl")
        else:
            raise ValueError(f"Unsupported export format: {format} for backend: {self.backend}")
        
        logger.info(f"Exported knowledge graph to {output_path}")
    
    def _export_json(self, output_path: Path) -> None:
        """Export graph as JSON"""
        export_data = {
            "metadata": {
                "backend": self.backend,
                "statistics": self.get_statistics(),
                "entity_types": FOOD_SAFETY_ENTITIES,
                "relation_types": FOOD_SAFETY_RELATIONS
            },
            "entities": self.query_entities(limit=10000),
            "relations": self.query_relations(limit=10000)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

def merge_knowledge_graphs(graphs: List[KnowledgeGraphBuilder], 
                          output_backend: str = "networkx") -> KnowledgeGraphBuilder:
    """
    Merge multiple knowledge graphs into one
    
    Args:
        graphs: List of knowledge graphs to merge
        output_backend: Backend for merged graph
    
    Returns:
        Merged knowledge graph
    """
    merged = KnowledgeGraphBuilder(backend=output_backend)
    
    for graph in graphs:
        entities = graph.query_entities(limit=100000)
        relations = graph.query_relations(limit=100000)
        
        # Add entities
        for entity in entities:
            merged.add_entity(
                entity["name"],
                entity["type"],
                entity.get("properties", {})
            )
        
        # Add relations
        for relation in relations:
            merged.add_relation(
                relation["subject"],
                relation["predicate"],
                relation["object"],
                relation.get("properties", {})
            )
    
    logger.info(f"Merged {len(graphs)} knowledge graphs")
    return merged