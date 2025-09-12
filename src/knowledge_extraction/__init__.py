"""Knowledge extraction module for AI-FS-KG-Gen"""

from .entity_extractor import EntityExtractor, extract_entities_batch
from .relation_extractor import RelationExtractor, extract_relations_batch

__all__ = [
    "EntityExtractor",
    "extract_entities_batch",
    "RelationExtractor", 
    "extract_relations_batch"
]
