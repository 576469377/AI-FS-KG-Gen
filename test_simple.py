#!/usr/bin/env python3
"""
Simplified test of the pipeline components
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_simple_extraction():
    """Test simple pattern-based extraction"""
    print("Testing simple pattern-based extraction...")
    
    try:
        from knowledge_extraction import EntityExtractor, RelationExtractor
        
        # Create extractors
        entity_extractor = EntityExtractor(model_type="pattern") 
        relation_extractor = RelationExtractor(model_type="pattern")
        
        test_text = """
        Salmonella bacteria contamination in chicken products can cause foodborne illness. 
        The temperature should be maintained at 4°C to prevent bacterial growth.
        """
        
        print(f"Test text: {test_text.strip()}")
        
        # Extract entities using patterns only
        entities = entity_extractor.extract_entities(test_text)
        print(f"Extracted entities: {entities}")
        
        # Extract relations
        relations = relation_extractor.extract_relations(test_text, entities)
        print(f"Extracted relations: {relations}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_kg_builder():
    """Test knowledge graph builder"""
    print("\nTesting knowledge graph builder...")
    
    try:
        from knowledge_graph import KnowledgeGraphBuilder
        
        kg = KnowledgeGraphBuilder(backend="networkx")
        
        # Add some test entities and relations
        kg.add_entity("Salmonella", "pathogen", {"description": "bacteria"})
        kg.add_entity("chicken", "food_product", {"category": "poultry"})
        kg.add_relation("chicken", "contaminated_with", "Salmonella")
        
        stats = kg.get_statistics()
        print(f"KG statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=== Simple Component Test ===")
    
    test1 = test_simple_extraction()
    test2 = test_kg_builder()
    
    if test1 and test2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

if __name__ == "__main__":
    main()