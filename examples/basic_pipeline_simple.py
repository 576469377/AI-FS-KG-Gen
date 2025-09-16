#!/usr/bin/env python3
"""
Basic pipeline example for AI-FS-KG-Gen

This example demonstrates how to use the pipeline with sample text data.
"""
import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Direct imports to avoid relative import issues
from utils.logger import setup_logger, get_logger
from utils.helpers import load_config, save_config
from data_ingestion import TextLoader
from data_processing import TextCleaner
from knowledge_extraction import EntityExtractor, RelationExtractor  
from knowledge_graph import KnowledgeGraphBuilder

def create_sample_data():
    """Create sample food safety text data"""
    sample_dir = Path(__file__).parent / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    
    # Sample text about food safety
    sample_texts = [
        """
        Escherichia coli (E. coli) bacteria can cause severe foodborne illness. 
        Ground beef contaminated with E. coli O157:H7 has been linked to several outbreaks.
        Proper cooking to an internal temperature of 160°F (71°C) can kill harmful bacteria.
        Raw vegetables should be washed thoroughly to prevent contamination.
        """,
        
        """
        Salmonella is a common cause of food poisoning. Poultry products, eggs, and dairy
        can harbor Salmonella bacteria. The FDA recommends cooking chicken to 165°F (74°C).
        Cross-contamination occurs when raw meat juices contact ready-to-eat foods.
        Proper hand washing and sanitization prevent the spread of pathogens.
        """,
        
        """
        Listeria monocytogenes is particularly dangerous for pregnant women and elderly.
        Soft cheeses, deli meats, and unpasteurized milk products may contain Listeria.
        Refrigeration temperatures below 40°F (4°C) slow bacterial growth but don't eliminate it.
        HACCP (Hazard Analysis Critical Control Points) systems help ensure food safety.
        """
    ]
    
    # Save sample texts
    file_paths = []
    for i, text in enumerate(sample_texts):
        file_path = sample_dir / f"sample_text_{i+1}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        file_paths.append(str(file_path))
    
    return file_paths

def run_simple_pipeline():
    """Run a simplified pipeline without the orchestrator"""
    print("=== AI-FS-KG-Gen Basic Pipeline Example ===\n")
    
    # Setup logging
    setup_logger("INFO", "basic_pipeline_example.log")
    logger = get_logger(__name__)
    
    # Create sample data
    print("Creating sample data...")
    sample_files = create_sample_data()
    print(f"Created {len(sample_files)} sample files")
    
    # Initialize components
    print("\nInitializing pipeline components...")
    text_loader = TextLoader()
    text_cleaner = TextCleaner()
    entity_extractor = EntityExtractor(model_type="pattern")
    relation_extractor = RelationExtractor(model_type="pattern")
    kg_builder = KnowledgeGraphBuilder(backend="networkx")
    
    all_entities = {}
    all_relations = []
    
    # Process each file
    print(f"\nProcessing {len(sample_files)} files...")
    for i, file_path in enumerate(sample_files):
        print(f"  Processing file {i+1}: {Path(file_path).name}")
        
        # Load and clean text
        document = text_loader.load_file(file_path)
        cleaned_text = document['content']
        
        # Extract entities
        entities = entity_extractor.extract_entities(cleaned_text)
        
        # Merge entities
        for entity_type, entity_list in entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = []
            all_entities[entity_type].extend(entity_list)
        
        # Extract relations
        relations = relation_extractor.extract_relations(cleaned_text, entities)
        all_relations.extend(relations)
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    
    # Add entities to KG
    for entity_type, entity_list in all_entities.items():
        for entity in entity_list:
            kg_builder.add_entity(
                entity['normalized_text'], 
                entity_type, 
                {'confidence': entity['confidence'], 'source': 'pattern'}
            )
    
    # Add relations to KG
    for relation in all_relations:
        kg_builder.add_relation(
            relation['subject'],
            relation['predicate'], 
            relation['object'],
            {'confidence': relation['confidence'], 'source': relation['source']}
        )
    
    # Get statistics
    stats = kg_builder.get_statistics()
    
    # Display results
    print(f"\n=== Pipeline Results ===")
    print(f"Total entities: {stats['entities']}")
    print(f"Total relations: {stats['relations']}")
    print(f"Entity types: {list(stats['entity_types'].keys())}")
    print(f"Relation types: {list(stats['relation_types'].keys())}")
    
    # Show sample entities
    print(f"\nSample entities:")
    entity_count = 0
    for entity_type, entity_list in all_entities.items():
        if entity_list and entity_count < 10:
            unique_entities = list({e['normalized_text'] for e in entity_list})
            for entity_text in unique_entities[:3]:
                print(f"  - {entity_text} ({entity_type})")
                entity_count += 1
                if entity_count >= 10:
                    break
    
    # Show sample relations  
    print(f"\nSample relations:")
    for i, relation in enumerate(all_relations[:5]):
        print(f"  - {relation['subject']} → {relation['predicate']} → {relation['object']}")
    
    # Export results
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export entities
    entities_file = output_dir / f"entities_{timestamp}.json"
    import json
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)
    
    # Export relations
    relations_file = output_dir / f"relations_{timestamp}.json"
    with open(relations_file, 'w', encoding='utf-8') as f:
        json.dump(all_relations, f, indent=2, ensure_ascii=False)
    
    # Export KG
    kg_file = output_dir / f"knowledge_graph_{timestamp}.json"
    
    # Create KG export data manually
    kg_data = {
        "metadata": {
            "backend": kg_builder.backend,
            "statistics": stats,
            "timestamp": timestamp
        },
        "entities": kg_builder.query_entities(limit=10000),
        "relations": kg_builder.query_relations(limit=10000)
    }
    
    with open(kg_file, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput files:")
    print(f"  - Entities: {entities_file}")
    print(f"  - Relations: {relations_file}")  
    print(f"  - Knowledge Graph: {kg_file}")
    
    print(f"\n🎉 Pipeline completed successfully!")
    return 0

def main():
    """Main function"""
    try:
        return run_simple_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())