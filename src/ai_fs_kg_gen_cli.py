#!/usr/bin/env python3
"""
Command-line interface for AI-FS-KG-Gen pipeline
"""
import sys
import os
from pathlib import Path

# Add src to path for proper imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Now we can import everything
from utils.logger import setup_logger, get_logger
from utils.helpers import load_config, save_config
from data_ingestion import TextLoader, ImageLoader, StructuredDataLoader
from data_processing import TextCleaner
from knowledge_extraction import EntityExtractor, RelationExtractor
from knowledge_graph import KnowledgeGraphBuilder
import argparse
import json
from datetime import datetime

def create_default_config():
    """Create a default pipeline configuration"""
    return {
        "input_sources": ["data/"],
        "input_types": ["text"],
        "use_llm": False,
        "use_vlm": False,
        "entity_extraction": True,
        "relation_extraction": True,
        "kg_backend": "networkx",
        "kg_output_format": "json",
        "max_workers": 2,
        "batch_size": 10,
        "output_dir": "output/"
    }

def run_pipeline(config):
    """Run the knowledge graph generation pipeline"""
    print("=== AI-FS-KG-Gen Pipeline ===")
    
    # Setup logging
    setup_logger("INFO", "pipeline.log")
    logger = get_logger(__name__)
    
    print(f"Input sources: {config['input_sources']}")
    print(f"Processing types: {config['input_types']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Initialize components
    print("\nInitializing pipeline components...")
    text_loader = TextLoader()
    entity_extractor = EntityExtractor(model_type="pattern")
    relation_extractor = RelationExtractor(model_type="pattern")
    kg_builder = KnowledgeGraphBuilder(backend=config['kg_backend'])
    
    all_entities = {}
    all_relations = []
    processed_files = 0
    
    # Process input sources
    print(f"\nProcessing input sources...")
    for source in config['input_sources']:
        source_path = Path(source)
        
        if source_path.is_file():
            files_to_process = [source_path]
        elif source_path.is_dir():
            files_to_process = []
            for ext in ['.txt', '.md', '.html']:
                files_to_process.extend(source_path.glob(f'**/*{ext}'))
        else:
            print(f"Warning: Source {source} not found, skipping...")
            continue
        
        for file_path in files_to_process:
            try:
                print(f"  Processing: {file_path.name}")
                
                # Load text
                document = text_loader.load_file(str(file_path))
                text_content = document['content']
                
                # Extract entities
                entities = entity_extractor.extract_entities(text_content)
                
                # Merge entities
                for entity_type, entity_list in entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entity_list)
                
                # Extract relations
                relations = relation_extractor.extract_relations(text_content, entities)
                all_relations.extend(relations)
                
                processed_files += 1
                
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")
                continue
    
    print(f"\nProcessed {processed_files} files")
    
    # Build knowledge graph
    print("Building knowledge graph...")
    
    # Add entities to KG
    entity_count = 0
    for entity_type, entity_list in all_entities.items():
        for entity in entity_list:
            kg_builder.add_entity(
                entity['normalized_text'],
                entity_type,
                {'confidence': entity['confidence'], 'source': 'pattern'}
            )
            entity_count += 1
    
    # Add relations to KG
    relation_count = 0
    for relation in all_relations:
        kg_builder.add_relation(
            relation['subject'],
            relation['predicate'],
            relation['object'],
            {'confidence': relation['confidence'], 'source': relation['source']}
        )
        relation_count += 1
    
    # Get statistics
    stats = kg_builder.get_statistics()
    
    # Display results
    print(f"\n=== Results ===")
    print(f"Files processed: {processed_files}")
    print(f"Entities extracted: {entity_count}")
    print(f"Relations extracted: {relation_count}")
    print(f"KG entities: {stats['entities']}")
    print(f"KG relations: {stats['relations']}")
    print(f"Entity types: {list(stats['entity_types'].keys())}")
    print(f"Relation types: {list(stats['relation_types'].keys())}")
    
    # Export results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export entities
    entities_file = output_dir / f"entities_{timestamp}.json"
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)
    
    # Export relations
    relations_file = output_dir / f"relations_{timestamp}.json"
    with open(relations_file, 'w', encoding='utf-8') as f:
        json.dump(all_relations, f, indent=2, ensure_ascii=False)
    
    # Export KG
    kg_file = output_dir / f"knowledge_graph_{timestamp}.json"
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
    
    # Export statistics
    stats_file = output_dir / f"pipeline_results_{timestamp}.json"
    pipeline_results = {
        "metadata": {
            "timestamp": timestamp,
            "files_processed": processed_files,
            "configuration": config
        },
        "statistics": stats,
        "output_files": {
            "entities": str(entities_file),
            "relations": str(relations_file),
            "knowledge_graph": str(kg_file)
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Output Files ===")
    print(f"Entities: {entities_file}")
    print(f"Relations: {relations_file}")
    print(f"Knowledge Graph: {kg_file}")
    print(f"Pipeline Results: {stats_file}")
    
    print(f"\n🎉 Pipeline completed successfully!")
    return True

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AI-FS-KG-Gen: Food Safety Knowledge Graph Generation Pipeline"
    )
    
    parser.add_argument(
        "input_sources",
        nargs="*",
        default=["data/"],
        help="Input sources (files or directories) to process"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (JSON or YAML)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--kg-backend",
        type=str,
        choices=["networkx", "neo4j", "rdf"],
        default="networkx",
        help="Knowledge graph backend"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Override config with command line arguments
    if args.input_sources:
        config["input_sources"] = args.input_sources
    
    config["output_dir"] = args.output_dir
    config["kg_backend"] = args.kg_backend
    config["batch_size"] = args.batch_size
    config["max_workers"] = args.max_workers
    
    try:
        success = run_pipeline(config)
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())