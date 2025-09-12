#!/usr/bin/env python3
"""
Basic pipeline example for AI-FS-KG-Gen

This example demonstrates how to use the pipeline with sample text data.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import AIFSKGPipeline, PipelineConfig

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
    for i, text in enumerate(sample_texts):
        file_path = sample_dir / f"sample_text_{i+1}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
    
    print(f"Created sample data in {sample_dir}")
    return str(sample_dir)

def run_basic_pipeline():
    """Run basic pipeline example"""
    print("=== AI-FS-KG-Gen Basic Pipeline Example ===\n")
    
    # Create sample data
    sample_dir = create_sample_data()
    
    # Create pipeline configuration
    config = PipelineConfig(
        input_sources=[sample_dir],
        input_types=["text"],
        use_llm=False,  # Disable LLM to avoid API requirements
        use_vlm=False,  # Disable VLM to avoid model downloads
        entity_extraction=True,
        relation_extraction=True,
        kg_backend="networkx",
        kg_output_format="json",
        max_workers=2,
        batch_size=5
    )
    
    try:
        # Initialize and run pipeline
        pipeline = AIFSKGPipeline(config)
        results = pipeline.run()
        
        # Display results
        print("Pipeline completed successfully!\n")
        
        print("=== Statistics ===")
        stats = results['statistics']
        print(f"Data items processed: {stats['data_items_processed']}")
        print(f"Total entities extracted: {stats['total_entities']}")
        print(f"Total relations extracted: {stats['total_relations']}")
        print(f"Entity types: {', '.join(stats['entity_types'])}")
        print(f"Relation types: {', '.join(stats['relation_types'])}")
        
        print(f"\n=== Knowledge Graph Statistics ===")
        kg_stats = stats['knowledge_graph_stats']
        print(f"Nodes: {kg_stats.get('nodes', 0)}")
        print(f"Edges: {kg_stats.get('edges', 0)}")
        
        print(f"\n=== Sample Entities ===")
        for entity_type, entities in list(results['entities'].items())[:3]:
            print(f"{entity_type}: {[e['text'] for e in entities[:5]]}")
        
        print(f"\n=== Sample Relations ===")
        for relation in results['relations'][:5]:
            print(f"{relation['subject']} --{relation['predicate']}--> {relation['object']}")
        
        print(f"\nOutput files saved to: {config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

def run_advanced_example():
    """Run advanced pipeline example with all features"""
    print("\n=== Advanced Pipeline Example ===")
    print("This example requires API keys and additional model downloads")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OpenAI API key not found. Set OPENAI_API_KEY environment variable for LLM features.")
        return False
    
    sample_dir = create_sample_data()
    
    config = PipelineConfig(
        input_sources=[sample_dir],
        input_types=["text"],
        use_llm=True,  # Enable LLM processing
        use_vlm=False,  # Keep VLM disabled for now
        llm_model="gpt-3.5-turbo",
        entity_extraction=True,
        relation_extraction=True,
        kg_backend="networkx",
        kg_output_format="json",
        save_intermediate_results=True
    )
    
    try:
        pipeline = AIFSKGPipeline(config)
        results = pipeline.run()
        
        print("Advanced pipeline completed successfully!")
        print(f"Results with LLM processing saved to: {config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Advanced pipeline failed: {str(e)}")
        return False

def main():
    """Main function"""
    print("AI-FS-KG-Gen Pipeline Examples")
    print("=" * 40)
    
    # Run basic example
    success = run_basic_pipeline()
    
    if success:
        print("\n" + "=" * 40)
        response = input("Run advanced example with LLM? (y/n): ")
        if response.lower() == 'y':
            run_advanced_example()
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()