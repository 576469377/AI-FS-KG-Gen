#!/usr/bin/env python3
"""
Complete working example demonstrating AI-FS-KG-Gen capabilities

This example shows how to:
1. Ingest multi-source data
2. Process with pattern-based extraction (no external dependencies)
3. Build a knowledge graph
4. Export results
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import AIFSKGPipeline, PipelineConfig
import json

def create_comprehensive_sample_data():
    """Create diverse sample data for testing"""
    sample_dir = Path(__file__).parent / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    
    # Enhanced food safety texts
    texts = [
        {
            "filename": "bacterial_contamination.txt",
            "content": """
            Escherichia coli O157:H7 contamination in ground beef represents a serious public health threat.
            The pathogen can survive at refrigeration temperatures of 4°C for extended periods.
            Proper cooking to internal temperature of 71°C (160°F) effectively destroys E. coli bacteria.
            Cross-contamination occurs when raw meat juices contact ready-to-eat foods.
            Hand washing with soap for 20 seconds prevents pathogen transmission.
            The FDA requires HACCP implementation in meat processing facilities.
            """
        },
        {
            "filename": "salmonella_outbreak.txt", 
            "content": """
            Salmonella enteritidis outbreak linked to contaminated eggs affected 200 people.
            Poultry products must be cooked to minimum internal temperature of 74°C (165°F).
            The CDC recommends pasteurization of egg products to eliminate Salmonella bacteria.
            Refrigeration below 4°C slows bacterial growth but does not eliminate pathogens.
            Sanitization with chlorine solution at 200 ppm concentration kills Salmonella.
            """
        },
        {
            "filename": "listeria_prevention.txt",
            "content": """
            Listeria monocytogenes poses significant risk to pregnant women and immunocompromised individuals.
            Ready-to-eat deli meats and soft cheeses frequently harbor Listeria bacteria.
            The pathogen can multiply at refrigeration temperatures unlike other foodborne bacteria.
            Heat treatment to 74°C for 15 seconds destroys Listeria in processed foods.
            Environmental monitoring programs detect Listeria in food processing facilities.
            """
        },
        {
            "filename": "food_allergens.txt",
            "content": """
            Eight major allergens account for 90% of food allergic reactions: milk, eggs, fish, shellfish, tree nuts, peanuts, wheat, and soy.
            Peanut allergen can cause severe anaphylactic reactions requiring epinephrine treatment.
            Cross-contact during processing introduces allergens into otherwise safe products.
            The FDA mandates allergen labeling on packaged food products.
            Cleaning protocols remove allergen residues from shared equipment.
            """
        }
    ]
    
    # Create text files
    for text_info in texts:
        file_path = sample_dir / text_info["filename"] 
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_info["content"].strip())
    
    # Create structured data (CSV format)
    csv_content = """product,pathogen,temperature_c,safety_measure
Ground Beef,E. coli O157:H7,71,Proper cooking
Eggs,Salmonella enteritidis,74,Pasteurization
Deli Meat,Listeria monocytogenes,74,Heat treatment
Milk,Various pathogens,63,Pasteurization
Chicken,Salmonella,74,Thorough cooking
Fish,Vibrio vulnificus,63,Adequate cooking
Shellfish,Norovirus,90,Proper cooking
Lettuce,E. coli,N/A,Washing
"""
    
    csv_path = sample_dir / "food_safety_data.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    # Create JSON data
    json_data = {
        "food_safety_incidents": [
            {
                "date": "2023-01-15",
                "pathogen": "Salmonella",
                "product": "ground turkey",
                "cases": 358,
                "states_affected": 28,
                "temperature_control": "inadequate"
            },
            {
                "date": "2023-03-22", 
                "pathogen": "E. coli O157:H7",
                "product": "romaine lettuce",
                "cases": 210,
                "states_affected": 16,
                "source": "irrigation water"
            },
            {
                "date": "2023-06-10",
                "pathogen": "Listeria monocytogenes", 
                "product": "ice cream",
                "cases": 23,
                "states_affected": 8,
                "facility_issue": "environmental contamination"
            }
        ]
    }
    
    json_path = sample_dir / "outbreak_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created comprehensive sample data in {sample_dir}")
    return str(sample_dir)

def run_comprehensive_pipeline():
    """Run comprehensive pipeline demonstration"""
    print("=== AI-FS-KG-Gen Comprehensive Pipeline Demo ===\n")
    
    # Create sample data
    sample_dir = create_comprehensive_sample_data()
    
    # Configure pipeline for maximum extraction without external dependencies
    config = PipelineConfig(
        input_sources=[sample_dir],
        input_types=["text", "structured"],  # Include structured data
        use_llm=False,  # Disable to avoid API dependencies
        use_vlm=False,  # Disable to avoid model downloads
        entity_extraction=True,
        relation_extraction=True,
        entity_confidence_threshold=0.5,  # Lower threshold for more extraction
        relation_confidence_threshold=0.5,
        kg_backend="networkx",
        kg_output_format="json",
        save_intermediate_results=True,
        max_workers=2,
        batch_size=5
    )
    
    try:
        # Initialize and run pipeline
        print("Initializing pipeline...")
        pipeline = AIFSKGPipeline(config)
        
        print("Running pipeline...")
        results = pipeline.run()
        
        # Display comprehensive results
        print("\n🎉 Pipeline completed successfully!\n")
        
        print("=" * 50)
        print("PIPELINE STATISTICS")
        print("=" * 50)
        
        stats = results['statistics']
        print(f"📊 Data items processed: {stats['data_items_processed']}")
        print(f"🔍 Total entities extracted: {stats['total_entities']}")
        print(f"🔗 Total relations extracted: {stats['total_relations']}")
        
        print(f"\n📋 Entity types found: {len(stats['entity_types'])}")
        for entity_type in stats['entity_types']:
            count = len(results['entities'].get(entity_type, []))
            print(f"  • {entity_type}: {count} entities")
        
        print(f"\n🔗 Relation types found: {len(stats['relation_types'])}")
        for relation_type in set(stats['relation_types']):
            count = sum(1 for r in results['relations'] if r.get('predicate') == relation_type)
            print(f"  • {relation_type}: {count} relations")
        
        print("\n" + "=" * 50)
        print("KNOWLEDGE GRAPH ANALYSIS")
        print("=" * 50)
        
        kg_stats = stats['knowledge_graph_stats']
        print(f"🕸️  Graph nodes: {kg_stats.get('nodes', 0)}")
        print(f"🔗 Graph edges: {kg_stats.get('edges', 0)}")
        
        if kg_stats.get('nodes', 0) > 0:
            density = kg_stats.get('edges', 0) / max(kg_stats.get('nodes', 1) * (kg_stats.get('nodes', 1) - 1), 1)
            print(f"📈 Graph density: {density:.4f}")
        
        print("\n" + "=" * 50)
        print("SAMPLE EXTRACTED KNOWLEDGE")
        print("=" * 50)
        
        # Show sample entities by type
        print("\n🧬 Sample Entities:")
        for entity_type, entities in list(results['entities'].items())[:5]:
            sample_entities = [e['text'] for e in entities[:3]]
            print(f"  {entity_type}: {', '.join(sample_entities)}")
            if len(entities) > 3:
                print(f"    ... and {len(entities) - 3} more")
        
        # Show sample relations
        print("\n🔗 Sample Relations:")
        for i, relation in enumerate(results['relations'][:8]):
            subject = relation['subject'][:30] + "..." if len(relation['subject']) > 30 else relation['subject']
            obj = relation['object'][:30] + "..." if len(relation['object']) > 30 else relation['object']
            print(f"  {i+1}. {subject} --{relation['predicate']}--> {obj}")
            print(f"     (confidence: {relation.get('confidence', 0):.2f})")
        
        if len(results['relations']) > 8:
            print(f"    ... and {len(results['relations']) - 8} more relations")
        
        print("\n" + "=" * 50)
        print("OUTPUT FILES")
        print("=" * 50)
        
        output_dir = Path(config.output_dir)
        output_files = list(output_dir.glob("*"))
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Generated files: {len(output_files)}")
        
        for file_path in sorted(output_files):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"  • {file_path.name} ({size_kb:.1f} KB)")
        
        # Show a snippet of the knowledge graph export
        kg_files = list(output_dir.glob("knowledge_graph_*.json"))
        if kg_files:
            print(f"\n📊 Knowledge Graph Sample (from {kg_files[0].name}):")
            with open(kg_files[0], 'r') as f:
                kg_data = json.load(f)
                sample_entities = kg_data.get('entities', [])[:3]
                sample_relations = kg_data.get('relations', [])[:3]
                
                print("  Entities:")
                for entity in sample_entities:
                    print(f"    - {entity.get('name', 'N/A')} ({entity.get('type', 'unknown')})")
                
                print("  Relations:")
                for relation in sample_relations:
                    print(f"    - {relation.get('subject', 'N/A')} → {relation.get('object', 'N/A')}")
        
        print("\n" + "=" * 50)
        print("USAGE RECOMMENDATIONS") 
        print("=" * 50)
        
        print("🚀 To enhance this pipeline further:")
        print("  1. Install spaCy model: python -m spacy download en_core_web_sm")
        print("  2. Add OpenAI API key for LLM processing")
        print("  3. Include image data for VLM analysis")
        print("  4. Use Neo4j for large-scale graph storage")
        print("  5. Fine-tune entity extraction for your domain")
        
        print(f"\n✅ Complete results saved in: {config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧬 AI-FS-KG-Gen - Food Safety Knowledge Graph Generator")
    print("=" * 60)
    
    success = run_comprehensive_pipeline()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("📖 Check docs/usage.md for advanced configuration options")
        print("🔧 See examples/ directory for more pipeline examples")
    else:
        print("\n❌ Demo failed - check error messages above")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()