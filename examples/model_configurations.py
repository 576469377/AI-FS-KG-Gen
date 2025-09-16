#!/usr/bin/env python3
"""
AI-FS-KG-Gen Model Configuration Examples

This script demonstrates different model configurations for various use cases.
Run with: python examples/model_configurations.py
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from pipeline import AIFSKGPipeline, PipelineConfig
    from data_processing.llm_processor import LLMProcessor
    from data_processing.vlm_processor import VLMProcessor
    from utils.logger import setup_logger, get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the project root directory")
    sys.exit(1)

# Setup logging
setup_logger()
logger = get_logger(__name__)

def demo_openai_configuration():
    """Demonstrate OpenAI model configuration"""
    print("\n=== OpenAI Configuration Demo ===")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Set it to run OpenAI demos.")
        return None
    
    try:
        # Create OpenAI-based pipeline
        config = PipelineConfig(
            input_sources=["./examples/sample_data/"],
            input_types=["text"],
            
            # OpenAI settings
            use_llm=True,
            use_vlm=False,  # Keep simple for demo
            llm_model="gpt-3.5-turbo",
            llm_provider="openai",
            
            # Processing settings
            entity_extraction=True,
            relation_extraction=True,
            entity_confidence_threshold=0.7,
            relation_confidence_threshold=0.6,
            
            # Output settings
            kg_backend="networkx",
            kg_output_format="json",
            output_dir="./examples/output/openai_demo/",
            
            # Performance
            max_workers=2,
            batch_size=5
        )
        
        # Test with sample text
        sample_text = """
        The FDA issued a recall notice for ground beef products from ABC Meat Company 
        due to potential E. coli O157:H7 contamination. The bacteria can cause severe 
        foodborne illness including bloody diarrhea and kidney failure. Consumers should 
        check their freezers and dispose of any affected products immediately.
        """
        
        # Create processor
        processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
        
        # Extract entities
        entity_result = processor.process_text(sample_text, task="extract_entities")
        print("✓ Entity extraction successful")
        print(f"  Entities found: {entity_result.get('entity_count', 0)}")
        
        # Generate summary
        summary_result = processor.process_text(sample_text, task="summarize")
        print("✓ Summarization successful")
        print(f"  Summary: {summary_result.get('result', '')[:100]}...")
        
        return config
        
    except Exception as e:
        print(f"✗ OpenAI demo failed: {e}")
        return None

def demo_local_model_configuration():
    """Demonstrate local model configuration"""
    print("\n=== Local Model Configuration Demo ===")
    
    try:
        # Create local model configuration
        config = PipelineConfig(
            input_sources=["./examples/sample_data/"],
            input_types=["text"],
            
            # Local model settings
            use_llm=True,
            use_vlm=False,
            llm_model="microsoft/DialoGPT-medium",  # Smaller model for demo
            llm_provider="huggingface",
            
            # Processing settings
            entity_extraction=True,
            relation_extraction=False,  # Simplify for demo
            entity_confidence_threshold=0.6,
            
            # Output settings
            kg_backend="networkx",
            kg_output_format="json",
            output_dir="./examples/output/local_demo/",
            
            # Performance (conservative for local)
            max_workers=1,
            batch_size=2
        )
        
        print("✓ Local model configuration created")
        print(f"  Model: {config.llm_model}")
        print(f"  Provider: {config.llm_provider}")
        
        # Note: Actual model loading would happen here in real usage
        print("  (Model loading skipped in demo)")
        
        return config
        
    except Exception as e:
        print(f"✗ Local model demo failed: {e}")
        return None

def demo_vision_model_configuration():
    """Demonstrate vision model configuration"""
    print("\n=== Vision Model Configuration Demo ===")
    
    try:
        # Create vision-focused configuration
        config = PipelineConfig(
            input_sources=["./examples/sample_images/"],
            input_types=["image"],
            
            # Vision model settings
            use_llm=False,  # Focus on vision
            use_vlm=True,
            vlm_model="Salesforce/blip-image-captioning-base",
            
            # Processing settings
            entity_extraction=False,  # From images
            relation_extraction=False,
            
            # Output settings
            kg_backend="networkx",
            kg_output_format="json",
            output_dir="./examples/output/vision_demo/",
            
            # Performance
            max_workers=2,
            batch_size=4
        )
        
        print("✓ Vision model configuration created")
        print(f"  VLM Model: {config.vlm_model}")
        
        # Note: Image processing would happen here
        print("  (Image processing skipped in demo - no sample images)")
        
        return config
        
    except Exception as e:
        print(f"✗ Vision model demo failed: {e}")
        return None

def demo_budget_configuration():
    """Demonstrate budget-conscious configuration"""
    print("\n=== Budget-Conscious Configuration Demo ===")
    
    try:
        # Create cost-optimized configuration
        config = PipelineConfig(
            input_sources=["./examples/sample_data/"],
            input_types=["text"],  # Text only to reduce costs
            
            # Budget model settings
            use_llm=True,
            use_vlm=False,  # Disable expensive VLM
            llm_model="gpt-3.5-turbo",  # Cheaper than GPT-4
            llm_provider="openai",
            
            # Processing settings (lower quality for cost)
            entity_extraction=True,
            relation_extraction=False,  # Reduce API calls
            entity_confidence_threshold=0.5,  # Lower threshold
            
            # Output settings
            kg_backend="networkx",  # No database costs
            kg_output_format="json",
            output_dir="./examples/output/budget_demo/",
            save_intermediate_results=False,  # Save storage
            
            # Performance (minimize API calls)
            max_workers=1,  # Sequential processing
            batch_size=5   # Small batches
        )
        
        print("✓ Budget configuration created")
        print("  Cost optimization features:")
        print("    - Text processing only")
        print("    - GPT-3.5-turbo (cost-effective)")
        print("    - No VLM processing")
        print("    - No relation extraction")
        print("    - In-memory graph backend")
        print("    - No intermediate result saving")
        
        return config
        
    except Exception as e:
        print(f"✗ Budget demo failed: {e}")
        return None

def demo_high_accuracy_configuration():
    """Demonstrate high-accuracy configuration"""
    print("\n=== High-Accuracy Configuration Demo ===")
    
    try:
        # Create high-accuracy configuration
        config = PipelineConfig(
            input_sources=["./examples/sample_data/"],
            input_types=["text", "image"],
            
            # High-quality model settings
            use_llm=True,
            use_vlm=True,
            llm_model="gpt-4",  # Best OpenAI model
            llm_provider="openai",
            vlm_model="Salesforce/blip-image-captioning-large",
            
            # Processing settings (high quality)
            entity_extraction=True,
            relation_extraction=True,
            entity_confidence_threshold=0.9,  # Very high confidence
            relation_confidence_threshold=0.8,
            
            # Output settings
            kg_backend="neo4j",  # Scalable backend
            kg_output_format="gexf",
            output_dir="./examples/output/accuracy_demo/",
            save_intermediate_results=True,
            
            # Performance (quality over speed)
            max_workers=4,
            batch_size=8
        )
        
        print("✓ High-accuracy configuration created")
        print("  Quality features:")
        print("    - GPT-4 for LLM processing")
        print("    - BLIP-large for VLM processing")
        print("    - High confidence thresholds")
        print("    - Full entity and relation extraction")
        print("    - Neo4j backend for scalability")
        print("    - Intermediate result saving")
        
        return config
        
    except Exception as e:
        print(f"✗ High-accuracy demo failed: {e}")
        return None

def compare_configurations():
    """Compare different configuration approaches"""
    print("\n=== Configuration Comparison ===")
    
    configurations = {
        "Development": {
            "models": "GPT-3.5-turbo",
            "cost": "Low-Medium",
            "speed": "Fast",
            "accuracy": "Good",
            "use_case": "Testing and prototyping"
        },
        "Budget": {
            "models": "GPT-3.5-turbo (text only)",
            "cost": "Low",
            "speed": "Fast",
            "accuracy": "Basic",
            "use_case": "Cost-sensitive deployments"
        },
        "Local": {
            "models": "Hugging Face models",
            "cost": "Free (compute only)",
            "speed": "Medium-Slow",
            "accuracy": "Variable",
            "use_case": "Privacy-sensitive environments"
        },
        "High-Accuracy": {
            "models": "GPT-4 + BLIP-large",
            "cost": "High",
            "speed": "Medium",
            "accuracy": "Excellent",
            "use_case": "Production systems requiring high quality"
        }
    }
    
    print("\n| Configuration | Models | Cost | Speed | Accuracy | Use Case |")
    print("|---------------|--------|------|-------|----------|----------|")
    
    for config_name, details in configurations.items():
        print(f"| {config_name:<13} | {details['models']:<15} | {details['cost']:<8} | {details['speed']:<9} | {details['accuracy']:<12} | {details['use_case']:<20} |")

def show_environment_setup():
    """Show environment setup for different scenarios"""
    print("\n=== Environment Setup Guide ===")
    
    print("\n1. OpenAI Setup:")
    print("   export OPENAI_API_KEY='sk-your-key-here'")
    
    print("\n2. Local Models Setup:")
    print("   pip install torch transformers accelerate")
    print("   export TRANSFORMERS_CACHE='./model_cache'")
    
    print("\n3. Neo4j Setup (for production):")
    print("   export NEO4J_URI='bolt://localhost:7687'")
    print("   export NEO4J_USER='neo4j'")
    print("   export NEO4J_PASSWORD='your-password'")
    
    print("\n4. GPU Setup (for local models):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def create_sample_data():
    """Create sample data for demos"""
    sample_dir = Path("examples/sample_data")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_text = """
Food Safety Alert: Salmonella Outbreak Linked to Fresh Produce

The Centers for Disease Control and Prevention (CDC) and the Food and Drug Administration (FDA) 
are investigating a multistate outbreak of Salmonella infections linked to contaminated lettuce. 
At least 47 people across 12 states have been infected with the outbreak strain of Salmonella Typhimurium.

Key Details:
- Pathogen: Salmonella Typhimurium
- Food Product: Fresh lettuce from ABC Farms
- Affected States: 12 states in the southwestern United States
- Illness Count: 47 confirmed cases
- Hospitalizations: 12 people hospitalized

The contaminated lettuce was distributed to grocery stores and restaurants between March 1-15, 2024. 
Consumers should not eat, and restaurants should not serve, any lettuce from ABC Farms with harvest 
dates between February 25 - March 10, 2024.

Symptoms of Salmonella infection include:
- Diarrhea
- Fever  
- Stomach cramps
- Nausea
- Vomiting

Most people recover without treatment within 4-7 days, but some infections may require hospitalization. 
Young children, elderly adults, and people with weakened immune systems are at higher risk for severe illness.

Prevention measures:
- Wash hands thoroughly with soap and water
- Rinse fresh produce under running water
- Keep raw foods separate from ready-to-eat foods
- Cook foods to proper internal temperatures
- Refrigerate perishable foods within 2 hours

For more information, contact the CDC at 1-800-CDC-INFO or visit www.cdc.gov/foodsafety.
"""
    
    with open(sample_dir / "food_safety_alert.txt", "w") as f:
        f.write(sample_text)
    
    print(f"✓ Sample data created in {sample_dir}")

def main():
    """Main demonstration function"""
    print("AI-FS-KG-Gen Model Configuration Examples")
    print("=" * 45)
    
    # Create sample data
    create_sample_data()
    
    # Environment setup guide
    show_environment_setup()
    
    # Run configuration demos
    configs = {}
    
    # OpenAI demo
    configs["openai"] = demo_openai_configuration()
    
    # Local model demo
    configs["local"] = demo_local_model_configuration()
    
    # Vision model demo
    configs["vision"] = demo_vision_model_configuration()
    
    # Budget demo
    configs["budget"] = demo_budget_configuration()
    
    # High-accuracy demo
    configs["accuracy"] = demo_high_accuracy_configuration()
    
    # Configuration comparison
    compare_configurations()
    
    # Save successful configurations
    output_dir = Path("examples/output/configurations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in configs.items():
        if config:
            config_dict = {
                "input_sources": config.input_sources,
                "input_types": config.input_types,
                "use_llm": config.use_llm,
                "use_vlm": config.use_vlm,
                "llm_model": config.llm_model,
                "llm_provider": getattr(config, 'llm_provider', 'openai'),
                "vlm_model": config.vlm_model,
                "entity_extraction": config.entity_extraction,
                "relation_extraction": config.relation_extraction,
                "entity_confidence_threshold": config.entity_confidence_threshold,
                "relation_confidence_threshold": config.relation_confidence_threshold,
                "kg_backend": config.kg_backend,
                "kg_output_format": config.kg_output_format,
                "max_workers": config.max_workers,
                "batch_size": config.batch_size
            }
            
            with open(output_dir / f"{name}_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
    
    print(f"\n✓ Configuration examples saved to {output_dir}")
    
    print("\n" + "=" * 45)
    print("Demo completed! Check the documentation for more details:")
    print("- docs/model_support.md - Complete model overview")
    print("- docs/llm_integration.md - LLM setup guide")
    print("- docs/configuration_examples.md - Deployment scenarios")
    print("- docs/troubleshooting.md - Common issues and solutions")

if __name__ == "__main__":
    main()