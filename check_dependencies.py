#!/usr/bin/env python3
"""
Dependency checker for AI-FS-KG-Gen
Checks which optional dependencies are available and provides recommendations
"""

import sys
import importlib.util

def check_dependency(name, import_name=None):
    """Check if a dependency is available"""
    import_name = import_name or name
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def main():
    """Check all dependencies and provide status report"""
    print("🧬 AI-FS-KG-Gen Dependency Status")
    print("=" * 50)
    
    # Core dependencies (required)
    core_deps = [
        ("loguru", None, "Logging system"),
        ("beautifulsoup4", "bs4", "HTML parsing"),  
        ("pillow", "PIL", "Image processing"),
        ("networkx", None, "Knowledge graph backend"),
    ]
    
    # Optional dependencies
    optional_deps = [
        ("pandas", None, "Excel/CSV processing"),
        ("numpy", None, "Numerical computing"),
        ("spacy", None, "Advanced NLP"),
        ("openai", None, "OpenAI LLM integration"),
        ("transformers", None, "Hugging Face models"),
        ("torch", None, "PyTorch models"),
        ("neo4j", None, "Neo4j graph database"),
        ("rdflib", None, "RDF/semantic web"),
    ]
    
    print("\n📋 Core Dependencies:")
    core_available = 0
    for name, import_name, desc in core_deps:
        available = check_dependency(name, import_name)
        status = "✅" if available else "❌"
        print(f"  {status} {name:<15} - {desc}")
        if available:
            core_available += 1
    
    print(f"\nCore dependencies: {core_available}/{len(core_deps)} available")
    
    print("\n🔧 Optional Dependencies:")
    optional_available = 0
    for name, import_name, desc in optional_deps:
        available = check_dependency(name, import_name)
        status = "✅" if available else "⚪" 
        print(f"  {status} {name:<15} - {desc}")
        if available:
            optional_available += 1
    
    print(f"\nOptional dependencies: {optional_available}/{len(optional_deps)} available")
    
    # Overall status
    print("\n" + "=" * 50)
    if core_available == len(core_deps):
        print("🎉 All core dependencies available! You can use the pipeline.")
        
        if optional_available > 0:
            print(f"💡 {optional_available} optional features available")
        else:
            print("💡 No optional features available (basic functionality only)")
            
        print("\n🚀 Quick test:")
        print("   python test_basic.py")
        print("   python examples/basic_pipeline_simple.py")
        
    else:
        print("⚠️  Some core dependencies missing!")
        print("   Install with: pip install -r requirements-minimal.txt")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not check_dependency("pandas"):
        print("   • Install pandas for CSV/Excel support: pip install pandas")
    if not check_dependency("spacy"):
        print("   • Install spaCy for better NLP: pip install spacy && python -m spacy download en_core_web_sm")
    if not check_dependency("openai"):
        print("   • Install OpenAI for LLM features: pip install openai")

if __name__ == "__main__":
    main()