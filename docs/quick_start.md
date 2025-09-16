# AI Model Integration - Quick Start Guide

This quick start guide helps you get up and running with AI models in the AI-FS-KG-Gen pipeline. Choose your scenario below and follow the appropriate path.

## 🚀 Choose Your Path

### 1. **Just Want to Try It** (API-based)
**Best for:** Quick testing, prototyping
**Requirements:** Internet connection, OpenAI API key
**Cost:** ~$0.01-0.10 per document

```bash
# 1. Set up API key
export OPENAI_API_KEY="sk-your-key-here"

# 2. Install and run
pip install openai
python examples/model_configurations.py

# 3. Basic usage
from pipeline import AIFSKGPipeline, PipelineConfig
config = PipelineConfig(
    input_sources=["your_data/"],
    use_llm=True,
    llm_model="gpt-3.5-turbo"
)
pipeline = AIFSKGPipeline(config)
results = pipeline.run()
```

### 2. **Privacy-First** (Local models only)
**Best for:** Sensitive data, no external dependencies
**Requirements:** 16GB+ RAM, GPU recommended
**Cost:** Free (compute only)

```bash
# 1. Install dependencies
pip install torch transformers accelerate

# 2. Run with local models
from pipeline import PipelineConfig
config = PipelineConfig(
    input_sources=["your_data/"],
    use_llm=True,
    llm_model="microsoft/DialoGPT-medium",
    llm_provider="huggingface"
)
```

### 3. **Production Ready** (High quality)
**Best for:** Live systems, high accuracy requirements
**Requirements:** Good budget, API access
**Cost:** Higher but excellent quality

```bash
# 1. Set up environment
export OPENAI_API_KEY="sk-your-production-key"
export NEO4J_URI="bolt://your-neo4j:7687"

# 2. Configure for production
config = PipelineConfig(
    input_sources=["production_data/"],
    use_llm=True,
    use_vlm=True,
    llm_model="gpt-4",
    vlm_model="Salesforce/blip-image-captioning-large",
    kg_backend="neo4j"
)
```

### 4. **Budget Conscious** (Cost optimized)
**Best for:** Limited budget, acceptable quality
**Requirements:** Basic setup
**Cost:** Minimal

```bash
# Use cheaper models and optimize processing
config = PipelineConfig(
    input_sources=["your_data/"],
    input_types=["text"],  # Text only
    use_llm=True,
    use_vlm=False,  # Disable expensive image processing
    llm_model="gpt-3.5-turbo",
    relation_extraction=False  # Reduce API calls
)
```

## 📋 Model Selection Cheat Sheet

| Scenario | LLM Model | VLM Model | Backend | Cost | Setup Time |
|----------|-----------|-----------|---------|------|------------|
| **Quick Test** | gpt-3.5-turbo | blip-base | networkx | $ | 5 minutes |
| **Development** | gpt-3.5-turbo | blip-base | networkx | $ | 10 minutes |
| **Production** | gpt-4 | blip-large | neo4j | $$$ | 30 minutes |
| **Privacy** | Llama-2-7b | blip-base | networkx | Free | 60 minutes |
| **Budget** | gpt-3.5-turbo | None | networkx | $ | 5 minutes |
| **High Volume** | Local models | clip-base | neo4j | Free | 90 minutes |

## 🛠️ Installation by Scenario

### Minimal Installation (API only)
```bash
pip install openai requests pillow networkx
export OPENAI_API_KEY="your-key"
```

### Full Installation (All features)
```bash
pip install -r requirements.txt
pip install torch transformers accelerate
python -m spacy download en_core_web_sm
```

### GPU Installation (Local models)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
```

## 🔧 Common Issues & Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| "CUDA out of memory" | Reduce batch_size to 1, or use CPU |
| "Rate limit exceeded" | Add delays or switch to local models |
| "Model not found" | Check spelling, install transformers |
| "API key invalid" | Verify key at platform.openai.com |
| "Slow processing" | Use smaller models or GPU |

## 📚 Documentation Index

- **[Model Support Guide](docs/model_support.md)** - Complete model overview (15+ models)
- **[LLM Integration Guide](docs/llm_integration.md)** - Step-by-step setup (OpenAI + Hugging Face)
- **[Configuration Examples](docs/configuration_examples.md)** - Real deployment scenarios
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Usage Guide](docs/usage.md)** - Basic usage patterns

## 🎯 Next Steps

1. **Choose your scenario** from the options above
2. **Follow the installation** steps for your chosen path
3. **Run the example** to verify everything works
4. **Read the detailed guides** for your specific models
5. **Check troubleshooting** if you encounter issues

## 💡 Pro Tips

- **Start simple**: Begin with OpenAI models, then migrate to local if needed
- **Test first**: Use small datasets to verify configuration before full runs
- **Monitor costs**: Track API usage if using paid services
- **Cache results**: Enable caching to avoid reprocessing same content
- **Use appropriate hardware**: Match model size to your available resources

## 🆘 Need Help?

1. **Check the logs** - Most issues show helpful error messages
2. **Try the troubleshooting guide** - Covers 90% of common issues
3. **Run diagnostics** - Use the built-in diagnostic functions
4. **Start with minimal config** - Gradually add complexity
5. **Open an issue** - If you're still stuck

---

**Ready to start?** Pick your scenario above and follow the quick setup steps!