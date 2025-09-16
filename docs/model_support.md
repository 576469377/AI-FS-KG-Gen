# Model Support Guide

## Overview

AI-FS-KG-Gen supports various AI models for different tasks in the knowledge graph generation pipeline. This guide provides comprehensive information about supported models, how to configure and use them, and best practices for different scenarios.

## Supported Model Categories

### 1. Large Language Models (LLMs)

LLMs are used for text understanding, entity extraction, relation extraction, and knowledge generation from textual data.

#### OpenAI Models
- **GPT-3.5-turbo** (Default)
  - Best for: General-purpose text processing, cost-effective
  - Max tokens: 4,096 input + output
  - Requirements: OpenAI API key
  - Cost: Low-moderate

- **GPT-4**
  - Best for: Complex reasoning, high-accuracy extraction
  - Max tokens: 8,192 input + output
  - Requirements: OpenAI API key
  - Cost: High

- **GPT-4-turbo**
  - Best for: Long documents, complex analysis
  - Max tokens: 128,000 input + output
  - Requirements: OpenAI API key
  - Cost: Moderate-high

#### Hugging Face Models
- **Microsoft DialoGPT**
  - Model: `microsoft/DialoGPT-medium`
  - Best for: Local deployment, privacy-sensitive data
  - Requirements: transformers, torch
  - Cost: Free (compute only)

- **Meta LLaMA 2**
  - Model: `meta-llama/Llama-2-7b-chat-hf`
  - Best for: High-quality local processing
  - Requirements: transformers, torch, GPU recommended
  - Cost: Free (compute only)

- **Google Flan-T5**
  - Model: `google/flan-t5-large`
  - Best for: Instruction following, structured output
  - Requirements: transformers, torch
  - Cost: Free (compute only)

#### Local/Open Source Models
- **CodeLlama**
  - Model: `codellama/CodeLlama-7b-Instruct-hf`
  - Best for: Code generation, structured data processing
  - Requirements: transformers, torch, 16GB+ RAM

- **Mistral 7B**
  - Model: `mistralai/Mistral-7B-Instruct-v0.1`
  - Best for: Efficient local processing
  - Requirements: transformers, torch, 8GB+ RAM

### 2. Vision-Language Models (VLMs)

VLMs process images and generate descriptions, answer questions about visual content, and extract information from images.

#### BLIP Models
- **BLIP Base** (Default)
  - Model: `Salesforce/blip-image-captioning-base`
  - Best for: General image captioning, lightweight processing
  - VRAM: 2-4GB
  - Tasks: Image captioning, visual question answering

- **BLIP Large**
  - Model: `Salesforce/blip-image-captioning-large`
  - Best for: High-quality captions, detailed analysis
  - VRAM: 6-8GB
  - Tasks: Detailed image captioning, complex VQA

#### CLIP Models
- **CLIP ViT-B/32**
  - Model: `openai/clip-vit-base-patch32`
  - Best for: Zero-shot image classification
  - VRAM: 1-2GB
  - Tasks: Image classification, similarity search

- **CLIP ViT-L/14**
  - Model: `openai/clip-vit-large-patch14`
  - Best for: High-accuracy classification
  - VRAM: 4-6GB
  - Tasks: Detailed image classification

#### Other VLMs
- **LLaVA**
  - Model: `llava-hf/llava-1.5-7b-hf`
  - Best for: Conversational image analysis
  - VRAM: 8-12GB
  - Tasks: Complex visual reasoning

### 3. Embedding Models

Used for semantic similarity and vector representations of text and entities.

#### Sentence Transformers
- **all-MiniLM-L6-v2** (Default)
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Best for: Fast, general-purpose embeddings
  - Dimensions: 384
  - Speed: Very fast

- **all-mpnet-base-v2**
  - Model: `sentence-transformers/all-mpnet-base-v2`
  - Best for: High-quality embeddings
  - Dimensions: 768
  - Speed: Moderate

#### Domain-Specific Models
- **BioBERT**
  - Model: `dmis-lab/biobert-base-cased-v1.1`
  - Best for: Biomedical and food safety text
  - Dimensions: 768
  - Specialty: Scientific literature

### 4. Named Entity Recognition (NER) Models

#### spaCy Models
- **en_core_web_sm** (Default)
  - Best for: General entity recognition, fast processing
  - Size: ~15MB
  - Entities: Person, Organization, Location, etc.

- **en_core_web_lg**
  - Best for: High-accuracy entity recognition
  - Size: ~750MB
  - Entities: Extended entity types with word vectors

#### Custom Food Safety Models
- **Food-BERT**
  - Model: Custom fine-tuned BERT for food safety entities
  - Best for: Food-specific entity recognition
  - Entities: Pathogens, allergens, food products, etc.

## Model Selection Guide

### By Use Case

#### Small-Scale Processing (< 1000 documents)
- **LLM**: GPT-3.5-turbo (OpenAI) or Flan-T5-large (local)
- **VLM**: BLIP-base
- **Embedding**: all-MiniLM-L6-v2
- **NER**: en_core_web_sm

#### Large-Scale Processing (> 10,000 documents)
- **LLM**: Local models (LLaMA 2, Mistral) for cost efficiency
- **VLM**: CLIP for classification, BLIP for detailed analysis
- **Embedding**: all-mpnet-base-v2 for better quality
- **NER**: en_core_web_lg or custom food safety models

#### Privacy-Sensitive Environments
- **LLM**: Local Hugging Face models only
- **VLM**: Local BLIP/CLIP models
- **Embedding**: Local sentence transformers
- **NER**: Local spaCy models

#### High-Accuracy Requirements
- **LLM**: GPT-4 or GPT-4-turbo
- **VLM**: BLIP-large or LLaVA
- **Embedding**: all-mpnet-base-v2
- **NER**: en_core_web_lg + custom models

### By Hardware Resources

#### CPU-Only Systems
- **LLM**: Smaller models (Flan-T5-small, DistilBERT)
- **VLM**: CLIP ViT-B/32 (lighter than BLIP)
- **Embedding**: all-MiniLM-L6-v2
- **NER**: en_core_web_sm

#### GPU Systems (8GB+ VRAM)
- **LLM**: LLaMA 2 7B, Mistral 7B
- **VLM**: BLIP-large, LLaVA
- **Embedding**: all-mpnet-base-v2
- **NER**: en_core_web_lg

#### High-End GPU Systems (16GB+ VRAM)
- **LLM**: LLaMA 2 13B, Code Llama 13B
- **VLM**: LLaVA 13B, BLIP-2
- **Embedding**: Large embedding models
- **NER**: Multiple ensemble models

## Configuration Examples

### Basic Configuration (API-based)
```python
config = PipelineConfig(
    # LLM settings
    use_llm=True,
    llm_model="gpt-3.5-turbo",
    llm_provider="openai",
    
    # VLM settings
    use_vlm=True,
    vlm_model="blip-base",
    
    # Processing settings
    entity_extraction=True,
    relation_extraction=True,
    
    # Performance settings
    max_workers=4,
    batch_size=10
)
```

### Local Models Configuration
```python
config = PipelineConfig(
    # Local LLM
    use_llm=True,
    llm_model="microsoft/DialoGPT-medium",
    llm_provider="huggingface",
    
    # Local VLM
    use_vlm=True,
    vlm_model="Salesforce/blip-image-captioning-base",
    
    # Local embeddings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    
    # GPU settings
    device="cuda",
    
    # Performance settings for local models
    max_workers=2,  # Reduce for GPU memory
    batch_size=4
)
```

### High-Performance Configuration
```python
config = PipelineConfig(
    # High-end LLM
    use_llm=True,
    llm_model="gpt-4-turbo",
    llm_provider="openai",
    
    # High-quality VLM
    use_vlm=True,
    vlm_model="Salesforce/blip-image-captioning-large",
    
    # Best embedding model
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    
    # High-accuracy NER
    ner_model="en_core_web_lg",
    
    # Performance settings
    max_workers=8,
    batch_size=20,
    
    # Quality settings
    entity_confidence_threshold=0.8,
    relation_confidence_threshold=0.7
)
```

## Model-Specific Configuration

### OpenAI Models
```python
# Environment variables
export OPENAI_API_KEY="your-api-key-here"

# Configuration
llm_config = {
    "model_type": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 2000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

### Hugging Face Models
```python
# Configuration
llm_config = {
    "model_type": "meta-llama/Llama-2-7b-chat-hf",
    "device": "cuda",  # or "cpu"
    "torch_dtype": "float16",  # for GPU efficiency
    "load_in_8bit": True,  # for memory efficiency
    "max_new_tokens": 500,
    "temperature": 0.7,
    "do_sample": True
}
```

### BLIP Models
```python
# Configuration
vlm_config = {
    "model_type": "Salesforce/blip-image-captioning-large",
    "device": "cuda",
    "max_length": 100,
    "num_beams": 5,
    "early_stopping": True
}
```

### CLIP Models
```python
# Configuration
vlm_config = {
    "model_type": "openai/clip-vit-base-patch32",
    "device": "cuda",
    "batch_size": 32
}
```

## Performance Considerations

### Memory Requirements

#### LLM Memory Usage
- **GPT-3.5-turbo**: API-based (no local memory)
- **LLaMA 2 7B**: ~13GB RAM/VRAM
- **LLaMA 2 13B**: ~26GB RAM/VRAM
- **Flan-T5-large**: ~3GB RAM/VRAM

#### VLM Memory Usage
- **BLIP-base**: ~2GB VRAM
- **BLIP-large**: ~6GB VRAM
- **CLIP ViT-B/32**: ~1GB VRAM
- **LLaVA 7B**: ~8GB VRAM

### Speed Optimization Tips

1. **Use appropriate batch sizes**
   - API models: 10-20 items per batch
   - Local models: 2-8 items per batch (GPU dependent)

2. **Enable model caching**
   ```python
   from transformers import AutoModel
   
   # Enable caching
   model = AutoModel.from_pretrained(
       "model-name",
       cache_dir="./model_cache"
   )
   ```

3. **Use mixed precision**
   ```python
   # For local models
   torch_dtype = torch.float16  # Half precision
   ```

4. **Optimize for your hardware**
   - CPU: Use smaller models, increase workers
   - GPU: Use larger batch sizes, fewer workers

## Cost Optimization

### API Model Costs (Approximate)
- **GPT-3.5-turbo**: $0.001-0.002 per 1K tokens
- **GPT-4**: $0.03-0.06 per 1K tokens
- **GPT-4-turbo**: $0.01-0.03 per 1K tokens

### Cost Reduction Strategies

1. **Use local models for bulk processing**
2. **Implement caching for repeated queries**
3. **Optimize prompts to reduce token usage**
4. **Use cheaper models for initial processing, expensive models for refinement**

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues
```
Error: CUDA out of memory
Solution: Reduce batch size or use CPU
```

```
Error: Model not found
Solution: Check model name and install required packages
pip install transformers torch
```

#### API Issues
```
Error: OpenAI API rate limit exceeded
Solution: Implement rate limiting or increase API tier
```

```
Error: Invalid API key
Solution: Check OPENAI_API_KEY environment variable
```

#### Performance Issues
```
Issue: Slow processing
Solutions:
- Reduce model size
- Increase batch size
- Use GPU if available
- Enable model caching
```

### Model Compatibility

#### Python Version Requirements
- **Python 3.8+**: Required for all models
- **Python 3.10+**: Recommended for latest features

#### Package Compatibility
```bash
# Core packages
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install sentence-transformers>=2.2.0

# Optional packages
pip install accelerate  # For large model loading
pip install bitsandbytes  # For 8-bit quantization
```

## Future Model Support

### Planned Additions
- **Anthropic Claude**: Advanced reasoning capabilities
- **Google PaLM**: Efficient processing
- **Custom fine-tuned models**: Domain-specific optimization
- **Multimodal models**: Combined text/image processing

### Model Request Process
To request support for a new model:
1. Open an issue on GitHub
2. Provide model details and use case
3. Include performance benchmarks if available
4. Specify integration requirements