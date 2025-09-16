# Configuration Examples

This document provides practical configuration examples for different deployment scenarios, model combinations, and use cases.

## Environment Setup Examples

### Development Environment (Local Testing)

**System Requirements:**
- 8GB+ RAM
- Python 3.8+
- Internet connection for API models

**Configuration:**
```python
# config/dev_config.py
from pipeline import PipelineConfig

DEV_CONFIG = PipelineConfig(
    # Input settings
    input_sources=["./examples/sample_data/"],
    input_types=["text"],
    
    # AI model settings (lightweight for development)
    use_llm=True,
    use_vlm=False,  # Disable VLM for faster testing
    llm_model="gpt-3.5-turbo",
    llm_provider="openai",
    
    # Extraction settings (lower thresholds for testing)
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.5,
    relation_confidence_threshold=0.5,
    
    # Backend settings
    kg_backend="networkx",  # In-memory for quick testing
    kg_output_format="json",
    
    # Output settings
    output_dir="./dev_output/",
    save_intermediate_results=True,
    
    # Performance settings (conservative for development)
    max_workers=2,
    batch_size=5
)
```

**Environment Variables:**
```bash
# .env.dev
OPENAI_API_KEY=your-dev-api-key
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

**Usage:**
```python
from config.dev_config import DEV_CONFIG
from pipeline import AIFSKGPipeline

pipeline = AIFSKGPipeline(DEV_CONFIG)
results = pipeline.run()
```

### Production Environment (High Throughput)

**System Requirements:**
- 32GB+ RAM
- GPU with 16GB+ VRAM (optional)
- High-speed internet
- Neo4j database

**Configuration:**
```python
# config/prod_config.py
from pipeline import PipelineConfig

PROD_CONFIG = PipelineConfig(
    # Input settings
    input_sources=[
        "/data/food_safety_docs/",
        "/data/regulatory_updates/",
        "/data/research_papers/"
    ],
    input_types=["text", "image", "structured"],
    
    # AI model settings (high performance)
    use_llm=True,
    use_vlm=True,
    llm_model="gpt-4-turbo",
    llm_provider="openai",
    vlm_model="Salesforce/blip-image-captioning-large",
    
    # Extraction settings (high accuracy)
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.8,
    relation_confidence_threshold=0.7,
    
    # Backend settings (scalable database)
    kg_backend="neo4j",
    kg_output_format="json",
    
    # Output settings
    output_dir="/var/output/food_safety_kg/",
    save_intermediate_results=True,
    
    # Performance settings (optimized for throughput)
    max_workers=8,
    batch_size=20
)
```

**Environment Variables:**
```bash
# .env.prod
OPENAI_API_KEY=your-prod-api-key
NEO4J_URI=bolt://neo4j-server:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Privacy-First Environment (No External APIs)

**System Requirements:**
- 64GB+ RAM
- GPU with 24GB+ VRAM
- No internet requirements

**Configuration:**
```python
# config/privacy_config.py
from pipeline import PipelineConfig

PRIVACY_CONFIG = PipelineConfig(
    # Input settings
    input_sources=["./sensitive_data/"],
    input_types=["text", "image"],
    
    # AI model settings (all local)
    use_llm=True,
    use_vlm=True,
    llm_model="meta-llama/Llama-2-13b-chat-hf",
    llm_provider="huggingface",
    vlm_model="Salesforce/blip-image-captioning-large",
    
    # Extraction settings
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.7,
    relation_confidence_threshold=0.6,
    
    # Backend settings (local graph database)
    kg_backend="networkx",
    kg_output_format="json",
    
    # Output settings
    output_dir="./private_output/",
    save_intermediate_results=True,
    
    # Performance settings (GPU optimized)
    max_workers=4,
    batch_size=8
)

# Additional local model configuration
LOCAL_MODEL_CONFIG = {
    "llm": {
        "device": "cuda",
        "torch_dtype": "float16",
        "load_in_8bit": True,
        "cache_dir": "./model_cache/"
    },
    "vlm": {
        "device": "cuda",
        "batch_size": 4
    }
}
```

**No External Dependencies:**
```bash
# All models cached locally
export TRANSFORMERS_CACHE=./model_cache/
export HF_HOME=./model_cache/
```

### Budget-Conscious Environment (Cost Optimization)

**System Requirements:**
- 16GB+ RAM
- CPU-only (no GPU required)
- Limited API usage

**Configuration:**
```python
# config/budget_config.py
from pipeline import PipelineConfig

BUDGET_CONFIG = PipelineConfig(
    # Input settings
    input_sources=["./data/"],
    input_types=["text"],  # Text only to reduce processing costs
    
    # AI model settings (cost-optimized)
    use_llm=True,
    use_vlm=False,  # Disable expensive VLM processing
    llm_model="gpt-3.5-turbo",  # Cheaper than GPT-4
    llm_provider="openai",
    
    # Extraction settings (balanced)
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.6,
    relation_confidence_threshold=0.6,
    
    # Backend settings
    kg_backend="networkx",
    kg_output_format="json",
    
    # Output settings
    output_dir="./budget_output/",
    save_intermediate_results=False,  # Save storage space
    
    # Performance settings (minimize API calls)
    max_workers=2,
    batch_size=10
)

# Cost optimization wrapper
class BudgetLLMProcessor:
    def __init__(self, base_processor, daily_limit_usd=10.0):
        self.base_processor = base_processor
        self.daily_limit = daily_limit_usd
        self.daily_usage = 0.0
        
    def process_text(self, text, task="extract_entities"):
        # Estimate cost (approximate)
        tokens = len(text.split()) * 1.3
        estimated_cost = tokens * 0.000002  # GPT-3.5-turbo pricing
        
        if self.daily_usage + estimated_cost > self.daily_limit:
            raise Exception(f"Daily budget limit reached: ${self.daily_limit}")
        
        result = self.base_processor.process_text(text, task)
        self.daily_usage += estimated_cost
        
        return result
```

## Model-Specific Configurations

### OpenAI Models Configuration

```python
# config/openai_models.py

# GPT-3.5 Turbo (Cost-effective)
GPT35_CONFIG = {
    "model_type": "gpt-3.5-turbo",
    "provider": "openai",
    "temperature": 0.7,
    "max_tokens": 1500,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# GPT-4 (High accuracy)
GPT4_CONFIG = {
    "model_type": "gpt-4",
    "provider": "openai",
    "temperature": 0.3,  # More deterministic
    "max_tokens": 2000,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# GPT-4 Turbo (Long context)
GPT4_TURBO_CONFIG = {
    "model_type": "gpt-4-turbo",
    "provider": "openai",
    "temperature": 0.5,
    "max_tokens": 4000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Usage example
from src.data_processing.llm_processor import LLMProcessor

def create_openai_processor(config_name="gpt35"):
    configs = {
        "gpt35": GPT35_CONFIG,
        "gpt4": GPT4_CONFIG,
        "gpt4_turbo": GPT4_TURBO_CONFIG
    }
    
    config = configs.get(config_name, GPT35_CONFIG)
    return LLMProcessor(
        model_type=config["model_type"],
        provider=config["provider"]
    )
```

### Hugging Face Models Configuration

```python
# config/huggingface_models.py

# LLaMA 2 Models
LLAMA2_7B_CONFIG = {
    "model_type": "meta-llama/Llama-2-7b-chat-hf",
    "provider": "huggingface",
    "device": "cuda",
    "torch_dtype": "float16",
    "load_in_8bit": False,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9
}

LLAMA2_13B_CONFIG = {
    "model_type": "meta-llama/Llama-2-13b-chat-hf",
    "provider": "huggingface",
    "device": "cuda",
    "torch_dtype": "float16",
    "load_in_8bit": True,  # Enable for memory efficiency
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9
}

# Mistral Models
MISTRAL_7B_CONFIG = {
    "model_type": "mistralai/Mistral-7B-Instruct-v0.1",
    "provider": "huggingface",
    "device": "cuda",
    "torch_dtype": "float16",
    "load_in_4bit": True,  # Even more memory efficient
    "max_new_tokens": 256,
    "temperature": 0.6,
    "do_sample": True
}

# Code Llama
CODE_LLAMA_CONFIG = {
    "model_type": "codellama/CodeLlama-7b-Instruct-hf",
    "provider": "huggingface",
    "device": "cuda",
    "torch_dtype": "float16",
    "load_in_8bit": True,
    "max_new_tokens": 512,
    "temperature": 0.1,  # Low temperature for code generation
    "do_sample": False   # Deterministic for code
}

# CPU-optimized models
CPU_OPTIMIZED_CONFIG = {
    "model_type": "microsoft/DialoGPT-medium",
    "provider": "huggingface",
    "device": "cpu",
    "torch_dtype": "float32",
    "max_new_tokens": 128,
    "temperature": 0.8,
    "do_sample": True
}
```

### Vision-Language Models Configuration

```python
# config/vlm_models.py

# BLIP Models
BLIP_BASE_CONFIG = {
    "model_type": "Salesforce/blip-image-captioning-base",
    "device": "cuda",
    "max_length": 50,
    "num_beams": 5,
    "early_stopping": True,
    "batch_size": 8
}

BLIP_LARGE_CONFIG = {
    "model_type": "Salesforce/blip-image-captioning-large",
    "device": "cuda",
    "max_length": 100,
    "num_beams": 8,
    "early_stopping": True,
    "batch_size": 4  # Larger model, smaller batch
}

# CLIP Models
CLIP_BASE_CONFIG = {
    "model_type": "openai/clip-vit-base-patch32",
    "device": "cuda",
    "batch_size": 32
}

CLIP_LARGE_CONFIG = {
    "model_type": "openai/clip-vit-large-patch14",
    "device": "cuda",
    "batch_size": 16
}

# LLaVA Models
LLAVA_CONFIG = {
    "model_type": "llava-hf/llava-1.5-7b-hf",
    "device": "cuda",
    "torch_dtype": "float16",
    "load_in_8bit": True,
    "max_new_tokens": 512
}
```

## Task-Specific Configurations

### Entity Extraction Focused

```python
# config/entity_extraction_config.py

ENTITY_EXTRACTION_CONFIG = PipelineConfig(
    input_sources=["./entity_data/"],
    input_types=["text"],
    
    # Focus on entity extraction
    use_llm=True,
    use_vlm=False,
    llm_model="gpt-4",  # High accuracy for entities
    
    # Entity-specific settings
    entity_extraction=True,
    relation_extraction=False,  # Disable to focus on entities
    entity_confidence_threshold=0.9,  # High confidence
    
    kg_backend="networkx",
    kg_output_format="json",
    
    # Optimized for entity extraction
    max_workers=6,
    batch_size=15
)

# Custom entity types for food safety
FOOD_SAFETY_ENTITIES = [
    "pathogen", "bacteria", "virus", "parasite",
    "food_product", "ingredient", "additive",
    "allergen", "contaminant", "toxin",
    "regulation", "standard", "guideline",
    "temperature", "ph_level", "time_limit",
    "processing_method", "storage_condition"
]
```

### Image Analysis Focused

```python
# config/image_analysis_config.py

IMAGE_ANALYSIS_CONFIG = PipelineConfig(
    input_sources=["./food_images/"],
    input_types=["image"],
    
    # Focus on image analysis
    use_llm=False,
    use_vlm=True,
    vlm_model="Salesforce/blip-image-captioning-large",
    
    # Image-specific settings
    entity_extraction=False,
    relation_extraction=False,
    
    kg_backend="networkx",
    kg_output_format="json",
    
    # Optimized for image processing
    max_workers=4,
    batch_size=8
)

# Image processing tasks
IMAGE_TASKS = [
    "caption",
    "classify", 
    "analyze_food_safety",
    "extract_text"
]
```

### Knowledge Graph Construction Focused

```python
# config/kg_construction_config.py

KG_CONSTRUCTION_CONFIG = PipelineConfig(
    input_sources=["./structured_data/"],
    input_types=["text", "structured"],
    
    # Balanced processing
    use_llm=True,
    use_vlm=False,
    llm_model="gpt-3.5-turbo",
    
    # Focus on both entities and relations
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.7,
    relation_confidence_threshold=0.7,
    
    # Advanced KG backend
    kg_backend="neo4j",
    kg_output_format="gexf",
    
    # KG-optimized settings
    max_workers=6,
    batch_size=12
)

# Neo4j specific settings
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your-password",
    "database": "food_safety_kg"
}
```

## Deployment Scenarios

### Docker Deployment

```dockerfile
# Dockerfile.api
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""

# Run configuration
EXPOSE 8000

CMD ["python", "-m", "src.api.main"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-fs-kg-gen:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    depends_on:
      - neo4j
    ports:
      - "8000:8000"

  neo4j:
    image: neo4j:5.0
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  neo4j_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-fs-kg-gen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-fs-kg-gen
  template:
    metadata:
      labels:
        app: ai-fs-kg-gen
    spec:
      containers:
      - name: ai-fs-kg-gen
        image: ai-fs-kg-gen:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: output-volume
          mountPath: /app/output
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: output-pvc
```

### AWS Lambda Deployment

```python
# lambda/handler.py
import json
import boto3
from src.pipeline import AIFSKGPipeline, PipelineConfig

def lambda_handler(event, context):
    """AWS Lambda handler for AI-FS-KG-Gen processing"""
    
    # Extract configuration from event
    input_text = event.get('text', '')
    task = event.get('task', 'extract_entities')
    
    # Create lightweight configuration for Lambda
    config = PipelineConfig(
        use_llm=True,
        use_vlm=False,  # Disable VLM for Lambda constraints
        llm_model="gpt-3.5-turbo",
        entity_extraction=True,
        relation_extraction=False,  # Simplified for Lambda
        kg_backend="networkx",
        max_workers=1,  # Single worker for Lambda
        batch_size=1
    )
    
    try:
        # Process with pipeline
        pipeline = AIFSKGPipeline(config)
        results = pipeline.process_single_text(input_text, task)
        
        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

## Monitoring and Observability

### Logging Configuration

```python
# config/logging_config.py
import logging
from loguru import logger
import sys

def setup_production_logging():
    """Setup comprehensive logging for production"""
    
    # Remove default handler
    logger.remove()
    
    # Add structured logging
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=False
    )
    
    # Add file logging with rotation
    logger.add(
        "logs/ai_fs_kg_gen_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz"
    )
    
    # Add error-only logging
    logger.add(
        "logs/errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="90 days"
    )

def setup_development_logging():
    """Setup simple logging for development"""
    
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
        level="DEBUG"
    )
```

### Metrics Collection

```python
# config/metrics_config.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('ai_fs_kg_requests_total', 'Total requests', ['method', 'status'])
REQUEST_DURATION = Histogram('ai_fs_kg_request_duration_seconds', 'Request duration')
ACTIVE_PROCESSING = Gauge('ai_fs_kg_active_processing', 'Currently processing items')
TOKEN_USAGE = Counter('ai_fs_kg_tokens_used_total', 'Total tokens used', ['model'])

class MetricsCollector:
    def __init__(self):
        self.start_time = None
    
    def start_request(self):
        self.start_time = time.time()
        ACTIVE_PROCESSING.inc()
    
    def end_request(self, method, status, tokens_used=0, model="unknown"):
        if self.start_time:
            duration = time.time() - self.start_time
            REQUEST_DURATION.observe(duration)
        
        REQUEST_COUNT.labels(method=method, status=status).inc()
        ACTIVE_PROCESSING.dec()
        
        if tokens_used > 0:
            TOKEN_USAGE.labels(model=model).inc(tokens_used)

# Usage
metrics = MetricsCollector()
metrics.start_request()
# ... processing ...
metrics.end_request("extract_entities", "success", tokens_used=150, model="gpt-3.5-turbo")
```

## Performance Tuning Examples

### Memory-Optimized Configuration

```python
# config/memory_optimized.py

MEMORY_OPTIMIZED_CONFIG = PipelineConfig(
    # Reduce memory usage
    use_llm=True,
    use_vlm=False,  # VLM uses significant memory
    llm_model="microsoft/DialoGPT-medium",  # Smaller model
    
    # Conservative settings
    max_workers=2,  # Fewer workers
    batch_size=4,   # Smaller batches
    
    # Don't save intermediate results
    save_intermediate_results=False,
    
    # Use NetworkX instead of Neo4j
    kg_backend="networkx"
)

# Additional memory optimizations
import gc
import torch

def optimize_memory():
    """Memory optimization utilities"""
    
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Enable memory-mapped files
    torch.backends.cuda.matmul.allow_tf32 = True
```

### Speed-Optimized Configuration

```python
# config/speed_optimized.py

SPEED_OPTIMIZED_CONFIG = PipelineConfig(
    # Use fastest models
    use_llm=True,
    use_vlm=True,
    llm_model="gpt-3.5-turbo",  # Faster than GPT-4
    vlm_model="openai/clip-vit-base-patch32",  # Faster than BLIP
    
    # Aggressive parallelization
    max_workers=8,
    batch_size=20,
    
    # Lower quality thresholds for speed
    entity_confidence_threshold=0.6,
    relation_confidence_threshold=0.5,
    
    # Fast backend
    kg_backend="networkx"
)

# Speed optimization techniques
def enable_speed_optimizations():
    """Enable various speed optimizations"""
    
    # Enable cuDNN benchmarking
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Enable TensorFloat-32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

This comprehensive configuration guide provides practical examples for various deployment scenarios, from development to production, and covers different model types, hardware configurations, and optimization strategies.