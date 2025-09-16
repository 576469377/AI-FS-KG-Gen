# Large Language Model Integration Guide

## Introduction

This guide provides step-by-step instructions for integrating Large Language Models (LLMs) into the AI-FS-KG-Gen pipeline. Whether you're using cloud-based APIs or local models, this guide will help you set up and configure LLMs for optimal food safety knowledge extraction.

## Quick Start

### 1. Basic OpenAI Integration

The fastest way to get started with LLMs is using OpenAI's API:

```bash
# 1. Install required dependencies
pip install openai>=1.0.0

# 2. Set your API key
export OPENAI_API_KEY="your-api-key-here"

# 3. Test basic functionality
python -c "
from src.data_processing.llm_processor import LLMProcessor
processor = LLMProcessor(model_type='gpt-3.5-turbo', provider='openai')
result = processor.process_text('E. coli contamination in ground beef', task='extract_entities')
print(result)
"
```

### 2. Basic Local Model Integration

For privacy or cost considerations, use local models:

```bash
# 1. Install additional dependencies
pip install torch transformers accelerate

# 2. Test with a local model
python -c "
from src.data_processing.llm_processor import LLMProcessor
processor = LLMProcessor(model_type='microsoft/DialoGPT-medium', provider='huggingface')
result = processor.process_text('Food safety regulations require proper storage', task='summarize')
print(result)
"
```

## Detailed Setup Instructions

### OpenAI Models Setup

#### Step 1: Get API Access
1. Visit [OpenAI's website](https://openai.com)
2. Create an account or log in
3. Navigate to API section
4. Generate a new API key
5. Set up billing (required for API usage)

#### Step 2: Configure Environment
```bash
# Method 1: Environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Method 2: .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Method 3: Configuration file
cat > config/llm_config.yaml << EOF
openai:
  api_key: "sk-your-key-here"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000
EOF
```

#### Step 3: Test Configuration
```python
from src.data_processing.llm_processor import LLMProcessor

# Test OpenAI connection
processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")

# Test with food safety text
test_text = """
The FDA issued a recall notice for ground beef products due to potential E. coli contamination. 
Consumers should check their freezers and dispose of any affected products immediately.
"""

# Extract entities
entities = processor.process_text(test_text, task="extract_entities")
print("Extracted entities:", entities)

# Generate summary
summary = processor.process_text(test_text, task="summarize")
print("Summary:", summary)
```

### Hugging Face Models Setup

#### Step 1: Install Dependencies
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers>=4.20.0
pip install accelerate
pip install sentencepiece  # For some models

# Optional: For memory optimization
pip install bitsandbytes  # 8-bit quantization
```

#### Step 2: Choose Your Model

**For CPU-only systems:**
```python
model_options = [
    "microsoft/DialoGPT-medium",      # 345M parameters
    "google/flan-t5-small",           # 77M parameters
    "distilbert-base-uncased",        # 66M parameters
]
```

**For GPU systems (8GB+ VRAM):**
```python
model_options = [
    "meta-llama/Llama-2-7b-chat-hf",  # 7B parameters
    "mistralai/Mistral-7B-Instruct-v0.1",  # 7B parameters
    "google/flan-t5-large",           # 770M parameters
]
```

**For high-end GPU systems (16GB+ VRAM):**
```python
model_options = [
    "meta-llama/Llama-2-13b-chat-hf", # 13B parameters
    "codellama/CodeLlama-13b-Instruct-hf", # 13B parameters
]
```

#### Step 3: Configure Local Model
```python
from src.data_processing.llm_processor import LLMProcessor
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configure for your hardware
if device == "cuda":
    # GPU configuration
    processor = LLMProcessor(
        model_type="meta-llama/Llama-2-7b-chat-hf",
        provider="huggingface"
    )
else:
    # CPU configuration
    processor = LLMProcessor(
        model_type="microsoft/DialoGPT-medium",
        provider="huggingface"
    )

# Test the model
test_result = processor.process_text(
    "Salmonella contamination in chicken products",
    task="extract_entities"
)
print(test_result)
```

## Advanced Configuration

### Custom Model Configuration

#### Creating a Custom LLM Configuration
```python
# config/custom_llm_config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CustomLLMConfig:
    model_name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 2000
    batch_size: int = 4
    device: str = "auto"
    additional_params: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_name,
            "provider": self.provider,
            "config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "batch_size": self.batch_size,
                "device": self.device,
                **(self.additional_params or {})
            }
        }

# Example configurations
PRODUCTION_CONFIG = CustomLLMConfig(
    model_name="gpt-4-turbo",
    provider="openai",
    temperature=0.3,  # More deterministic
    max_tokens=3000
)

DEVELOPMENT_CONFIG = CustomLLMConfig(
    model_name="gpt-3.5-turbo",
    provider="openai",
    temperature=0.7,
    max_tokens=1500
)

LOCAL_CONFIG = CustomLLMConfig(
    model_name="microsoft/DialoGPT-medium",
    provider="huggingface",
    temperature=0.8,
    max_tokens=500,
    batch_size=2,
    device="cpu"
)
```

### Memory Optimization for Large Models

#### 8-bit Quantization
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model with 8-bit quantization
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Reduces memory by ~50%
    device_map="auto",
    torch_dtype=torch.float16
)
```

#### Gradient Checkpointing
```python
# For training or fine-tuning scenarios
model.gradient_checkpointing_enable()
```

#### Model Sharding
```python
# For very large models
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, checkpoint_path, device_map="auto"
)
```

## Food Safety-Specific Prompting

### Entity Extraction Prompts

#### Basic Entity Extraction
```python
def create_entity_extraction_prompt(text: str) -> str:
    return f"""
Extract food safety entities from the following text. Identify:
- Food products and ingredients
- Pathogens and microorganisms  
- Chemical compounds and allergens
- Safety standards and regulations
- Processing methods and conditions

Text: {text}

Return entities in JSON format:
{{
    "food_products": [...],
    "pathogens": [...],
    "allergens": [...],
    "chemicals": [...],
    "standards": [...],
    "processes": [...]
}}
"""

# Usage
processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
text = "Ground beef contaminated with E. coli O157:H7 must be recalled per FDA regulations."
prompt = create_entity_extraction_prompt(text)
result = processor._call_openai(prompt, max_tokens=500)
```

#### Advanced Entity Extraction with Context
```python
def create_advanced_entity_prompt(text: str, context: str = "") -> str:
    return f"""
You are an expert food safety analyst. Extract entities from the text below, considering the food safety context.

Context: {context if context else "General food safety analysis"}

Text: {text}

Extract the following entity types with confidence scores:
1. Food Products: specific foods, ingredients, products
2. Pathogens: bacteria, viruses, parasites, microorganisms
3. Allergens: known food allergens (milk, eggs, nuts, etc.)
4. Chemical Compounds: additives, preservatives, contaminants
5. Safety Standards: regulations, guidelines, standards (FDA, HACCP, etc.)
6. Risk Factors: conditions that increase food safety risks
7. Control Measures: actions to prevent/control risks

Format as JSON with confidence scores (0.0-1.0):
{{
    "entities": {{
        "food_products": [
            {{"text": "entity", "confidence": 0.95, "context": "surrounding context"}}
        ],
        "pathogens": [...],
        "allergens": [...],
        "chemicals": [...],
        "standards": [...],
        "risk_factors": [...],
        "control_measures": [...]
    }}
}}
"""
```

### Relation Extraction Prompts

#### Basic Relation Extraction
```python
def create_relation_extraction_prompt(text: str, entities: Dict) -> str:
    return f"""
Given the following text and identified entities, extract relationships between them.

Text: {text}

Entities: {entities}

Extract relationships in the format (subject, predicate, object) where:
- subject and object are entities from the list above
- predicate describes the relationship

Common food safety relationships:
- contains, contaminated_with, causes, prevents
- regulated_by, tested_for, stored_at
- processed_by, derived_from, associated_with

Return as JSON:
{{
    "relations": [
        {{
            "subject": "entity1",
            "predicate": "relationship",
            "object": "entity2",
            "confidence": 0.85,
            "evidence": "text snippet supporting this relation"
        }}
    ]
}}
"""
```

### Knowledge Generation Prompts

#### Risk Assessment Generation
```python
def create_risk_assessment_prompt(text: str) -> str:
    return f"""
Based on the following food safety information, generate a structured risk assessment.

Text: {text}

Provide a risk assessment including:
1. Identified hazards
2. Risk level (Low/Medium/High)
3. Affected populations
4. Control measures
5. Recommendations

Format as structured JSON:
{{
    "risk_assessment": {{
        "hazards": [...],
        "risk_level": "Medium",
        "affected_populations": [...],
        "control_measures": [...],
        "recommendations": [...]
    }}
}}
"""
```

## Integration Patterns

### Batch Processing Pattern

```python
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time

class BatchLLMProcessor:
    def __init__(self, processor: LLMProcessor, max_workers: int = 4):
        self.processor = processor
        self.max_workers = max_workers
    
    def process_batch(self, texts: List[str], task: str = "extract_entities") -> List[Dict[str, Any]]:
        """Process multiple texts in parallel"""
        
        def process_single(text: str) -> Dict[str, Any]:
            try:
                return self.processor.process_text(text, task)
            except Exception as e:
                return {"error": str(e), "text": text[:100]}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single, texts))
        
        return results
    
    def process_with_rate_limiting(self, texts: List[str], delay: float = 1.0) -> List[Dict[str, Any]]:
        """Process with rate limiting for API models"""
        results = []
        
        for text in texts:
            try:
                result = self.processor.process_text(text, "extract_entities")
                results.append(result)
                time.sleep(delay)  # Rate limiting
            except Exception as e:
                results.append({"error": str(e), "text": text[:100]})
        
        return results

# Usage
processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
batch_processor = BatchLLMProcessor(processor, max_workers=3)

texts = [
    "E. coli contamination in lettuce",
    "Salmonella outbreak linked to eggs",
    "FDA approves new food additive"
]

results = batch_processor.process_batch(texts)
```

### Caching Pattern

```python
import hashlib
import json
import os
from typing import Optional

class CachedLLMProcessor:
    def __init__(self, processor: LLMProcessor, cache_dir: str = "./llm_cache"):
        self.processor = processor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str, task: str) -> str:
        """Generate cache key for text and task"""
        content = f"{text}:{task}:{self.processor.model_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load result from cache"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass
    
    def process_text(self, text: str, task: str = "extract_entities") -> Dict[str, Any]:
        """Process text with caching"""
        cache_key = self._get_cache_key(text, task)
        
        # Try to load from cache
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result
        
        # Process with LLM
        result = self.processor.process_text(text, task)
        result["from_cache"] = False
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        return result

# Usage
processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
cached_processor = CachedLLMProcessor(processor)

# First call - processes with LLM
result1 = cached_processor.process_text("E. coli in beef", "extract_entities")
print(f"From cache: {result1.get('from_cache', False)}")

# Second call - loads from cache
result2 = cached_processor.process_text("E. coli in beef", "extract_entities")
print(f"From cache: {result2.get('from_cache', False)}")
```

## Error Handling and Reliability

### Robust Error Handling

```python
import time
import random
from typing import Optional, Callable

class RobustLLMProcessor:
    def __init__(self, processor: LLMProcessor, max_retries: int = 3):
        self.processor = processor
        self.max_retries = max_retries
    
    def process_with_retry(self, text: str, task: str = "extract_entities") -> Dict[str, Any]:
        """Process text with automatic retries"""
        
        for attempt in range(self.max_retries):
            try:
                result = self.processor.process_text(text, task)
                return result
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    return {
                        "task": task,
                        "result": None,
                        "error": str(e),
                        "attempts": self.max_retries
                    }
    
    def process_with_fallback(self, text: str, task: str, fallback_processor: Optional[LLMProcessor] = None) -> Dict[str, Any]:
        """Process with fallback to alternative processor"""
        
        try:
            result = self.processor.process_text(text, task)
            result["processor_used"] = "primary"
            return result
            
        except Exception as e:
            if fallback_processor:
                try:
                    result = fallback_processor.process_text(text, task)
                    result["processor_used"] = "fallback"
                    result["primary_error"] = str(e)
                    return result
                except Exception as fallback_error:
                    return {
                        "task": task,
                        "result": None,
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error)
                    }
            else:
                return {
                    "task": task,
                    "result": None,
                    "error": str(e)
                }

# Usage
primary_processor = LLMProcessor(model_type="gpt-4", provider="openai")
fallback_processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")

robust_processor = RobustLLMProcessor(primary_processor)

# Process with retry
result = robust_processor.process_with_retry("Food safety text...")

# Process with fallback
result = robust_processor.process_with_fallback(
    "Food safety text...", 
    "extract_entities",
    fallback_processor
)
```

## Performance Monitoring

### Token Usage Tracking

```python
import time
from collections import defaultdict

class LLMUsageTracker:
    def __init__(self, processor: LLMProcessor):
        self.processor = processor
        self.usage_stats = defaultdict(int)
        self.start_time = time.time()
    
    def process_text(self, text: str, task: str = "extract_entities") -> Dict[str, Any]:
        """Process text and track usage"""
        
        # Estimate tokens (rough approximation)
        estimated_input_tokens = len(text.split()) * 1.3
        
        start_time = time.time()
        result = self.processor.process_text(text, task)
        processing_time = time.time() - start_time
        
        # Update stats
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_processing_time"] += processing_time
        self.usage_stats["estimated_input_tokens"] += estimated_input_tokens
        self.usage_stats[f"{task}_requests"] += 1
        
        # Add metadata to result
        result["usage_metadata"] = {
            "processing_time": processing_time,
            "estimated_input_tokens": estimated_input_tokens
        }
        
        return result
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total_time = time.time() - self.start_time
        
        return {
            "session_duration": total_time,
            "requests_per_minute": self.usage_stats["total_requests"] / (total_time / 60),
            "average_processing_time": self.usage_stats["total_processing_time"] / max(1, self.usage_stats["total_requests"]),
            "total_estimated_tokens": self.usage_stats["estimated_input_tokens"],
            "detailed_stats": dict(self.usage_stats)
        }

# Usage
processor = LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
tracked_processor = LLMUsageTracker(processor)

# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
results = []

for text in texts:
    result = tracked_processor.process_text(text, "extract_entities")
    results.append(result)

# Get usage report
report = tracked_processor.get_usage_report()
print(json.dumps(report, indent=2))
```

## Troubleshooting Common Issues

### Issue 1: API Rate Limits
```python
# Solution: Implement rate limiting
import time

class RateLimitedProcessor:
    def __init__(self, processor: LLMProcessor, requests_per_minute: int = 20):
        self.processor = processor
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def process_text(self, text: str, task: str = "extract_entities") -> Dict[str, Any]:
        # Ensure minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        result = self.processor.process_text(text, task)
        self.last_request_time = time.time()
        
        return result
```

### Issue 2: CUDA Out of Memory
```python
# Solution: Automatic batch size reduction
class AdaptiveBatchProcessor:
    def __init__(self, processor: LLMProcessor, initial_batch_size: int = 8):
        self.processor = processor
        self.batch_size = initial_batch_size
    
    def process_batch(self, texts: List[str], task: str = "extract_entities") -> List[Dict[str, Any]]:
        results = []
        i = 0
        
        while i < len(texts):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Process batch
                for text in batch:
                    result = self.processor.process_text(text, task)
                    results.append(result)
                i += self.batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    self.batch_size = max(1, self.batch_size // 2)
                    print(f"CUDA OOM detected. Reducing batch size to {self.batch_size}")
                    
                    # Clear cache and retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        return results
```

### Issue 3: Model Loading Failures
```python
# Solution: Progressive model loading
def load_model_safely(model_name: str, fallback_models: List[str] = None) -> LLMProcessor:
    """Load model with fallbacks"""
    
    try:
        return LLMProcessor(model_type=model_name, provider="huggingface")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        
        if fallback_models:
            for fallback in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback}")
                    return LLMProcessor(model_type=fallback, provider="huggingface")
                except Exception as fallback_error:
                    print(f"Fallback {fallback} also failed: {fallback_error}")
                    continue
        
        # Final fallback to OpenAI if available
        try:
            print("Trying OpenAI as final fallback")
            return LLMProcessor(model_type="gpt-3.5-turbo", provider="openai")
        except Exception:
            raise RuntimeError("No working LLM processor could be initialized")

# Usage
processor = load_model_safely(
    "meta-llama/Llama-2-7b-chat-hf",
    fallback_models=[
        "microsoft/DialoGPT-medium",
        "google/flan-t5-small"
    ]
)
```

## Best Practices

### 1. Model Selection
- Start with OpenAI models for prototyping
- Move to local models for production/privacy
- Use appropriate model size for your hardware
- Consider cost vs. performance trade-offs

### 2. Prompt Engineering
- Be specific about output format
- Provide examples in prompts
- Use domain-specific terminology
- Test prompts with different model types

### 3. Error Handling
- Always implement retry logic for API calls
- Have fallback processors available
- Log errors for debugging
- Gracefully handle partial failures

### 4. Performance Optimization
- Cache frequently used results
- Use appropriate batch sizes
- Monitor token usage and costs
- Implement rate limiting for API models

### 5. Security
- Never log API keys
- Use environment variables for secrets
- Validate inputs to prevent injection
- Monitor for unusual usage patterns