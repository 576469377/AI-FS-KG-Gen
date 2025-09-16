# Troubleshooting Guide for Model Issues

This guide helps you diagnose and resolve common issues when working with AI models in the AI-FS-KG-Gen pipeline.

## Common Issues Quick Reference

| Issue | Symptoms | Quick Fix |
|-------|----------|-----------|
| CUDA Out of Memory | `RuntimeError: CUDA out of memory` | Reduce batch size, use CPU, enable 8-bit loading |
| API Rate Limits | `Rate limit exceeded` | Add delays, implement exponential backoff |
| Model Not Found | `Model not found` | Check model name, install required packages |
| Slow Processing | Takes too long | Use smaller models, increase workers, enable GPU |
| Poor Accuracy | Low quality results | Use larger models, adjust confidence thresholds |
| Connection Errors | API timeouts | Check internet, API keys, implement retries |

## OpenAI API Issues

### Issue 1: Invalid API Key

**Symptoms:**
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Diagnosis:**
```python
import os
import openai

# Check if API key is set
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key exists: {api_key is not None}")
print(f"API Key starts with 'sk-': {api_key.startswith('sk-') if api_key else False}")

# Test API key
try:
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("API key is valid")
except Exception as e:
    print(f"API key error: {e}")
```

**Solutions:**
1. **Check API Key Format:**
   ```bash
   # Valid format: sk-...
   export OPENAI_API_KEY="sk-your-actual-key-here"
   ```

2. **Verify API Key in Dashboard:**
   - Go to https://platform.openai.com/api-keys
   - Verify key exists and is active
   - Check usage limits

3. **Environment Variable Issues:**
   ```python
   # Method 1: Set in code (not recommended for production)
   import openai
   openai.api_key = "your-key-here"
   
   # Method 2: Use .env file
   from dotenv import load_dotenv
   load_dotenv()
   
   # Method 3: System environment
   # In bash: export OPENAI_API_KEY="your-key"
   ```

### Issue 2: Rate Limiting

**Symptoms:**
```
openai.error.RateLimitError: Rate limit reached for requests
```

**Diagnosis:**
```python
import time
from datetime import datetime

class RateLimitDiagnostic:
    def __init__(self):
        self.request_times = []
    
    def log_request(self):
        self.request_times.append(datetime.now())
        
        # Check requests per minute
        recent_requests = [t for t in self.request_times 
                          if (datetime.now() - t).seconds < 60]
        
        print(f"Requests in last minute: {len(recent_requests)}")
        return len(recent_requests)

# Usage
diagnostic = RateLimitDiagnostic()
```

**Solutions:**
1. **Implement Rate Limiting:**
   ```python
   import time
   import random
   
   class RateLimitedProcessor:
       def __init__(self, requests_per_minute=20):
           self.min_delay = 60.0 / requests_per_minute
           self.last_request = 0
       
       def process_with_rate_limit(self, text):
           # Ensure minimum delay
           elapsed = time.time() - self.last_request
           if elapsed < self.min_delay:
               time.sleep(self.min_delay - elapsed)
           
           try:
               result = self.process_text(text)
               self.last_request = time.time()
               return result
           except Exception as e:
               if "rate limit" in str(e).lower():
                   # Exponential backoff
                   delay = min(60, 2 ** (getattr(self, 'retry_count', 0)))
                   time.sleep(delay + random.uniform(0, 1))
                   self.retry_count = getattr(self, 'retry_count', 0) + 1
                   raise
   ```

2. **Upgrade API Tier:**
   - Visit OpenAI billing dashboard
   - Increase rate limits with higher tier

3. **Batch Processing:**
   ```python
   def process_batch_with_delays(texts, delay=1.0):
       results = []
       for i, text in enumerate(texts):
           try:
               result = process_text(text)
               results.append(result)
               
               # Add delay between requests
               if i < len(texts) - 1:
                   time.sleep(delay)
                   
           except Exception as e:
               print(f"Failed to process item {i}: {e}")
               results.append(None)
       
       return results
   ```

### Issue 3: Context Length Exceeded

**Symptoms:**
```
openai.error.InvalidRequestError: This model's maximum context length is X tokens
```

**Diagnosis:**
```python
import tiktoken

def diagnose_token_count(text, model="gpt-3.5-turbo"):
    """Count tokens in text"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        print(f"Text length: {len(text)} characters")
        print(f"Token count: {len(tokens)} tokens")
        
        # Model limits
        limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000
        }
        
        limit = limits.get(model, 4096)
        print(f"Model limit: {limit} tokens")
        print(f"Over limit: {len(tokens) > limit}")
        
        return len(tokens)
    except Exception as e:
        # Fallback estimation
        estimated = len(text.split()) * 1.3
        print(f"Estimated tokens: {estimated}")
        return estimated

# Usage
text = "Your long text here..."
token_count = diagnose_token_count(text, "gpt-3.5-turbo")
```

**Solutions:**
1. **Text Chunking:**
   ```python
   def chunk_text(text, max_tokens=3000, overlap=200):
       """Split text into overlapping chunks"""
       words = text.split()
       chunks = []
       
       # Rough estimate: 1 token ≈ 0.75 words
       max_words = int(max_tokens * 0.75)
       overlap_words = int(overlap * 0.75)
       
       for i in range(0, len(words), max_words - overlap_words):
           chunk = " ".join(words[i:i + max_words])
           chunks.append(chunk)
       
       return chunks
   
   # Usage
   long_text = "Your very long text..."
   chunks = chunk_text(long_text)
   
   all_results = []
   for chunk in chunks:
       result = process_text(chunk)
       all_results.append(result)
   ```

2. **Summarization First:**
   ```python
   def process_long_text(text):
       """Process long text by summarizing first"""
       
       if estimate_tokens(text) > 3000:
           # First pass: summarize
           summary = process_text(text, task="summarize", max_tokens=500)
           
           # Second pass: extract from summary
           entities = process_text(summary["result"], task="extract_entities")
           
           return {
               "summary": summary,
               "entities": entities,
               "processing_method": "summarize_first"
           }
       else:
           # Direct processing
           return process_text(text, task="extract_entities")
   ```

## Local Model Issues

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB. GPU 0 has a total capacity of Y GB
```

**Diagnosis:**
```python
import torch

def diagnose_gpu_memory():
    """Check GPU memory status"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.1f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1e9:.1f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1e9:.1f} GB")
    else:
        print("CUDA not available")

# Usage
diagnose_gpu_memory()
```

**Solutions:**
1. **Reduce Batch Size:**
   ```python
   # Before
   config = PipelineConfig(batch_size=16)
   
   # After
   config = PipelineConfig(batch_size=4)  # Or even 1
   ```

2. **Enable 8-bit Quantization:**
   ```python
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-7b-chat-hf",
       load_in_8bit=True,  # Reduces memory by ~50%
       device_map="auto"
   )
   ```

3. **Use Gradient Checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

4. **Clear Memory Between Batches:**
   ```python
   import gc
   import torch
   
   def clear_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   
   # Use between processing batches
   process_batch(batch1)
   clear_memory()
   process_batch(batch2)
   ```

5. **Switch to CPU:**
   ```python
   config = PipelineConfig(
       llm_model="microsoft/DialoGPT-medium",  # Smaller model
       device="cpu"
   )
   ```

### Issue 2: Model Loading Failures

**Symptoms:**
```
OSError: Can't load the model from 'model-name'. Make sure you have git and git-lfs installed.
```

**Diagnosis:**
```python
import subprocess
import os
from transformers import AutoModel

def diagnose_model_loading(model_name):
    """Diagnose model loading issues"""
    
    # Check git and git-lfs
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✓ Git is installed")
    except:
        print("✗ Git is not installed")
    
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("✓ Git LFS is installed")
    except:
        print("✗ Git LFS is not installed")
    
    # Check disk space
    disk_usage = os.statvfs('.')
    free_space_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
    print(f"Free disk space: {free_space_gb:.1f} GB")
    
    # Try loading model
    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        print(f"✓ Model {model_name} is cached locally")
    except:
        print(f"✗ Model {model_name} not in cache")
        
        # Try downloading
        try:
            model = AutoModel.from_pretrained(model_name)
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

# Usage
diagnose_model_loading("meta-llama/Llama-2-7b-chat-hf")
```

**Solutions:**
1. **Install Missing Dependencies:**
   ```bash
   # Install git and git-lfs
   sudo apt-get update
   sudo apt-get install git git-lfs
   
   # Or on macOS
   brew install git git-lfs
   
   # Initialize git-lfs
   git lfs install
   ```

2. **Clear and Rebuild Cache:**
   ```python
   import shutil
   from transformers import AutoModel
   
   # Clear cache
   cache_dir = "~/.cache/huggingface"
   shutil.rmtree(os.path.expanduser(cache_dir), ignore_errors=True)
   
   # Download fresh
   model = AutoModel.from_pretrained(
       "model-name",
       cache_dir="./model_cache"  # Custom cache location
   )
   ```

3. **Manual Download:**
   ```bash
   # Clone repository manually
   git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   
   # Then load from local path
   model = AutoModel.from_pretrained("./Llama-2-7b-chat-hf")
   ```

4. **Use Alternative Models:**
   ```python
   # Fallback model hierarchy
   model_candidates = [
       "meta-llama/Llama-2-7b-chat-hf",
       "microsoft/DialoGPT-medium",
       "google/flan-t5-small"
   ]
   
   for model_name in model_candidates:
       try:
           model = AutoModel.from_pretrained(model_name)
           print(f"Successfully loaded {model_name}")
           break
       except Exception as e:
           print(f"Failed to load {model_name}: {e}")
           continue
   ```

### Issue 3: Slow Processing on CPU

**Symptoms:**
- Very slow text processing
- High CPU usage
- Long startup times

**Diagnosis:**
```python
import time
import psutil
import threading

def monitor_performance():
    """Monitor CPU and memory usage"""
    
    def monitor():
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
            time.sleep(5)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def benchmark_processing(processor, text):
    """Benchmark processing speed"""
    
    start_time = time.time()
    result = processor.process_text(text)
    processing_time = time.time() - start_time
    
    tokens_per_second = len(text.split()) / processing_time
    
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Tokens per second: {tokens_per_second:.1f}")
    
    return result

# Usage
monitor_performance()
result = benchmark_processing(processor, "Your test text here")
```

**Solutions:**
1. **Optimize Model Selection:**
   ```python
   # CPU-optimized models
   CPU_FRIENDLY_MODELS = [
       "microsoft/DialoGPT-small",    # 117M parameters
       "google/flan-t5-small",        # 77M parameters
       "distilbert-base-uncased"      # 66M parameters
   ]
   
   # Use smaller, faster models for CPU
   processor = LLMProcessor(
       model_type="microsoft/DialoGPT-small",
       provider="huggingface"
   )
   ```

2. **Increase Workers for CPU:**
   ```python
   import multiprocessing
   
   # Use all CPU cores
   max_workers = multiprocessing.cpu_count()
   
   config = PipelineConfig(
       max_workers=max_workers,
       batch_size=1  # Process items individually
   )
   ```

3. **Enable Threading:**
   ```python
   # Set environment variables for better threading
   import os
   os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
   os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
   ```

## Vision Model Issues

### Issue 1: Image Loading Errors

**Symptoms:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**Diagnosis:**
```python
from PIL import Image
import os

def diagnose_image(image_path):
    """Diagnose image loading issues"""
    
    # Check file exists
    if not os.path.exists(image_path):
        print(f"✗ File does not exist: {image_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(image_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("✗ File is empty")
        return False
    
    # Check file extension
    _, ext = os.path.splitext(image_path)
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if ext.lower() not in supported_formats:
        print(f"✗ Unsupported format: {ext}")
        return False
    
    # Try to load image
    try:
        with Image.open(image_path) as img:
            print(f"✓ Image loaded successfully")
            print(f"  Size: {img.size}")
            print(f"  Mode: {img.mode}")
            print(f"  Format: {img.format}")
            return True
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

# Usage
diagnose_image("path/to/your/image.jpg")
```

**Solutions:**
1. **Convert Image Format:**
   ```python
   from PIL import Image
   
   def convert_image(input_path, output_path=None):
       """Convert image to supported format"""
       
       if output_path is None:
           base, _ = os.path.splitext(input_path)
           output_path = f"{base}_converted.jpg"
       
       try:
           with Image.open(input_path) as img:
               # Convert to RGB if necessary
               if img.mode != 'RGB':
                   img = img.convert('RGB')
               
               img.save(output_path, 'JPEG', quality=95)
               print(f"Converted {input_path} to {output_path}")
               return output_path
       except Exception as e:
           print(f"Conversion failed: {e}")
           return None
   ```

2. **Batch Image Validation:**
   ```python
   def validate_image_directory(directory):
       """Validate all images in directory"""
       
       valid_images = []
       invalid_images = []
       
       for filename in os.listdir(directory):
           filepath = os.path.join(directory, filename)
           
           if diagnose_image(filepath):
               valid_images.append(filepath)
           else:
               invalid_images.append(filepath)
       
       print(f"Valid images: {len(valid_images)}")
       print(f"Invalid images: {len(invalid_images)}")
       
       return valid_images, invalid_images
   ```

### Issue 2: VLM Model Memory Issues

**Symptoms:**
```
RuntimeError: CUDA out of memory (VLM models)
```

**Solutions:**
1. **Reduce Image Resolution:**
   ```python
   from PIL import Image
   
   def resize_image(image, max_size=512):
       """Resize image to reduce memory usage"""
       
       if isinstance(image, str):
           image = Image.open(image)
       
       # Calculate new size maintaining aspect ratio
       width, height = image.size
       if max(width, height) > max_size:
           if width > height:
               new_width = max_size
               new_height = int(height * max_size / width)
           else:
               new_height = max_size
               new_width = int(width * max_size / height)
           
           image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
       
       return image
   
   # Usage in VLM processor
   def process_image_memory_efficient(image_path):
       image = resize_image(image_path, max_size=384)  # Smaller for memory
       return vlm_processor.process_image(image, task="caption")
   ```

2. **Process Images Sequentially:**
   ```python
   def process_images_sequentially(image_paths):
       """Process images one by one to avoid memory buildup"""
       
       results = []
       
       for i, image_path in enumerate(image_paths):
           try:
               result = vlm_processor.process_image(image_path)
               results.append(result)
               
               # Clear memory after each image
               if i % 5 == 0:  # Clear every 5 images
                   clear_memory()
                   
           except Exception as e:
               print(f"Failed to process {image_path}: {e}")
               results.append(None)
       
       return results
   ```

## Network and Connectivity Issues

### Issue 1: Proxy/Firewall Issues

**Symptoms:**
```
ConnectionError: HTTPSConnectionPool
```

**Diagnosis:**
```python
import requests
import os

def diagnose_connectivity():
    """Test network connectivity"""
    
    # Test basic internet
    try:
        response = requests.get("https://httpbin.org/ip", timeout=10)
        print(f"✓ Internet connection: {response.status_code}")
    except Exception as e:
        print(f"✗ Internet connection failed: {e}")
    
    # Test OpenAI API
    try:
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        print(f"✓ OpenAI API reachable: {response.status_code}")
    except Exception as e:
        print(f"✗ OpenAI API unreachable: {e}")
    
    # Test Hugging Face
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        print(f"✓ Hugging Face reachable: {response.status_code}")
    except Exception as e:
        print(f"✗ Hugging Face unreachable: {e}")
    
    # Check proxy settings
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        value = os.getenv(var)
        if value:
            print(f"Proxy {var}: {value}")

# Usage
diagnose_connectivity()
```

**Solutions:**
1. **Configure Proxy:**
   ```python
   import os
   import requests
   
   # Set proxy for requests
   proxies = {
       'http': 'http://proxy.company.com:8080',
       'https': 'https://proxy.company.com:8080'
   }
   
   # For requests
   response = requests.get(url, proxies=proxies)
   
   # For transformers (set environment variable)
   os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
   os.environ['HTTPS_PROXY'] = 'https://proxy.company.com:8080'
   ```

2. **Use Local Mode:**
   ```python
   # Force local-only mode
   config = PipelineConfig(
       use_llm=True,
       llm_provider="huggingface",  # No API calls
       llm_model="microsoft/DialoGPT-medium",
       
       # Use cached models only
       local_files_only=True
   )
   ```

## Performance Optimization Issues

### Issue 1: Poor Model Accuracy

**Symptoms:**
- Low quality entity extraction
- Irrelevant results
- Missing important information

**Diagnosis:**
```python
def evaluate_extraction_quality(text, expected_entities):
    """Evaluate extraction quality"""
    
    result = processor.process_text(text, task="extract_entities")
    extracted = result.get("result", {})
    
    # Calculate precision/recall for each entity type
    for entity_type, expected in expected_entities.items():
        found = extracted.get(entity_type, [])
        
        if isinstance(found, list):
            found_texts = [item.get("text", item) if isinstance(item, dict) else item for item in found]
        else:
            found_texts = found
        
        true_positives = len(set(expected) & set(found_texts))
        precision = true_positives / len(found_texts) if found_texts else 0
        recall = true_positives / len(expected) if expected else 0
        
        print(f"{entity_type}:")
        print(f"  Expected: {expected}")
        print(f"  Found: {found_texts}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")

# Usage
test_text = "E. coli contamination in ground beef causes foodborne illness"
expected = {
    "pathogen": ["E. coli"],
    "food_product": ["ground beef"],
    "safety_concern": ["contamination", "foodborne illness"]
}

evaluate_extraction_quality(test_text, expected)
```

**Solutions:**
1. **Improve Prompts:**
   ```python
   def create_better_prompt(text):
       return f"""
   You are an expert food safety analyst. Extract entities from this text with high precision.
   
   Text: {text}
   
   Extract these entity types:
   - Pathogens: bacteria, viruses, parasites (e.g., E. coli, Salmonella)
   - Food Products: specific foods, ingredients (e.g., ground beef, lettuce)
   - Safety Concerns: contamination, illnesses, violations
   
   Return only valid entities you are confident about.
   Format as JSON: {{"pathogens": [...], "food_products": [...], "safety_concerns": [...]}}
   """
   ```

2. **Use Better Models:**
   ```python
   # Upgrade to higher quality model
   high_quality_config = PipelineConfig(
       llm_model="gpt-4",  # Instead of gpt-3.5-turbo
       entity_confidence_threshold=0.8,  # Higher threshold
       vlm_model="Salesforce/blip-image-captioning-large"  # Instead of base
   )
   ```

3. **Ensemble Approach:**
   ```python
   def ensemble_extraction(text):
       """Use multiple models and combine results"""
       
       # Use different models
       gpt35_result = processor_gpt35.process_text(text, "extract_entities")
       gpt4_result = processor_gpt4.process_text(text, "extract_entities")
       
       # Combine results (take intersection for high confidence)
       combined_entities = {}
       
       for entity_type in gpt35_result.get("result", {}):
           gpt35_entities = set(gpt35_result["result"].get(entity_type, []))
           gpt4_entities = set(gpt4_result["result"].get(entity_type, []))
           
           # High confidence: found by both models
           high_conf = list(gpt35_entities & gpt4_entities)
           
           # Medium confidence: found by either model
           medium_conf = list(gpt35_entities | gpt4_entities)
           
           combined_entities[entity_type] = {
               "high_confidence": high_conf,
               "medium_confidence": medium_conf
           }
       
       return combined_entities
   ```

This troubleshooting guide provides systematic approaches to diagnosing and resolving the most common issues users encounter when working with AI models in the pipeline.