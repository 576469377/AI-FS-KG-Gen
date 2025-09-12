"""
Large Language Model processing for AI-FS-KG-Gen pipeline
"""
import os
from typing import List, Dict, Any, Optional, Union
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import get_model_config
from utils.logger import get_logger
from utils.helpers import chunk_text, clean_text

logger = get_logger(__name__)

class LLMProcessor:
    """
    Large Language Model processor for text understanding and generation
    """
    
    def __init__(self, model_type: str = "gpt-3.5-turbo", provider: str = "openai"):
        """
        Initialize LLM processor
        
        Args:
            model_type: Type of LLM model to use
            provider: Model provider (openai, huggingface)
        """
        self.model_type = model_type
        self.provider = provider
        self.config = get_model_config("llm")
        
        # Initialize model based on provider
        if provider == "openai":
            self._init_openai()
        elif provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"LLMProcessor initialized with {provider} model: {model_type}")
    
    def _init_openai(self):
        """Initialize OpenAI API"""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        self.client = openai
    
    def _init_huggingface(self):
        """Initialize Hugging Face model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_type,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1
            )
            logger.info(f"Loaded Hugging Face model on {device}")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise
    
    def process_text(self, text: str, task: str = "summarize", **kwargs) -> Dict[str, Any]:
        """
        Process text using LLM for various tasks
        
        Args:
            text: Input text to process
            task: Processing task (summarize, extract_entities, classify, etc.)
            **kwargs: Additional task-specific parameters
        
        Returns:
            Dictionary containing processed results
        """
        text = clean_text(text)
        
        if len(text) > self.config.get("max_tokens", 2000) * 4:  # Rough token estimate
            # Process in chunks for long text
            return self._process_long_text(text, task, **kwargs)
        
        if task == "summarize":
            return self._summarize(text, **kwargs)
        elif task == "extract_entities":
            return self._extract_entities(text, **kwargs)
        elif task == "extract_relations":
            return self._extract_relations(text, **kwargs)
        elif task == "classify":
            return self._classify_text(text, **kwargs)
        elif task == "generate_knowledge":
            return self._generate_knowledge(text, **kwargs)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _process_long_text(self, text: str, task: str, **kwargs) -> Dict[str, Any]:
        """Process long text by chunking"""
        chunks = chunk_text(text, max_length=2000, overlap=200)
        results = []
        
        for chunk in chunks:
            try:
                result = self.process_text(chunk, task, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process chunk: {str(e)}")
                continue
        
        # Combine results
        return self._combine_chunk_results(results, task)
    
    def _summarize(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Summarize text"""
        prompt = f"""Please provide a concise summary of the following text, focusing on food safety aspects:

Text: {text}

Summary:"""
        
        if self.provider == "openai":
            response = self._call_openai(prompt, max_tokens=max_length)
            summary = response.strip()
        else:
            response = self.pipeline(prompt, max_length=len(prompt) + max_length, num_return_sequences=1)
            summary = response[0]['generated_text'][len(prompt):].strip()
        
        return {
            "task": "summarize",
            "result": summary,
            "original_length": len(text),
            "summary_length": len(summary)
        }
    
    def _extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract entities from text"""
        if entity_types is None:
            entity_types = ["Food_Product", "Ingredient", "Pathogen", "Allergen", "Chemical_Compound"]
        
        prompt = f"""Extract food safety entities from the following text. Focus on these entity types: {', '.join(entity_types)}

Text: {text}

Entities (format as JSON):"""
        
        if self.provider == "openai":
            response = self._call_openai(prompt, max_tokens=500)
        else:
            response = self.pipeline(prompt, max_length=len(prompt) + 500, num_return_sequences=1)
            response = response[0]['generated_text'][len(prompt):].strip()
        
        try:
            import json
            entities = json.loads(response)
        except:
            # Fallback: extract simple entities
            entities = self._extract_simple_entities(text)
        
        return {
            "task": "extract_entities",
            "result": entities,
            "entity_count": len(entities) if isinstance(entities, list) else sum(len(v) for v in entities.values()) if isinstance(entities, dict) else 0
        }
    
    def _extract_relations(self, text: str) -> Dict[str, Any]:
        """Extract relations between entities"""
        prompt = f"""Extract relationships between food safety entities in the following text. Format as triples (subject, predicate, object):

Text: {text}

Relations (format as JSON list):"""
        
        if self.provider == "openai":
            response = self._call_openai(prompt, max_tokens=500)
        else:
            response = self.pipeline(prompt, max_length=len(prompt) + 500, num_return_sequences=1)
            response = response[0]['generated_text'][len(prompt):].strip()
        
        try:
            import json
            relations = json.loads(response)
        except:
            relations = []
        
        return {
            "task": "extract_relations",
            "result": relations,
            "relation_count": len(relations)
        }
    
    def _classify_text(self, text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify text into categories"""
        if categories is None:
            categories = ["Food Safety Alert", "Regulation", "Research", "Product Information", "Risk Assessment"]
        
        prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}

Text: {text}

Category:"""
        
        if self.provider == "openai":
            response = self._call_openai(prompt, max_tokens=50)
            category = response.strip()
        else:
            response = self.pipeline(prompt, max_length=len(prompt) + 50, num_return_sequences=1)
            category = response[0]['generated_text'][len(prompt):].strip()
        
        return {
            "task": "classify",
            "result": category,
            "confidence": 0.8  # Placeholder confidence
        }
    
    def _generate_knowledge(self, text: str) -> Dict[str, Any]:
        """Generate knowledge statements from text"""
        prompt = f"""Generate factual knowledge statements from the following food safety text:

Text: {text}

Knowledge statements:"""
        
        if self.provider == "openai":
            response = self._call_openai(prompt, max_tokens=400)
        else:
            response = self.pipeline(prompt, max_length=len(prompt) + 400, num_return_sequences=1)
            response = response[0]['generated_text'][len(prompt):].strip()
        
        # Split into individual statements
        statements = [s.strip() for s in response.split('\n') if s.strip()]
        
        return {
            "task": "generate_knowledge",
            "result": statements,
            "statement_count": len(statements)
        }
    
    def _call_openai(self, prompt: str, max_tokens: int = 150) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.ChatCompletion.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": "You are an expert in food safety and knowledge extraction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.config.get("temperature", 0.7)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Simple entity extraction fallback"""
        from utils.helpers import extract_food_terms
        return extract_food_terms(text)
    
    def _combine_chunk_results(self, results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        """Combine results from multiple chunks"""
        if not results:
            return {"task": task, "result": None, "chunk_count": 0}
        
        if task == "summarize":
            combined_summary = " ".join([r["result"] for r in results if r.get("result")])
            return {
                "task": task,
                "result": combined_summary,
                "chunk_count": len(results)
            }
        elif task == "extract_entities":
            all_entities = []
            for r in results:
                if isinstance(r.get("result"), list):
                    all_entities.extend(r["result"])
                elif isinstance(r.get("result"), dict):
                    for entities in r["result"].values():
                        all_entities.extend(entities)
            return {
                "task": task,
                "result": list(set(all_entities)),  # Remove duplicates
                "chunk_count": len(results)
            }
        elif task == "extract_relations":
            all_relations = []
            for r in results:
                if isinstance(r.get("result"), list):
                    all_relations.extend(r["result"])
            return {
                "task": task,
                "result": all_relations,
                "chunk_count": len(results)
            }
        else:
            return {
                "task": task,
                "result": [r.get("result") for r in results],
                "chunk_count": len(results)
            }

def process_text_batch(texts: List[str], processor: Optional[LLMProcessor] = None, 
                      task: str = "summarize", **kwargs) -> List[Dict[str, Any]]:
    """
    Process multiple texts in batch
    
    Args:
        texts: List of texts to process
        processor: Optional LLMProcessor instance
        task: Processing task
        **kwargs: Additional task parameters
    
    Returns:
        List of processing results
    """
    if processor is None:
        processor = LLMProcessor()
    
    results = []
    for text in texts:
        try:
            result = processor.process_text(text, task, **kwargs)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process text: {str(e)}")
            results.append({"task": task, "result": None, "error": str(e)})
    
    return results