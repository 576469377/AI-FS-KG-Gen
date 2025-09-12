"""
Helper functions and utilities for AI-FS-KG-Gen pipeline
"""
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def generate_hash(text: str) -> str:
    """
    Generate MD5 hash for text content
    
    Args:
        text: Input text
    
    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
    
    return text

def chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        max_length: Maximum length per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within the last 100 characters
            sentence_end = text.rfind('.', end - 100, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            break
    
    return chunks

def normalize_entity(entity: str) -> str:
    """
    Normalize entity name for consistency
    
    Args:
        entity: Entity name
    
    Returns:
        Normalized entity name
    """
    # Convert to lowercase and remove extra spaces
    entity = entity.lower().strip()
    
    # Replace underscores and hyphens with spaces
    entity = re.sub(r'[_-]', ' ', entity)
    
    # Remove extra spaces
    entity = re.sub(r'\s+', ' ', entity)
    
    return entity

def extract_food_terms(text: str) -> List[str]:
    """
    Extract potential food-related terms from text
    
    Args:
        text: Input text
    
    Returns:
        List of potential food terms
    """
    # Basic food-related keywords pattern
    food_patterns = [
        r'\b(?:food|ingredient|product|item|substance|compound|chemical|nutrient|vitamin|mineral|protein|carbohydrate|fat|fiber)\b',
        r'\b(?:bacteria|virus|pathogen|microorganism|contamination|toxin|allergen|additive|preservative)\b',
        r'\b(?:temperature|pH|acidity|moisture|humidity|storage|processing|cooking|preparation)\b',
        r'\b(?:safety|quality|standard|regulation|certification|inspection|testing|analysis)\b'
    ]
    
    terms = []
    for pattern in food_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        terms.extend([match.group() for match in matches])
    
    return list(set(terms))

def validate_file_path(file_path: Union[str, Path], required_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file path and extension
    
    Args:
        file_path: Path to file
        required_extensions: List of required file extensions
    
    Returns:
        True if valid, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    if not file_path.is_file():
        return False
    
    if required_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in required_extensions]:
            return False
    
    return True

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters
    
    Args:
        filename: Original filename
    
    Returns:
        Safe filename
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "untitled"
    
    return filename