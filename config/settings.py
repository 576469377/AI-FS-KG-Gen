"""
Configuration settings for AI-FS-KG-Gen pipeline
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VLM_MODEL = "Salesforce/blip-image-captioning-base"

# Knowledge Graph settings
KG_DATABASE_URL = os.getenv("KG_DATABASE_URL", "bolt://localhost:7687")
KG_USERNAME = os.getenv("KG_USERNAME", "neo4j")
KG_PASSWORD = os.getenv("KG_PASSWORD", "password")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Processing settings
BATCH_SIZE = 32
MAX_TEXT_LENGTH = 8000
MAX_WORKERS = 4

# Food safety specific entity types
FOOD_SAFETY_ENTITIES = [
    "Food_Product",
    "Ingredient",
    "Chemical_Compound",
    "Pathogen",
    "Allergen",
    "Nutrient",
    "Safety_Standard",
    "Regulation",
    "Testing_Method",
    "Risk_Factor",
    "Contamination_Source",
    "Processing_Method",
    "Storage_Condition",
    "Temperature_Range",
    "pH_Level",
    "Microorganism",
    "Toxin",
    "Additive",
    "Preservative",
    "Certification"
]

# Relation types for food safety knowledge graph
FOOD_SAFETY_RELATIONS = [
    "contains",
    "causes",
    "prevents",
    "regulates",
    "tests_for",
    "associated_with",
    "derived_from",
    "processed_by",
    "stored_at",
    "tolerates",
    "exceeds",
    "complies_with",
    "certified_by",
    "requires",
    "inhibits",
    "activates",
    "interacts_with",
    "classified_as",
    "measured_by",
    "indicates"
]

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get configuration for specific model type"""
    configs = {
        "llm": {
            "model_name": DEFAULT_LLM_MODEL,
            "temperature": 0.7,
            "max_tokens": 2000,
            "api_key": OPENAI_API_KEY
        },
        "embedding": {
            "model_name": DEFAULT_EMBEDDING_MODEL,
            "device": "cpu",
            "batch_size": BATCH_SIZE
        },
        "vlm": {
            "model_name": DEFAULT_VLM_MODEL,
            "device": "cpu",
            "batch_size": 8
        }
    }
    return configs.get(model_type, {})

def get_kg_config() -> Dict[str, Any]:
    """Get knowledge graph configuration"""
    return {
        "database_url": KG_DATABASE_URL,
        "username": KG_USERNAME,
        "password": KG_PASSWORD,
        "entity_types": FOOD_SAFETY_ENTITIES,
        "relation_types": FOOD_SAFETY_RELATIONS
    }