"""
Entity extraction for food safety knowledge graph construction
"""
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import FOOD_SAFETY_ENTITIES
from utils.logger import get_logger
from utils.helpers import normalize_entity
from data_processing.text_cleaner import TextCleaner

logger = get_logger(__name__)

# Optional imports
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    spacy = None
    HAS_SPACY = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    AutoModelForTokenClassification = None
    pipeline = None
    HAS_TRANSFORMERS = False

class EntityExtractor:
    """
    Entity extraction for food safety domain
    """
    
    def __init__(self, model_type: str = "spacy", custom_entities: Optional[List[str]] = None):
        """
        Initialize entity extractor
        
        Args:
            model_type: Type of model to use (spacy, biobert, custom)
            custom_entities: Additional entity types to extract
        """
        self.model_type = model_type
        self.entity_types = FOOD_SAFETY_ENTITIES + (custom_entities or [])
        self.text_cleaner = TextCleaner()
        
        self._load_model()
        self._setup_patterns()
        
        logger.info(f"EntityExtractor initialized with {model_type} model")
    
    def _load_model(self):
        """Load the NER model"""
        try:
            if self.model_type == "spacy":
                if not HAS_SPACY:
                    logger.warning("spaCy not available. Using pattern-based extraction only.")
                    self.model_type = "pattern"
                    self.nlp = None
                    return
                    
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    # Add custom entity types to spacy
                    if "food_safety" not in self.nlp.pipe_names:
                        ruler = self.nlp.add_pipe("entity_ruler", name="food_safety")
                        # Add patterns after creating the ruler (this will be called in _setup_patterns)
                        self._ruler = ruler
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found. Using pattern-based extraction only.")
                    self.model_type = "pattern"
                    self.nlp = None
            
            elif self.model_type == "biobert":
                if not HAS_TRANSFORMERS:
                    logger.warning("Transformers not available. Using pattern-based extraction only.")
                    self.model_type = "pattern"
                    self.nlp = None
                    return
                    
                self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
                self.model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple"
                )
            
            else:
                # Default to pattern-based
                self.model_type = "pattern"
                self.nlp = None
                
        except Exception as e:
            logger.warning(f"Failed to load NER model, falling back to pattern-based extraction: {str(e)}")
            self.model_type = "pattern"
            self.nlp = None
    
    def _setup_patterns(self):
        """Setup regex patterns for food safety entities"""
        self.patterns = {
            "temperature": [
                r'\b\d+\s*(?:°[CF]|degrees?\s+(?:celsius|fahrenheit|centigrade))\b',
                r'\b(?:freezing|frozen|room\s+temperature|ambient\s+temperature)\b'
            ],
            "ph_level": [
                r'\bpH\s*\d+(?:\.\d+)?\b',
                r'\b(?:acidic|alkaline|neutral|basic)\s*pH\b'
            ],
            "microorganism": [
                # Specific bacteria names
                r'\bE\.?\s*coli\b',
                r'\bSalmonella\w*\b',
                r'\bListeria\w*\b',
                r'\bStaphylococcus\w*\b',
                r'\bClostridium\w*\b',
                r'\bCampylobacter\w*\b',
                r'\bShigella\w*\b',
                r'\bVibrio\w*\b',
                r'\bBacillus\w*\b',
                # Scientific names (genus species) - more specific pattern
                r'\b(?:Escherichia|Salmonella|Listeria|Staphylococcus|Clostridium|Campylobacter|Shigella|Vibrio|Bacillus)\s+[a-z]+\b',
                # General bacterial terms
                r'\bbacteri[ao]\b',
                r'\bpathogen[s]?\b',
                r'\bmicroorganism[s]?\b'
            ],
            "food_product": [
                r'\b(?:chicken|beef|pork|fish|seafood|poultry|meat|dairy|milk|cheese|eggs?|vegetables?|fruits?)\b',
                r'\b(?:ground\s+beef|deli\s+meat|soft\s+cheese|unpasteurized\s+milk)\b'
            ],
            "chemical_compound": [
                # Be more specific for chemical compounds
                r'\b[A-Z][a-z]*(?:-[A-Z]?[a-z]*){2,}\b',  # Hyphenated compounds with at least 2 hyphens
                r'\b(?:sodium|potassium|calcium|chloride|nitrate|sulfate|phosphate)\s+\w+\b',
                r'\b\w+\s+(?:acid|salt|compound|chemical|solution)\b'
            ],
            "measurement": [
                r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|ppm|ppb|%)\b',
                r'\b\d+(?:\.\d+)?\s*(?:milligrams?|grams?|kilograms?|milliliters?|liters?|percent)\b'
            ],
            "allergen": [
                r'\b(?:peanuts?|tree\s+nuts?|milk|eggs?|fish|shellfish|soy|wheat|sesame)\b',
                r'\b(?:gluten|lactose|casein|albumin)\b'
            ],
            "additive": [
                r'\bE\d{3,4}\b',  # E-numbers
                r'\b(?:preservative|antioxidant|emulsifier|stabilizer|colorant|flavoring)\b'
            ],
            "storage_condition": [
                r'\b(?:refrigerat\w+|stor\w+|kept|maintain\w*)\b',
                r'\b(?:cold\s+storage|dry\s+storage|cool\s+place)\b'
            ],
            "safety_concern": [
                r'\b(?:contamination|outbreak|recall|warning|alert|illness|poisoning)\b',
                r'\b(?:unsafe|hazardous|dangerous|risky|toxic|spoiled|rotten|expired)\b'
            ]
        }
        
        # Add patterns to spaCy entity ruler if available
        if hasattr(self, '_ruler') and self._ruler is not None:
            spacy_patterns = []
            for entity_type, regex_patterns in self.patterns.items():
                for pattern in regex_patterns:
                    # Convert entity type to uppercase for consistency with spaCy
                    label = entity_type.upper()
                    # Add regex patterns to spaCy (note: spaCy requires pattern dict format)
                    spacy_patterns.append({
                        "label": label,
                        "pattern": [{"TEXT": {"REGEX": pattern}}]
                    })
            
            if spacy_patterns:
                self._ruler.add_patterns(spacy_patterns)
                logger.info(f"Added {len(spacy_patterns)} patterns to spaCy entity ruler")
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for entity extraction
        
        Returns:
            Dictionary of extracted entities by type
        """
        # Preprocess text
        text = self.text_cleaner.preprocess_for_ner(text)
        
        entities = {}
        
        # Extract using the loaded model
        if self.model_type == "spacy" and self.nlp:
            entities.update(self._extract_with_spacy(text, confidence_threshold))
        elif self.model_type == "biobert":
            entities.update(self._extract_with_biobert(text, confidence_threshold))
        
        # Extract using regex patterns
        pattern_entities = self._extract_with_patterns(text)
        
        # Merge pattern entities with model entities
        for entity_type, entity_list in pattern_entities.items():
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].extend(entity_list)
        
        # Post-process and deduplicate
        entities = self._post_process_entities(entities)
        
        return entities
    
    def _extract_with_spacy(self, text: str, confidence_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                if entity_type not in entities:
                    entities[entity_type] = []
                
                entities[entity_type].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8,  # SpaCy doesn't provide confidence scores
                    "label": ent.label_
                })
        
        return entities
    
    def _extract_with_biobert(self, text: str, confidence_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using BioBERT"""
        try:
            results = self.ner_pipeline(text)
            entities = {}
            
            for result in results:
                if result["score"] >= confidence_threshold:
                    entity_type = self._map_biobert_label(result["entity_group"])
                    if entity_type:
                        if entity_type not in entities:
                            entities[entity_type] = []
                        
                        entities[entity_type].append({
                            "text": result["word"],
                            "start": result["start"],
                            "end": result["end"],
                            "confidence": result["score"],
                            "label": result["entity_group"]
                        })
            
            return entities
            
        except Exception as e:
            logger.warning(f"BioBERT extraction failed: {str(e)}")
            return {}
    
    def _extract_with_patterns(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using regex patterns"""
        entities = {}
        
        for entity_type, patterns in self.patterns.items():
            if entity_type not in entities:
                entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities[entity_type].append({
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9,  # High confidence for pattern matches
                        "label": entity_type
                    })
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy labels to food safety entity types"""
        # First check our custom entity labels (uppercase from patterns)
        custom_mapping = {
            "TEMPERATURE": "temperature",
            "PH_LEVEL": "ph_level", 
            "MICROORGANISM": "microorganism",
            "FOOD_PRODUCT": "food_product",
            "CHEMICAL_COMPOUND": "chemical_compound",
            "MEASUREMENT": "measurement",
            "ALLERGEN": "allergen",
            "ADDITIVE": "additive",
            "STORAGE_CONDITION": "storage_condition",
            "SAFETY_CONCERN": "safety_concern"
        }
        
        if label in custom_mapping:
            return custom_mapping[label]
        
        # Standard spaCy labels
        standard_mapping = {
            "PERSON": None,  # Usually not relevant for food safety
            "ORG": "organization",
            "GPE": "location",
            "PRODUCT": "food_product",
            "SUBSTANCE": "chemical_compound",
            "CHEMICAL": "chemical_compound",
            "DISEASE": "pathogen",
            "MONEY": None,
            "PERCENT": "measurement",
            "QUANTITY": "measurement",
            "DATE": "date",
            "TIME": "time"
        }
        return standard_mapping.get(label)
    
    def _map_biobert_label(self, label: str) -> Optional[str]:
        """Map BioBERT labels to food safety entity types"""
        # BioBERT typically uses BIO tagging
        if label.startswith("B-") or label.startswith("I-"):
            base_label = label[2:]
        else:
            base_label = label
        
        mapping = {
            "CHEMICAL": "chemical_compound",
            "DISEASE": "pathogen",
            "GENE": None,  # Usually not relevant
            "PROTEIN": "nutrient",
            "CELLLINE": None,
            "CELLTYPE": None,
            "DNA": None,
            "RNA": None,
            "ORGANISM": "microorganism"
        }
        return mapping.get(base_label.upper())
    
    def _post_process_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Post-process extracted entities"""
        processed = {}
        
        for entity_type, entity_list in entities.items():
            # Remove duplicates and normalize
            seen = set()
            unique_entities = []
            
            for entity in entity_list:
                normalized_text = normalize_entity(entity["text"])
                
                # Skip very short or invalid entities
                if len(normalized_text) < 2:
                    continue
                
                # Create a key for deduplication
                key = (normalized_text, entity["start"], entity["end"])
                
                if key not in seen:
                    seen.add(key)
                    entity["normalized_text"] = normalized_text
                    unique_entities.append(entity)
            
            if unique_entities:
                processed[entity_type] = unique_entities
        
        return processed
    
    def extract_food_products(self, text: str) -> List[Dict[str, Any]]:
        """
        Specifically extract food products from text
        
        Args:
            text: Input text
        
        Returns:
            List of food product entities
        """
        entities = self.extract_entities(text)
        food_products = entities.get("food_product", [])
        
        # Also look for candidates in other entity types
        candidates = self.text_cleaner.extract_food_entities_candidates(text)
        
        for candidate in candidates:
            # Check if it's not already in food_products
            if not any(normalize_entity(candidate) == normalize_entity(fp["text"]) 
                      for fp in food_products):
                food_products.append({
                    "text": candidate,
                    "normalized_text": normalize_entity(candidate),
                    "confidence": 0.6,
                    "label": "food_product_candidate"
                })
        
        return food_products
    
    def extract_safety_concerns(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract food safety concerns and risks
        
        Args:
            text: Input text
        
        Returns:
            List of safety concern entities
        """
        safety_patterns = [
            r'\b(?:contamination|contaminated|outbreak|recall|warning|alert)\b',
            r'\b(?:unsafe|hazardous|dangerous|risky|toxic|poisonous)\b',
            r'\b(?:spoiled|rotten|moldy|expired|stale)\b',
            r'\b(?:foodborne|illness|sickness|disease|infection)\b'
        ]
        
        concerns = []
        for pattern in safety_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                concerns.append({
                    "text": match.group(),
                    "context": context,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8,
                    "label": "safety_concern"
                })
        
        return concerns

def extract_entities_batch(texts: List[str], extractor: Optional[EntityExtractor] = None, 
                          **kwargs) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract entities from multiple texts in batch
    
    Args:
        texts: List of texts to process
        extractor: Optional EntityExtractor instance
        **kwargs: Additional extraction parameters
    
    Returns:
        List of entity extraction results
    """
    if extractor is None:
        extractor = EntityExtractor()
    
    results = []
    for text in texts:
        try:
            entities = extractor.extract_entities(text, **kwargs)
            results.append(entities)
        except Exception as e:
            logger.warning(f"Failed to extract entities from text: {str(e)}")
            results.append({})
    
    return results