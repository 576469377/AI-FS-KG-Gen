"""
Relation extraction for food safety knowledge graph construction
"""
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from itertools import combinations
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import FOOD_SAFETY_RELATIONS
from utils.logger import get_logger
from utils.helpers import normalize_entity
from data_processing.text_cleaner import TextCleaner

logger = get_logger(__name__)

class RelationExtractor:
    """
    Relation extraction for food safety domain
    """
    
    def __init__(self, model_type: str = "pattern", custom_relations: Optional[List[str]] = None):
        """
        Initialize relation extractor
        
        Args:
            model_type: Type of model to use (pattern, transformer, spacy)
            custom_relations: Additional relation types to extract
        """
        self.model_type = model_type
        self.relation_types = FOOD_SAFETY_RELATIONS + (custom_relations or [])
        self.text_cleaner = TextCleaner()
        
        self._setup_patterns()
        if model_type != "pattern":
            self._load_model()
        
        logger.info(f"RelationExtractor initialized with {model_type} approach")
    
    def _setup_patterns(self):
        """Setup regex patterns for relation extraction"""
        self.relation_patterns = {
            "contains": [
                r'(\w+(?:\s+\w+)*)\s+(?:contains?|has|includes?|with)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:rich\s+in|source\s+of|loaded\s+with)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:made\s+)?(?:from|of|with)\s+(\w+(?:\s+\w+)*)'
            ],
            "causes": [
                r'(\w+(?:\s+\w+)*)\s+(?:causes?|leads?\s+to|results?\s+in|triggers?)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:responsible\s+for|linked\s+to)\s+(\w+(?:\s+\w+)*)'
            ],
            "prevents": [
                r'(\w+(?:\s+\w+)*)\s+(?:prevents?|stops?|blocks?|inhibits?)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:protects?\s+against|guards?\s+against)\s+(\w+(?:\s+\w+)*)'
            ],
            "regulates": [
                r'(\w+(?:\s+\w+)*)\s+(?:regulates?|controls?|governs?|manages?)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:regulated\s+by|controlled\s+by)\s+(\w+(?:\s+\w+)*)'
            ],
            "tests_for": [
                r'(?:test\w*|check\w*|screen\w*|analyze\w*)\s+(\w+(?:\s+\w+)*)\s+(?:for|to\s+detect)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:tested|checked|screened|analyzed)\s+(?:for|to\s+detect)\s+(\w+(?:\s+\w+)*)'
            ],
            "associated_with": [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:associated\s+with|related\s+to|linked\s+to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:and|&)\s+(\w+(?:\s+\w+)*)\s+(?:are\s+)?(?:associated|related|linked)'
            ],
            "stored_at": [
                r'(?:store|keep|maintain)\s+(\w+(?:\s+\w+)*)\s+(?:at|in|under)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:should\s+be\s+)?(?:stored|kept|maintained)\s+(?:at|in|under)\s+(\w+(?:\s+\w+)*)'
            ],
            "processed_by": [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:processed|treated|prepared)\s+(?:by|using|with)\s+(\w+(?:\s+\w+)*)',
                r'(?:process|treat|prepare)\s+(\w+(?:\s+\w+)*)\s+(?:by|using|with)\s+(\w+(?:\s+\w+)*)'
            ],
            "exceeds": [
                r'(\w+(?:\s+\w+)*)\s+(?:exceeds?|surpasses?|is\s+above)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:levels?\s+)?(?:exceed|surpass|are\s+above)\s+(\w+(?:\s+\w+)*)'
            ],
            "complies_with": [
                r'(\w+(?:\s+\w+)*)\s+(?:complies?\s+with|meets?|follows?|adheres?\s+to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:compliant\s+with|in\s+accordance\s+with)\s+(\w+(?:\s+\w+)*)'
            ]
        }
        
        # Dependency patterns for spaCy
        self.dependency_patterns = [
            # Subject-verb-object patterns
            {"label": "contains", "pattern": [{"DEP": "nsubj"}, {"POS": "VERB", "LEMMA": {"IN": ["contain", "have", "include"]}}, {"DEP": "dobj"}]},
            {"label": "causes", "pattern": [{"DEP": "nsubj"}, {"POS": "VERB", "LEMMA": {"IN": ["cause", "lead", "trigger"]}}, {"DEP": "dobj"}]},
            {"label": "prevents", "pattern": [{"DEP": "nsubj"}, {"POS": "VERB", "LEMMA": {"IN": ["prevent", "stop", "block", "inhibit"]}}, {"DEP": "dobj"}]},
        ]
    
    def _load_model(self):
        """Load the relation extraction model"""
        try:
            if self.model_type == "spacy":
                self.nlp = spacy.load("en_core_web_sm")
            elif self.model_type == "transformer":
                # Load a relation extraction model (placeholder - would need specific model)
                model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback to similarity
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Note: For actual relation extraction, you'd use a model trained on relation data
                self.similarity_pipeline = pipeline("feature-extraction", model=model_name)
        except Exception as e:
            logger.error(f"Failed to load relation extraction model: {str(e)}")
            raise
    
    def extract_relations(self, text: str, entities: Optional[Dict[str, List[Dict]]] = None, 
                         confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Extract relations from text
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            confidence_threshold: Minimum confidence for relation extraction
        
        Returns:
            List of extracted relations
        """
        # Preprocess text
        text = self.text_cleaner.preprocess_for_ner(text)
        
        relations = []
        
        # Extract using different methods based on model type
        if self.model_type == "pattern":
            relations.extend(self._extract_with_patterns(text, confidence_threshold))
        elif self.model_type == "spacy":
            relations.extend(self._extract_with_spacy(text, entities, confidence_threshold))
        elif self.model_type == "transformer":
            relations.extend(self._extract_with_transformer(text, entities, confidence_threshold))
        
        # Post-process relations
        relations = self._post_process_relations(relations)
        
        return relations
    
    def _extract_with_patterns(self, text: str, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Extract relations using regex patterns"""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        subject = groups[0].strip()
                        obj = groups[1].strip()
                        
                        # Skip if subject or object are too short or generic
                        if len(subject) < 2 or len(obj) < 2:
                            continue
                        
                        relations.append({
                            "subject": normalize_entity(subject),
                            "predicate": relation_type,
                            "object": normalize_entity(obj),
                            "confidence": 0.8,  # High confidence for pattern matches
                            "source": "pattern",
                            "text_span": match.group(),
                            "start": match.start(),
                            "end": match.end()
                        })
        
        return relations
    
    def _extract_with_spacy(self, text: str, entities: Optional[Dict], confidence_threshold: float) -> List[Dict[str, Any]]:
        """Extract relations using spaCy dependency parsing"""
        doc = self.nlp(text)
        relations = []
        
        # Method 1: Dependency pattern matching
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subject and object
                    subject = None
                    obj = None
                    
                    for child in token.children:
                        if child.dep_ == "nsubj":
                            subject = self._get_entity_span(child)
                        elif child.dep_ in ["dobj", "pobj"]:
                            obj = self._get_entity_span(child)
                    
                    if subject and obj:
                        relation_type = self._classify_relation_by_verb(token.lemma_)
                        if relation_type:
                            relations.append({
                                "subject": normalize_entity(subject),
                                "predicate": relation_type,
                                "object": normalize_entity(obj),
                                "confidence": 0.7,
                                "source": "spacy_dependency",
                                "verb": token.lemma_
                            })
        
        # Method 2: Entity pair relation classification
        if entities:
            entity_pairs = self._get_entity_pairs(entities, text)
            for pair in entity_pairs:
                relation = self._classify_entity_pair_relation(pair["subject"], pair["object"], 
                                                            pair["context"], doc)
                if relation and relation["confidence"] >= confidence_threshold:
                    relations.append(relation)
        
        return relations
    
    def _extract_with_transformer(self, text: str, entities: Optional[Dict], confidence_threshold: float) -> List[Dict[str, Any]]:
        """Extract relations using transformer-based approach"""
        # This is a simplified implementation
        # In practice, you'd use a model specifically trained for relation extraction
        relations = []
        
        if entities:
            entity_pairs = self._get_entity_pairs(entities, text)
            
            for pair in entity_pairs:
                # Use similarity or classification to determine relation
                relation_type = self._classify_relation_similarity(
                    pair["subject"], pair["object"], pair["context"]
                )
                
                if relation_type:
                    relations.append({
                        "subject": normalize_entity(pair["subject"]),
                        "predicate": relation_type,
                        "object": normalize_entity(pair["object"]),
                        "confidence": 0.6,  # Lower confidence for similarity-based
                        "source": "transformer_similarity",
                        "context": pair["context"]
                    })
        
        return relations
    
    def _get_entity_span(self, token) -> str:
        """Get the full entity span from a token"""
        # Extend to include compound nouns and adjectives
        start = token.i
        end = token.i + 1
        
        # Extend left for compound nouns and adjectives
        for i in range(token.i - 1, -1, -1):
            if token.doc[i].pos_ in ["NOUN", "ADJ", "PROPN"] and token.doc[i].dep_ in ["compound", "amod"]:
                start = i
            else:
                break
        
        # Extend right for compound nouns
        for i in range(token.i + 1, len(token.doc)):
            if token.doc[i].pos_ in ["NOUN", "PROPN"] and token.doc[i].dep_ in ["compound"]:
                end = i + 1
            else:
                break
        
        return token.doc[start:end].text
    
    def _classify_relation_by_verb(self, verb_lemma: str) -> Optional[str]:
        """Classify relation type based on verb"""
        verb_mapping = {
            "contain": "contains",
            "have": "contains",
            "include": "contains",
            "cause": "causes",
            "lead": "causes",
            "trigger": "causes",
            "prevent": "prevents",
            "stop": "prevents",
            "block": "prevents",
            "inhibit": "prevents",
            "regulate": "regulates",
            "control": "regulates",
            "manage": "regulates",
            "test": "tests_for",
            "check": "tests_for",
            "screen": "tests_for",
            "analyze": "tests_for",
            "store": "stored_at",
            "keep": "stored_at",
            "maintain": "stored_at",
            "process": "processed_by",
            "treat": "processed_by",
            "prepare": "processed_by"
        }
        
        return verb_mapping.get(verb_lemma)
    
    def _get_entity_pairs(self, entities: Dict[str, List[Dict]], text: str) -> List[Dict[str, Any]]:
        """Get all possible entity pairs for relation extraction"""
        all_entities = []
        
        # Flatten entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                all_entities.append({
                    "text": entity["text"],
                    "type": entity_type,
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0)
                })
        
        pairs = []
        
        # Create pairs of entities that are reasonably close to each other
        for i, entity1 in enumerate(all_entities):
            for entity2 in all_entities[i+1:]:
                # Only consider entities that are within a reasonable distance
                distance = abs(entity1["start"] - entity2["start"])
                if distance < 200:  # Within 200 characters
                    # Extract context between entities
                    start = min(entity1["start"], entity2["start"])
                    end = max(entity1["end"], entity2["end"])
                    context = text[max(0, start-50):min(len(text), end+50)]
                    
                    pairs.append({
                        "subject": entity1["text"],
                        "subject_type": entity1["type"],
                        "object": entity2["text"],
                        "object_type": entity2["type"],
                        "context": context,
                        "distance": distance
                    })
        
        return pairs
    
    def _classify_entity_pair_relation(self, subject: str, obj: str, context: str, doc) -> Optional[Dict[str, Any]]:
        """Classify relation between entity pair using context"""
        # Simple heuristic-based classification
        context_lower = context.lower()
        
        # Check for common relation indicators in context
        if any(word in context_lower for word in ["contain", "has", "include", "with", "rich in"]):
            return {
                "subject": normalize_entity(subject),
                "predicate": "contains",
                "object": normalize_entity(obj),
                "confidence": 0.6,
                "source": "context_heuristic"
            }
        elif any(word in context_lower for word in ["cause", "lead to", "result in", "trigger"]):
            return {
                "subject": normalize_entity(subject),
                "predicate": "causes",
                "object": normalize_entity(obj),
                "confidence": 0.6,
                "source": "context_heuristic"
            }
        elif any(word in context_lower for word in ["prevent", "stop", "block", "inhibit"]):
            return {
                "subject": normalize_entity(subject),
                "predicate": "prevents",
                "object": normalize_entity(obj),
                "confidence": 0.6,
                "source": "context_heuristic"
            }
        
        return None
    
    def _classify_relation_similarity(self, subject: str, obj: str, context: str) -> Optional[str]:
        """Classify relation using similarity (placeholder implementation)"""
        # This is a very simplified approach
        # In practice, you'd use embeddings and trained classifiers
        
        context_lower = context.lower()
        
        # Simple keyword-based classification
        if "contain" in context_lower or "include" in context_lower:
            return "contains"
        elif "cause" in context_lower or "lead" in context_lower:
            return "causes"
        elif "prevent" in context_lower or "stop" in context_lower:
            return "prevents"
        elif "regulate" in context_lower or "control" in context_lower:
            return "regulates"
        
        return None
    
    def _post_process_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extracted relations"""
        processed = []
        seen = set()
        
        for relation in relations:
            # Create a key for deduplication
            key = (
                relation["subject"].lower(),
                relation["predicate"],
                relation["object"].lower()
            )
            
            if key not in seen:
                seen.add(key)
                
                # Validate relation
                if self._is_valid_relation(relation):
                    processed.append(relation)
        
        return processed
    
    def _is_valid_relation(self, relation: Dict[str, Any]) -> bool:
        """Validate if a relation is meaningful"""
        subject = relation["subject"].strip()
        obj = relation["object"].strip()
        
        # Skip if subject or object are too short
        if len(subject) < 2 or len(obj) < 2:
            return False
        
        # Skip if subject and object are the same
        if subject.lower() == obj.lower():
            return False
        
        # Skip if confidence is too low
        if relation.get("confidence", 0) < 0.3:
            return False
        
        return True
    
    def extract_food_safety_relations(self, text: str, entities: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Extract food safety specific relations
        
        Args:
            text: Input text
            entities: Pre-extracted entities
        
        Returns:
            List of food safety relations
        """
        all_relations = self.extract_relations(text, entities)
        
        # Filter for food safety relevant relations
        food_safety_relations = []
        
        for relation in all_relations:
            if self._is_food_safety_relation(relation):
                food_safety_relations.append(relation)
        
        return food_safety_relations
    
    def _is_food_safety_relation(self, relation: Dict[str, Any]) -> bool:
        """Check if relation is relevant to food safety"""
        food_safety_keywords = [
            "food", "pathogen", "bacteria", "virus", "contamination", "safety",
            "allergen", "additive", "preservative", "nutrient", "temperature",
            "storage", "processing", "cooking", "hygiene", "sanitation"
        ]
        
        subject = relation["subject"].lower()
        obj = relation["object"].lower()
        
        return (any(keyword in subject for keyword in food_safety_keywords) or
                any(keyword in obj for keyword in food_safety_keywords))

def extract_relations_batch(texts: List[str], extractor: Optional[RelationExtractor] = None,
                           entities_list: Optional[List[Dict]] = None, **kwargs) -> List[List[Dict[str, Any]]]:
    """
    Extract relations from multiple texts in batch
    
    Args:
        texts: List of texts to process
        extractor: Optional RelationExtractor instance
        entities_list: Optional list of pre-extracted entities for each text
        **kwargs: Additional extraction parameters
    
    Returns:
        List of relation extraction results
    """
    if extractor is None:
        extractor = RelationExtractor()
    
    results = []
    for i, text in enumerate(texts):
        try:
            entities = entities_list[i] if entities_list and i < len(entities_list) else None
            relations = extractor.extract_relations(text, entities, **kwargs)
            results.append(relations)
        except Exception as e:
            logger.warning(f"Failed to extract relations from text {i}: {str(e)}")
            results.append([])
    
    return results