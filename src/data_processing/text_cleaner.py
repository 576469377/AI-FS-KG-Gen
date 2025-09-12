"""
Text cleaning and preprocessing utilities
"""
import re
import string
from typing import List, Dict, Any, Optional
import unicodedata
from utils.logger import get_logger

logger = get_logger(__name__)

class TextCleaner:
    """
    Text cleaning and preprocessing utilities for food safety text
    """
    
    def __init__(self):
        """Initialize text cleaner"""
        # Common food safety abbreviations and their expansions
        self.abbreviations = {
            "FDA": "Food and Drug Administration",
            "USDA": "United States Department of Agriculture", 
            "CDC": "Centers for Disease Control and Prevention",
            "WHO": "World Health Organization",
            "HACCP": "Hazard Analysis Critical Control Points",
            "GMP": "Good Manufacturing Practices",
            "SSOP": "Sanitation Standard Operating Procedures",
            "CCP": "Critical Control Point",
            "pH": "potential of Hydrogen",
            "ppm": "parts per million",
            "CFU": "Colony Forming Units",
            "E. coli": "Escherichia coli",
            "S. aureus": "Staphylococcus aureus",
            "L. monocytogenes": "Listeria monocytogenes",
            "C. botulinum": "Clostridium botulinum"
        }
        
        # Food safety stopwords to potentially preserve
        self.food_safety_terms = {
            "pathogen", "bacteria", "virus", "contamination", "sanitize",
            "disinfect", "sterilize", "pasteurize", "allergen", "additive",
            "preservative", "spoilage", "rancid", "mold", "yeast", "toxin",
            "residue", "pesticide", "antibiotic", "hormone", "temperature",
            "refrigerate", "freeze", "thaw", "cook", "heat", "cool", "storage"
        }
        
        logger.info("TextCleaner initialized")
    
    def clean_text(self, text: str, preserve_food_terms: bool = True) -> str:
        """
        Clean and normalize text while preserving food safety terms
        
        Args:
            text: Input text to clean
            preserve_food_terms: Whether to preserve food safety specific terms
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Fix common encoding issues
        text = self._fix_encoding_issues(text)
        
        # Clean whitespace
        text = self._clean_whitespace(text)
        
        # Remove extra punctuation but preserve important ones
        text = self._clean_punctuation(text)
        
        # Normalize case while preserving important terms
        if preserve_food_terms:
            text = self._preserve_case_sensitive_terms(text)
        else:
            text = text.lower()
        
        return text.strip()
    
    def preprocess_for_ner(self, text: str) -> str:
        """
        Preprocess text specifically for Named Entity Recognition
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text, preserve_food_terms=True)
        
        # Split sentences properly
        text = self._fix_sentence_boundaries(text)
        
        # Fix common NER issues
        text = self._fix_ner_issues(text)
        
        return text
    
    def preprocess_for_llm(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Preprocess text for LLM input
        
        Args:
            text: Input text
            max_length: Maximum text length
        
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = self.clean_text(text)
        
        # Remove excessive repetition
        text = self._remove_repetition(text)
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Break at word boundary
        
        return text
    
    def extract_food_entities_candidates(self, text: str) -> List[str]:
        """
        Extract potential food entity candidates from text
        
        Args:
            text: Input text
        
        Returns:
            List of potential food entities
        """
        text = self.clean_text(text)
        
        # Patterns for food entities
        patterns = [
            # Scientific names (genus species)
            r'\b[A-Z][a-z]+ [a-z]+(?:\s+[a-z]+)?\b',
            # Chemical compounds
            r'\b[A-Z][a-z]*(?:-[A-Z]?[a-z]*)*\b',
            # Food products with numbers/measurements
            r'\b\w+(?:\s+\w+)*\s+\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|%|ppm)\b',
            # Capitalized food terms
            r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b'
        ]
        
        candidates = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            candidates.extend([match.group().strip() for match in matches])
        
        # Filter candidates
        filtered_candidates = []
        for candidate in candidates:
            if self._is_potential_food_entity(candidate):
                filtered_candidates.append(candidate)
        
        return list(set(filtered_candidates))
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common food safety abbreviations"""
        for abbrev, expansion in self.abbreviations.items():
            # Replace abbreviation with expansion (case insensitive)
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        # Common encoding fixes
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            'Â°': '°',
            'Â±': '±',
            'Â©': '©',
            'Â®': '®'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace tabs and newlines with spaces
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean punctuation while preserving important ones"""
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([()[\]{}])\s*', r' \1 ', text)
        
        # Remove extra spaces created
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _preserve_case_sensitive_terms(self, text: str) -> str:
        """Preserve case for important food safety terms"""
        # Create a mapping of lowercase to original case for food safety terms
        term_mapping = {}
        
        # Find all food safety terms and preserve their case
        for term in self.food_safety_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group()
                term_mapping[original.lower()] = original
        
        # Convert to lowercase
        text_lower = text.lower()
        
        # Restore important terms
        for lower_term, original_term in term_mapping.items():
            text_lower = text_lower.replace(lower_term, original_term)
        
        return text_lower
    
    def _fix_sentence_boundaries(self, text: str) -> str:
        """Fix sentence boundary issues"""
        # Add space after period if missing
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Fix abbreviations that might break sentence segmentation
        # Common abbreviations that shouldn't end sentences
        abbrevs = ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Inc', 'Ltd', 'Corp', 'etc', 'vs', 'e.g', 'i.e']
        for abbrev in abbrevs:
            text = re.sub(f'{re.escape(abbrev)}\\.', f'{abbrev}', text)
        
        return text
    
    def _fix_ner_issues(self, text: str) -> str:
        """Fix common NER preprocessing issues"""
        # Ensure proper spacing around parentheses for chemical names
        text = re.sub(r'(\w)\(', r'\1 (', text)
        text = re.sub(r'\)(\w)', r') \1', text)
        
        # Fix hyphenated compounds
        text = re.sub(r'(\w)-(\w)', r'\1-\2', text)  # Remove spaces in hyphenated words
        
        # Fix temperature and measurement units
        text = re.sub(r'(\d+)\s*°\s*([CF])', r'\1°\2', text)
        text = re.sub(r'(\d+)\s*(mg|g|kg|ml|l|ppm|%)', r'\1\2', text)
        
        return text
    
    def _remove_repetition(self, text: str) -> str:
        """Remove excessive repetition"""
        # Remove repeated words
        words = text.split()
        cleaned_words = []
        prev_word = None
        repeat_count = 0
        
        for word in words:
            if word.lower() == prev_word:
                repeat_count += 1
                if repeat_count < 2:  # Allow up to 2 repetitions
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
                repeat_count = 0
            prev_word = word.lower()
        
        return ' '.join(cleaned_words)
    
    def _is_potential_food_entity(self, candidate: str) -> bool:
        """Check if a candidate string is potentially a food entity"""
        candidate_lower = candidate.lower()
        
        # Skip common non-food words
        skip_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'up', 'down',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'
        }
        
        if candidate_lower in skip_words:
            return False
        
        # Skip very short or very long candidates
        if len(candidate) < 2 or len(candidate) > 50:
            return False
        
        # Skip candidates that are all numbers or all punctuation
        if candidate.isdigit() or all(c in string.punctuation for c in candidate):
            return False
        
        # Prefer candidates with food safety terms
        if any(term in candidate_lower for term in self.food_safety_terms):
            return True
        
        # Accept candidates that look like scientific names
        if re.match(r'^[A-Z][a-z]+ [a-z]+$', candidate):
            return True
        
        # Accept candidates with proper capitalization
        if candidate[0].isupper() and not candidate.isupper():
            return True
        
        return False

def clean_text_batch(texts: List[str], cleaner: Optional[TextCleaner] = None, **kwargs) -> List[str]:
    """
    Clean multiple texts in batch
    
    Args:
        texts: List of texts to clean
        cleaner: Optional TextCleaner instance
        **kwargs: Additional cleaning parameters
    
    Returns:
        List of cleaned texts
    """
    if cleaner is None:
        cleaner = TextCleaner()
    
    cleaned_texts = []
    for text in texts:
        try:
            cleaned = cleaner.clean_text(text, **kwargs)
            cleaned_texts.append(cleaned)
        except Exception as e:
            logger.warning(f"Failed to clean text: {str(e)}")
            cleaned_texts.append(text)  # Return original if cleaning fails
    
    return cleaned_texts