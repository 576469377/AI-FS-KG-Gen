"""
Text data loading and processing for AI-FS-KG-Gen pipeline
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import requests
from bs4 import BeautifulSoup

# Optional imports
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from utils.logger import get_logger
from utils.helpers import validate_file_path, clean_text, generate_hash

logger = get_logger(__name__)

class TextLoader:
    """
    Text data loader for various text formats and sources
    """
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize text loader
        
        Args:
            supported_formats: List of supported file formats
        """
        self.supported_formats = supported_formats or ['.txt', '.md', '.html', '.pdf']
        logger.info(f"TextLoader initialized with formats: {self.supported_formats}")
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load text content from a file
        
        Args:
            file_path: Path to the text file
        
        Returns:
            Dictionary containing text content and metadata
        """
        file_path = Path(file_path)
        
        if not validate_file_path(file_path, self.supported_formats):
            raise ValueError(f"Invalid file path or unsupported format: {file_path}")
        
        logger.info(f"Loading text file: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.pdf':
                content = self._load_pdf(file_path)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                content = self._load_html(file_path)
            else:
                content = self._load_text_file(file_path)
            
            metadata = {
                'source': str(file_path),
                'format': file_path.suffix.lower(),
                'size': file_path.stat().st_size,
                'hash': generate_hash(content),
                'length': len(content)
            }
            
            return {
                'content': clean_text(content),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Load all text files from a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
        
        Yields:
            Dictionary containing text content and metadata for each file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        logger.info(f"Loading text files from directory: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    yield self.load_file(file_path)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
                    continue
    
    def load_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Load text content from a web URL
        
        Args:
            url: URL to load content from
            headers: Optional HTTP headers
        
        Returns:
            Dictionary containing text content and metadata
        """
        logger.info(f"Loading content from URL: {url}")
        
        try:
            headers = headers or {'User-Agent': 'AI-FS-KG-Gen/1.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Extract text from HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = soup.get_text()
            
            metadata = {
                'source': url,
                'format': 'html',
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'hash': generate_hash(content),
                'length': len(content)
            }
            
            return {
                'content': clean_text(content),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load content from a plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_html(self, file_path: Path) -> str:
        """Load content from an HTML file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load content from a PDF file"""
        if not HAS_PDF:
            logger.warning(f"pdfplumber not available, cannot load PDF: {file_path}")
            raise ImportError("pdfplumber is required for PDF processing. Install with: pip install pdfplumber")
        
        content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content.append(text)
        
        return '\n'.join(content)

def load_text_batch(file_paths: List[str], loader: Optional[TextLoader] = None) -> List[Dict[str, Any]]:
    """
    Load multiple text files in batch
    
    Args:
        file_paths: List of file paths
        loader: Optional TextLoader instance
    
    Returns:
        List of loaded text documents
    """
    if loader is None:
        loader = TextLoader()
    
    documents = []
    
    for file_path in file_paths:
        try:
            document = loader.load_file(file_path)
            documents.append(document)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {str(e)}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} out of {len(file_paths)} files")
    return documents