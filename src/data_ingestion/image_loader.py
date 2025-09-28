"""
Image data loading and processing for AI-FS-KG-Gen pipeline
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from io import BytesIO
import requests
import hashlib
from utils.logger import get_logger
from utils.helpers import validate_file_path, generate_hash

# Optional PIL import
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = get_logger(__name__)

class ImageLoader:
    """
    Image data loader for various image formats
    """
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize image loader
        
        Args:
            supported_formats: List of supported image formats
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL/Pillow not available. Image processing functionality will be limited.")
        
        self.supported_formats = supported_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        logger.info(f"ImageLoader initialized with formats: {self.supported_formats}")
        self.pil_available = PIL_AVAILABLE
    
    def load_image(self, image_path: str) -> Dict[str, Any]:
        """
        Load image from file path
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary containing image data and metadata
        """
        if not self.pil_available:
            logger.error("PIL/Pillow not available. Cannot process images.")
            raise ImportError("PIL/Pillow is required for image processing. Install with: pip install pillow")
        
        image_path = Path(image_path)
        
        if not validate_file_path(image_path, self.supported_formats):
            raise ValueError(f"Invalid image path or unsupported format: {image_path}")
        
        logger.info(f"Loading image: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate hash for the image
            image_bytes = image_path.read_bytes()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            metadata = {
                'source': str(image_path),
                'format': image.format or image_path.suffix.lower()[1:],
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'file_size': image_path.stat().st_size,
                'hash': image_hash
            }
            
            return {
                'image': image,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Load all images from a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
        
        Yields:
            Dictionary containing image data and metadata for each image
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        logger.info(f"Loading images from directory: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    yield self.load_image(file_path)
                except Exception as e:
                    logger.warning(f"Skipping image {file_path} due to error: {str(e)}")
                    continue
    
    def load_from_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Load image from URL
        
        Args:
            url: URL to load image from
            headers: Optional HTTP headers
        
        Returns:
            Dictionary containing image data and metadata
        """
        logger.info(f"Loading image from URL: {url}")
        
        try:
            headers = headers or {'User-Agent': 'AI-FS-KG-Gen/1.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate hash for the image
            image_hash = hashlib.md5(response.content).hexdigest()
            
            metadata = {
                'source': url,
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'content_length': len(response.content),
                'content_type': response.headers.get('content-type', ''),
                'hash': image_hash
            }
            
            return {
                'image': image,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading image from URL {url}: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            max_size: Maximum size for the longer dimension
        
        Returns:
            Resized PIL Image object
        """
        width, height = image.size
        
        if max(width, height) <= max_size:
            return image
        
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract basic features from image
        
        Args:
            image: PIL Image object
        
        Returns:
            Dictionary containing image features
        """
        # Basic image statistics
        width, height = image.size
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        
        # Calculate basic statistics
        import numpy as np
        image_array = np.array(gray_image)
        
        features = {
            'dimensions': (width, height),
            'aspect_ratio': aspect_ratio,
            'mean_brightness': float(np.mean(image_array)),
            'std_brightness': float(np.std(image_array)),
            'min_brightness': int(np.min(image_array)),
            'max_brightness': int(np.max(image_array)),
            'total_pixels': width * height
        }
        
        return features

def load_image_batch(image_paths: List[str], loader: Optional[ImageLoader] = None) -> List[Dict[str, Any]]:
    """
    Load multiple images in batch
    
    Args:
        image_paths: List of image file paths
        loader: Optional ImageLoader instance
    
    Returns:
        List of loaded image documents
    """
    if loader is None:
        loader = ImageLoader()
    
    images = []
    
    for image_path in image_paths:
        try:
            image_data = loader.load_image(image_path)
            images.append(image_data)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {str(e)}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} out of {len(image_paths)} images")
    return images