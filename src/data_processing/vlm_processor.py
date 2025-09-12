"""
Vision Language Model processing for AI-FS-KG-Gen pipeline
"""
from typing import List, Dict, Any, Optional, Union
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModelForVision2Seq
)
import requests
from io import BytesIO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import get_model_config
from utils.logger import get_logger

logger = get_logger(__name__)

class VLMProcessor:
    """
    Vision Language Model processor for image understanding and analysis
    """
    
    def __init__(self, model_type: str = "blip-base", task_type: str = "captioning"):
        """
        Initialize VLM processor
        
        Args:
            model_type: Type of VLM model to use
            task_type: Primary task type (captioning, classification, qa)
        """
        self.model_type = model_type
        self.task_type = task_type
        self.config = get_model_config("vlm")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
        logger.info(f"VLMProcessor initialized with {model_type} for {task_type} on {self.device}")
    
    def _load_model(self):
        """Load the VLM model based on model type"""
        try:
            if self.model_type == "blip-base":
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            elif self.model_type == "blip-large":
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            elif self.model_type == "clip-vit":
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            else:
                # Try loading as generic vision-language model
                self.processor = AutoProcessor.from_pretrained(self.model_type)
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_type)
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load VLM model {self.model_type}: {str(e)}")
            raise
    
    def process_image(self, image: Union[Image.Image, str], task: str = "caption", **kwargs) -> Dict[str, Any]:
        """
        Process image using VLM for various tasks
        
        Args:
            image: PIL Image object or path to image
            task: Processing task (caption, classify, qa, analyze_food_safety)
            **kwargs: Additional task-specific parameters
        
        Returns:
            Dictionary containing processing results
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        if task == "caption":
            return self._generate_caption(image, **kwargs)
        elif task == "classify":
            return self._classify_image(image, **kwargs)
        elif task == "qa":
            return self._visual_qa(image, **kwargs)
        elif task == "analyze_food_safety":
            return self._analyze_food_safety(image, **kwargs)
        elif task == "extract_text":
            return self._extract_text_from_image(image, **kwargs)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _generate_caption(self, image: Image.Image, max_length: int = 50) -> Dict[str, Any]:
        """Generate image caption"""
        try:
            if "blip" in self.model_type:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=5,
                        early_stopping=True
                    )
                
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                # For other models, use a generic approach
                caption = "Image processing completed"
            
            return {
                "task": "caption",
                "result": caption,
                "confidence": 0.8  # Placeholder confidence
            }
            
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}")
            return {
                "task": "caption",
                "result": "",
                "error": str(e)
            }
    
    def _classify_image(self, image: Image.Image, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify image into categories"""
        if categories is None:
            categories = [
                "food product", "food packaging", "food preparation",
                "food storage", "contamination", "kitchen equipment",
                "laboratory testing", "food safety violation"
            ]
        
        try:
            if "clip" in self.model_type:
                # Use CLIP for zero-shot classification
                text_inputs = [f"a photo of {category}" for category in categories]
                
                inputs = self.processor(
                    text=text_inputs,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get best category
                best_idx = probs.argmax().item()
                confidence = probs[0][best_idx].item()
                
                return {
                    "task": "classify",
                    "result": categories[best_idx],
                    "confidence": confidence,
                    "all_scores": dict(zip(categories, probs[0].tolist()))
                }
            else:
                # Fallback for other models
                caption_result = self._generate_caption(image)
                caption = caption_result.get("result", "")
                
                # Simple keyword matching
                best_category = "unknown"
                for category in categories:
                    if category.replace(" ", "").lower() in caption.replace(" ", "").lower():
                        best_category = category
                        break
                
                return {
                    "task": "classify",
                    "result": best_category,
                    "confidence": 0.6,
                    "caption": caption
                }
                
        except Exception as e:
            logger.error(f"Image classification failed: {str(e)}")
            return {
                "task": "classify",
                "result": "unknown",
                "error": str(e)
            }
    
    def _visual_qa(self, image: Image.Image, question: str = "What food safety issues can you identify?") -> Dict[str, Any]:
        """Visual question answering"""
        try:
            if "blip" in self.model_type:
                inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=5,
                        early_stopping=True
                    )
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                # Fallback: combine caption with question
                caption_result = self._generate_caption(image)
                caption = caption_result.get("result", "")
                answer = f"Based on the image showing: {caption}"
            
            return {
                "task": "qa",
                "question": question,
                "result": answer,
                "confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"Visual QA failed: {str(e)}")
            return {
                "task": "qa",
                "question": question,
                "result": "",
                "error": str(e)
            }
    
    def _analyze_food_safety(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive food safety analysis of image"""
        results = {}
        
        # Generate caption
        caption_result = self._generate_caption(image, max_length=100)
        results["caption"] = caption_result
        
        # Classify image
        classification_result = self._classify_image(image)
        results["classification"] = classification_result
        
        # Ask food safety specific questions
        safety_questions = [
            "Are there any visible contamination signs?",
            "Is the food properly stored?",
            "Are hygiene standards being followed?",
            "Is the packaging intact?",
            "Are there expiration date concerns?"
        ]
        
        qa_results = []
        for question in safety_questions:
            qa_result = self._visual_qa(image, question)
            qa_results.append(qa_result)
        
        results["safety_qa"] = qa_results
        
        # Extract food safety insights
        insights = self._extract_safety_insights(results)
        results["insights"] = insights
        
        return {
            "task": "analyze_food_safety",
            "result": results,
            "summary": insights
        }
    
    def _extract_text_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text from image (OCR-like functionality)"""
        # This is a placeholder - in practice, you'd use OCR libraries like EasyOCR or Tesseract
        # For VLM, we can use caption to identify text elements
        
        text_questions = [
            "What text is visible in this image?",
            "Are there any labels or signs?",
            "What information is written on packages or containers?"
        ]
        
        text_results = []
        for question in text_questions:
            qa_result = self._visual_qa(image, question)
            text_results.append(qa_result)
        
        return {
            "task": "extract_text",
            "result": text_results,
            "extracted_text": [result.get("result", "") for result in text_results]
        }
    
    def _extract_safety_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract food safety insights from analysis results"""
        insights = []
        
        caption = analysis_results.get("caption", {}).get("result", "")
        classification = analysis_results.get("classification", {}).get("result", "")
        qa_results = analysis_results.get("safety_qa", [])
        
        # Analyze caption for safety keywords
        safety_keywords = [
            "contamination", "dirty", "expired", "moldy", "rotten",
            "clean", "fresh", "hygienic", "proper", "sealed"
        ]
        
        for keyword in safety_keywords:
            if keyword in caption.lower():
                if keyword in ["contamination", "dirty", "expired", "moldy", "rotten"]:
                    insights.append(f"Potential safety concern: {keyword} detected in image")
                else:
                    insights.append(f"Positive safety indicator: {keyword} observed")
        
        # Analyze classification results
        if "contamination" in classification or "violation" in classification:
            insights.append("Image classified as potential food safety issue")
        elif "food product" in classification or "food preparation" in classification:
            insights.append("Standard food-related image detected")
        
        # Analyze QA results
        for qa in qa_results:
            answer = qa.get("result", "").lower()
            if any(word in answer for word in ["yes", "contamination", "problem", "issue"]):
                insights.append(f"Safety concern identified: {qa.get('question', '')}")
            elif any(word in answer for word in ["no", "clean", "proper", "good"]):
                insights.append(f"Safety standard met: {qa.get('question', '')}")
        
        return insights if insights else ["No specific food safety issues identified"]

def process_image_batch(images: List[Union[Image.Image, str]], processor: Optional[VLMProcessor] = None,
                       task: str = "caption", **kwargs) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch
    
    Args:
        images: List of PIL Images or image paths
        processor: Optional VLMProcessor instance
        task: Processing task
        **kwargs: Additional task parameters
    
    Returns:
        List of processing results
    """
    if processor is None:
        processor = VLMProcessor()
    
    results = []
    for image in images:
        try:
            result = processor.process_image(image, task, **kwargs)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process image: {str(e)}")
            results.append({"task": task, "result": None, "error": str(e)})
    
    return results