"""
Pipeline orchestration for AI-FS-KG-Gen
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger import setup_logger, get_logger
from utils.helpers import load_config, save_config
from data_ingestion import TextLoader, ImageLoader, StructuredDataLoader
from data_processing import TextCleaner

logger = get_logger(__name__)

# Optional imports for LLM and VLM processing
try:
    from data_processing import LLMProcessor
    HAS_LLM = True
except ImportError as e:
    HAS_LLM = False
    logger.warning(f"LLM processor not available: {e}")

try:
    from data_processing import VLMProcessor
    HAS_VLM = True
except ImportError as e:
    HAS_VLM = False
    logger.warning(f"VLM processor not available: {e}")

from knowledge_extraction import EntityExtractor, RelationExtractor
from knowledge_graph import KnowledgeGraphBuilder

# Set default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logger = get_logger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    # Input configuration
    input_sources: List[str]
    input_types: List[str]  # text, image, structured
    
    # Processing configuration
    use_llm: bool = True
    use_vlm: bool = True
    llm_model: str = "gpt-3.5-turbo"
    vlm_model: str = "blip-base"
    
    # Extraction configuration
    entity_extraction: bool = True
    relation_extraction: bool = True
    entity_confidence_threshold: float = 0.6
    relation_confidence_threshold: float = 0.6
    
    # Knowledge graph configuration
    kg_backend: str = "networkx"
    kg_output_format: str = "json"
    
    # Output configuration
    output_dir: str = str(OUTPUT_DIR)
    save_intermediate_results: bool = True
    
    # Processing configuration
    max_workers: int = 4
    batch_size: int = 10

class AIFSKGPipeline:
    """
    Main pipeline orchestrator for AI-FS-KG-Gen
    """
    
    def __init__(self, config: Union[PipelineConfig, str, Dict]):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration (PipelineConfig, config file path, or dict)
        """
        # Setup logging
        setup_logger()
        
        # Load configuration
        if isinstance(config, str):
            config_dict = load_config(config)
            self.config = PipelineConfig(**config_dict)
        elif isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config
        
        # Initialize components
        self._init_components()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "metadata": {
                "pipeline_id": datetime.now().isoformat(),
                "config": asdict(self.config)
            },
            "data": [],
            "entities": {},
            "relations": [],
            "knowledge_graph": None,
            "statistics": {}
        }
        
        logger.info("AI-FS-KG-Gen pipeline initialized")
    
    def _init_components(self):
        """Initialize pipeline components"""
        # Data loaders
        self.text_loader = TextLoader()
        self.image_loader = ImageLoader()
        self.structured_loader = StructuredDataLoader()
        
        # Text processing
        self.text_cleaner = TextCleaner()
        
        # AI processors
        self.llm_processor = None
        self.vlm_processor = None
        
        if self.config.use_llm and HAS_LLM:
            try:
                self.llm_processor = LLMProcessor(
                    model_type=self.config.llm_model,
                    provider="openai" if "gpt" in self.config.llm_model else "huggingface"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM processor: {e}")
                self.llm_processor = None
        elif self.config.use_llm and not HAS_LLM:
            logger.warning("LLM processing requested but LLM dependencies not available")
        
        if self.config.use_vlm and HAS_VLM:
            try:
                self.vlm_processor = VLMProcessor(
                    model_type=self.config.vlm_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize VLM processor: {e}")
                self.vlm_processor = None
        elif self.config.use_vlm and not HAS_VLM:
            logger.warning("VLM processing requested but VLM dependencies not available")
        
        # Knowledge extraction
        self.entity_extractor = None
        self.relation_extractor = None
        
        if self.config.entity_extraction:
            try:
                self.entity_extractor = EntityExtractor(model_type="spacy")
            except Exception as e:
                logger.warning(f"Failed to initialize entity extractor: {e}")
                self.entity_extractor = None
        
        if self.config.relation_extraction:
            try:
                self.relation_extractor = RelationExtractor(model_type="pattern")
            except Exception as e:
                logger.warning(f"Failed to initialize relation extractor: {e}")
                self.relation_extractor = None
        
        # Knowledge graph
        self.kg_builder = KnowledgeGraphBuilder(backend=self.config.kg_backend)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Pipeline results
        """
        logger.info("Starting AI-FS-KG-Gen pipeline execution")
        
        try:
            # Step 1: Data ingestion
            logger.info("Step 1: Data ingestion")
            self._ingest_data()
            
            # Step 2: Data processing
            logger.info("Step 2: Data processing")
            self._process_data()
            
            # Step 3: Knowledge extraction
            logger.info("Step 3: Knowledge extraction")
            self._extract_knowledge()
            
            # Step 4: Knowledge graph construction
            logger.info("Step 4: Knowledge graph construction")
            self._build_knowledge_graph()
            
            # Step 5: Output generation
            logger.info("Step 5: Output generation")
            self._generate_outputs()
            
            # Update final statistics
            self._update_statistics()
            
            logger.info("Pipeline execution completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self.results["error"] = str(e)
            raise
    
    def _ingest_data(self):
        """Ingest data from various sources"""
        for source in self.config.input_sources:
            source_path = Path(source)
            
            if source_path.is_file():
                self._ingest_file(source_path)
            elif source_path.is_dir():
                self._ingest_directory(source_path)
            elif source.startswith(("http://", "https://")):
                self._ingest_url(source)
            else:
                logger.warning(f"Unsupported source: {source}")
        
        logger.info(f"Ingested {len(self.results['data'])} data items")
    
    def _ingest_file(self, file_path: Path):
        """Ingest single file"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext in ['.txt', '.md', '.html', '.pdf']:
                data = self.text_loader.load_file(str(file_path))
                data['data_type'] = 'text'
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                data = self.image_loader.load_image(str(file_path))
                data['data_type'] = 'image'
            elif file_ext in ['.csv', '.json', '.xlsx', '.xls']:
                data = self.structured_loader.load_file(str(file_path))
                data['data_type'] = 'structured'
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return
            
            self.results['data'].append(data)
            
        except Exception as e:
            logger.warning(f"Failed to ingest file {file_path}: {e}")
    
    def _ingest_directory(self, dir_path: Path):
        """Ingest all files from directory"""
        for data_type in self.config.input_types:
            if data_type == 'text':
                for data in self.text_loader.load_directory(str(dir_path)):
                    data['data_type'] = 'text'
                    self.results['data'].append(data)
            elif data_type == 'image':
                for data in self.image_loader.load_directory(str(dir_path)):
                    data['data_type'] = 'image'
                    self.results['data'].append(data)
            elif data_type == 'structured':
                for data in self.structured_loader.load_directory(str(dir_path)):
                    data['data_type'] = 'structured'
                    self.results['data'].append(data)
    
    def _ingest_url(self, url: str):
        """Ingest content from URL"""
        try:
            if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                data = self.image_loader.load_from_url(url)
                data['data_type'] = 'image'
            else:
                data = self.text_loader.load_url(url)
                data['data_type'] = 'text'
            
            self.results['data'].append(data)
            
        except Exception as e:
            logger.warning(f"Failed to ingest URL {url}: {e}")
    
    def _process_data(self):
        """Process ingested data using AI models"""
        processed_data = []
        
        # Process in batches
        for i in range(0, len(self.results['data']), self.config.batch_size):
            batch = self.results['data'][i:i + self.config.batch_size]
            
            # Process batch with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for data_item in batch:
                    future = executor.submit(self._process_single_item, data_item)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        processed_item = future.result()
                        if processed_item:
                            processed_data.append(processed_item)
                    except Exception as e:
                        logger.warning(f"Failed to process data item: {e}")
        
        # Update results with processed data
        self.results['data'] = processed_data
        
        logger.info(f"Processed {len(processed_data)} data items")
    
    def _process_single_item(self, data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item"""
        try:
            data_type = data_item.get('data_type')
            
            if data_type == 'text':
                return self._process_text_item(data_item)
            elif data_type == 'image':
                return self._process_image_item(data_item)
            elif data_type == 'structured':
                return self._process_structured_item(data_item)
            
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            return None
    
    def _process_text_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process text data item"""
        content = data_item.get('content', '')
        
        # Clean text
        cleaned_content = self.text_cleaner.clean_text(content)
        data_item['cleaned_content'] = cleaned_content
        
        # LLM processing
        if self.llm_processor and cleaned_content:
            try:
                # Summarize
                summary = self.llm_processor.process_text(cleaned_content, task="summarize")
                data_item['summary'] = summary
                
                # Classify
                classification = self.llm_processor.process_text(cleaned_content, task="classify")
                data_item['classification'] = classification
                
            except Exception as e:
                logger.warning(f"LLM processing failed: {e}")
        
        return data_item
    
    def _process_image_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process image data item"""
        image = data_item.get('image')
        
        if self.vlm_processor and image:
            try:
                # Generate caption
                caption = self.vlm_processor.process_image(image, task="caption")
                data_item['caption'] = caption
                
                # Food safety analysis
                safety_analysis = self.vlm_processor.process_image(image, task="analyze_food_safety")
                data_item['safety_analysis'] = safety_analysis
                
            except Exception as e:
                logger.warning(f"VLM processing failed: {e}")
        
        return data_item
    
    def _process_structured_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process structured data item"""
        data = data_item.get('data')
        
        if hasattr(data, 'to_dict'):  # pandas DataFrame
            data_item['processed_data'] = data.to_dict('records')
        elif isinstance(data, list):
            data_item['processed_data'] = data
        elif isinstance(data, dict):
            data_item['processed_data'] = [data]
        
        return data_item
    
    def _extract_knowledge(self):
        """Extract entities and relations from processed data"""
        all_entities = {}
        all_relations = []
        
        for data_item in self.results['data']:
            try:
                # Extract text content for processing
                text_content = self._extract_text_content(data_item)
                
                if text_content and self.config.entity_extraction and self.entity_extractor:
                    # Extract entities
                    entities = self.entity_extractor.extract_entities(
                        text_content,
                        confidence_threshold=self.config.entity_confidence_threshold
                    )
                    
                    # Merge entities
                    for entity_type, entity_list in entities.items():
                        if entity_type not in all_entities:
                            all_entities[entity_type] = []
                        all_entities[entity_type].extend(entity_list)
                    
                    # Extract relations
                    if self.config.relation_extraction and self.relation_extractor:
                        relations = self.relation_extractor.extract_relations(
                            text_content,
                            entities,
                            confidence_threshold=self.config.relation_confidence_threshold
                        )
                        all_relations.extend(relations)
                
            except Exception as e:
                logger.warning(f"Knowledge extraction failed for item: {e}")
        
        self.results['entities'] = all_entities
        self.results['relations'] = all_relations
        
        logger.info(f"Extracted {sum(len(v) for v in all_entities.values())} entities and {len(all_relations)} relations")
    
    def _extract_text_content(self, data_item: Dict[str, Any]) -> str:
        """Extract text content from various data types"""
        data_type = data_item.get('data_type')
        
        if data_type == 'text':
            return data_item.get('cleaned_content', data_item.get('content', ''))
        elif data_type == 'image':
            # Use image caption as text
            caption = data_item.get('caption', {})
            return caption.get('result', '') if isinstance(caption, dict) else str(caption)
        elif data_type == 'structured':
            # Convert structured data to text
            processed_data = data_item.get('processed_data', [])
            text_parts = []
            
            for record in processed_data:
                if isinstance(record, dict):
                    text_parts.extend([str(v) for v in record.values() if v])
            
            return ' '.join(text_parts)
        
        return ''
    
    def _build_knowledge_graph(self):
        """Build knowledge graph from extracted knowledge"""
        build_stats = self.kg_builder.build_from_extractions(
            self.results['entities'],
            self.results['relations']
        )
        
        self.results['knowledge_graph'] = {
            "backend": self.config.kg_backend,
            "build_statistics": build_stats,
            "graph_statistics": self.kg_builder.get_statistics()
        }
        
        logger.info(f"Built knowledge graph with {build_stats}")
    
    def _generate_outputs(self):
        """Generate various output formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert any DataFrames to serializable format
        serializable_results = self._make_serializable(self.results.copy())
        
        # Save complete results
        results_file = self.output_dir / f"pipeline_results_{timestamp}.json"
        save_config(serializable_results, results_file)
        
        # Export knowledge graph
        kg_file = self.output_dir / f"knowledge_graph_{timestamp}.{self.config.kg_output_format}"
        self.kg_builder.export_graph(str(kg_file), self.config.kg_output_format)
        
        # Save intermediate results if requested
        if self.config.save_intermediate_results:
            # Entities
            entities_file = self.output_dir / f"entities_{timestamp}.json"
            save_config(self.results['entities'], entities_file)
            
            # Relations
            relations_file = self.output_dir / f"relations_{timestamp}.json"
            save_config(self.results['relations'], relations_file)
        
        logger.info(f"Generated outputs in {self.output_dir}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)  # Test if serializable
                return obj
            except TypeError:
                return str(obj)
    
    def _update_statistics(self):
        """Update final pipeline statistics"""
        self.results['statistics'] = {
            "data_items_processed": len(self.results['data']),
            "total_entities": sum(len(v) for v in self.results['entities'].values()),
            "total_relations": len(self.results['relations']),
            "entity_types": list(self.results['entities'].keys()),
            "relation_types": list(set(r.get('predicate', '') for r in self.results['relations'])),
            "knowledge_graph_stats": self.results.get('knowledge_graph', {}).get('graph_statistics', {})
        }

def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration"""
    return PipelineConfig(
        input_sources=[],
        input_types=["text", "image", "structured"],
        use_llm=True,
        use_vlm=True,
        entity_extraction=True,
        relation_extraction=True
    )

def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-FS-KG-Gen Pipeline")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--input", type=str, nargs="+", help="Input sources")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM processing")
    
    args = parser.parse_args()
    
    if args.config:
        config = PipelineConfig(**load_config(args.config))
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.input:
        config.input_sources = args.input
    if args.output:
        config.output_dir = args.output
    if args.no_llm:
        config.use_llm = False
    if args.no_vlm:
        config.use_vlm = False
    
    # Run pipeline
    pipeline = AIFSKGPipeline(config)
    results = pipeline.run()
    
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    print(f"Statistics: {results['statistics']}")

if __name__ == "__main__":
    main()