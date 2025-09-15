# API Reference

This document provides a comprehensive reference for the AI-FS-KG-Gen API.

## Pipeline Module

### AIFSKGPipeline

Main pipeline orchestrator for AI-FS-KG-Gen.

```python
from pipeline import AIFSKGPipeline, PipelineConfig

pipeline = AIFSKGPipeline(config)
results = pipeline.run()
```

#### Constructor

```python
AIFSKGPipeline(config: Union[PipelineConfig, str, Dict])
```

**Parameters:**
- `config`: Pipeline configuration (PipelineConfig object, config file path, or dict)

#### Methods

##### run()

Execute the complete pipeline.

```python
def run() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Pipeline results containing:
  - `metadata`: Pipeline metadata and configuration
  - `data`: Processed data items
  - `entities`: Extracted entities by type
  - `relations`: Extracted relations
  - `knowledge_graph`: Knowledge graph information
  - `statistics`: Processing statistics

**Raises:**
- `ValueError`: If configuration is invalid
- `Exception`: If pipeline execution fails

### PipelineConfig

Configuration dataclass for pipeline execution.

```python
@dataclass
class PipelineConfig:
    input_sources: List[str]
    input_types: List[str]
    use_llm: bool = True
    use_vlm: bool = True
    llm_model: str = "gpt-3.5-turbo"
    vlm_model: str = "blip-base"
    entity_extraction: bool = True
    relation_extraction: bool = True
    entity_confidence_threshold: float = 0.6
    relation_confidence_threshold: float = 0.6
    kg_backend: str = "networkx"
    kg_output_format: str = "json"
    output_dir: str = str(OUTPUT_DIR)
    save_intermediate_results: bool = True
    max_workers: int = 4
    batch_size: int = 10
```

**Parameters:**
- `input_sources`: List of input file paths, directories, or URLs
- `input_types`: List of data types to process (`["text", "image", "structured"]`)
- `use_llm`: Enable Large Language Model processing
- `use_vlm`: Enable Vision Language Model processing
- `llm_model`: LLM model name
- `vlm_model`: VLM model name
- `entity_extraction`: Enable entity extraction
- `relation_extraction`: Enable relation extraction
- `entity_confidence_threshold`: Minimum confidence for entities (0.0-1.0)
- `relation_confidence_threshold`: Minimum confidence for relations (0.0-1.0)
- `kg_backend`: Knowledge graph backend (`"networkx"`, `"neo4j"`, `"rdf"`)
- `kg_output_format`: Output format (`"json"`, `"gexf"`, `"rdf"`)
- `output_dir`: Output directory path
- `save_intermediate_results`: Save intermediate processing results
- `max_workers`: Maximum number of parallel workers
- `batch_size`: Processing batch size

## Data Ingestion Module

### TextLoader

Handles text file loading and processing.

```python
from data_ingestion import TextLoader

loader = TextLoader()
data = loader.load_file("path/to/file.txt")
```

#### Methods

##### load_file(file_path: str) -> Dict[str, Any]

Load a single text file.

**Parameters:**
- `file_path`: Path to text file

**Returns:**
- `Dict[str, Any]`: Data dictionary with content and metadata

##### load_directory(directory_path: str) -> List[Dict[str, Any]]

Load all text files from a directory.

**Parameters:**
- `directory_path`: Path to directory

**Returns:**
- `List[Dict[str, Any]]`: List of data dictionaries

##### load_url(url: str) -> Dict[str, Any]

Load text content from URL.

**Parameters:**
- `url`: URL to fetch content from

**Returns:**
- `Dict[str, Any]`: Data dictionary with content and metadata

### ImageLoader

Handles image file loading and processing.

```python
from data_ingestion import ImageLoader

loader = ImageLoader()
data = loader.load_image("path/to/image.jpg")
```

#### Methods

##### load_image(image_path: str) -> Dict[str, Any]

Load a single image file.

**Parameters:**
- `image_path`: Path to image file

**Returns:**
- `Dict[str, Any]`: Data dictionary with image and metadata

##### load_directory(directory_path: str) -> List[Dict[str, Any]]

Load all images from a directory.

##### load_from_url(url: str) -> Dict[str, Any]

Load image from URL.

### StructuredDataLoader

Handles structured data (CSV, JSON, Excel) loading.

```python
from data_ingestion import StructuredDataLoader

loader = StructuredDataLoader()
data = loader.load_file("path/to/data.csv")
```

#### Methods

##### load_file(file_path: str) -> Dict[str, Any]

Load structured data file.

##### load_directory(directory_path: str) -> List[Dict[str, Any]]

Load all structured data files from directory.

## Data Processing Module

### TextCleaner

Text preprocessing and cleaning utilities.

```python
from data_processing import TextCleaner

cleaner = TextCleaner()
clean_text = cleaner.clean_text("Raw text content")
```

#### Methods

##### clean_text(text: str) -> str

Clean and normalize text content.

##### extract_food_entities_candidates(text: str) -> List[str]

Extract potential food safety entity candidates.

### LLMProcessor

Large Language Model processing interface.

```python
from data_processing import LLMProcessor

processor = LLMProcessor(model_type="gpt-3.5-turbo")
result = processor.process_text(text, task="summarize")
```

#### Methods

##### process_text(text: str, task: str = "analyze") -> Dict[str, Any]

Process text using LLM.

**Parameters:**
- `text`: Input text
- `task`: Processing task (`"analyze"`, `"summarize"`, `"classify"`)

### VLMProcessor

Vision Language Model processing interface.

```python
from data_processing import VLMProcessor

processor = VLMProcessor(model_type="blip-base")
result = processor.process_image(image, task="caption")
```

#### Methods

##### process_image(image: Any, task: str = "caption") -> Dict[str, Any]

Process image using VLM.

**Parameters:**
- `image`: Input image
- `task`: Processing task (`"caption"`, `"analyze_food_safety"`)

## Knowledge Extraction Module

### EntityExtractor

Entity extraction from text content.

```python
from knowledge_extraction import EntityExtractor

extractor = EntityExtractor(model_type="spacy")
entities = extractor.extract_entities(text)
```

#### Methods

##### extract_entities(text: str, confidence_threshold: float = 0.6) -> Dict[str, List[Dict]]

Extract entities from text.

**Parameters:**
- `text`: Input text
- `confidence_threshold`: Minimum confidence score

**Returns:**
- `Dict[str, List[Dict]]`: Entities grouped by type

### RelationExtractor

Relation extraction between entities.

```python
from knowledge_extraction import RelationExtractor

extractor = RelationExtractor(model_type="pattern")
relations = extractor.extract_relations(text, entities)
```

#### Methods

##### extract_relations(text: str, entities: Dict, confidence_threshold: float = 0.6) -> List[Dict]

Extract relations from text and entities.

**Parameters:**
- `text`: Input text
- `entities`: Previously extracted entities
- `confidence_threshold`: Minimum confidence score

**Returns:**
- `List[Dict]`: List of relation dictionaries

## Knowledge Graph Module

### KnowledgeGraphBuilder

Knowledge graph construction and management.

```python
from knowledge_graph import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(backend="networkx")
stats = builder.build_from_extractions(entities, relations)
```

#### Methods

##### build_from_extractions(entities: Dict, relations: List) -> Dict

Build knowledge graph from extracted entities and relations.

**Parameters:**
- `entities`: Entity dictionary from extraction
- `relations`: Relations list from extraction

**Returns:**
- `Dict`: Build statistics

##### export_graph(file_path: str, format: str = "json") -> None

Export knowledge graph to file.

**Parameters:**
- `file_path`: Output file path
- `format`: Export format (`"json"`, `"gexf"`, `"rdf"`)

##### get_statistics() -> Dict

Get knowledge graph statistics.

**Returns:**
- `Dict`: Graph statistics (nodes, edges, etc.)

## Configuration Module

### Settings

Project configuration and constants.

```python
from config import settings

entities = settings.FOOD_SAFETY_ENTITIES
relations = settings.FOOD_SAFETY_RELATIONS
llm_config = settings.get_model_config("llm")
```

#### Constants

- `FOOD_SAFETY_ENTITIES`: Predefined food safety entity types
- `FOOD_SAFETY_RELATIONS`: Predefined food safety relation types

#### Methods

##### get_model_config(model_type: str) -> Dict

Get model configuration.

##### get_kg_config() -> Dict

Get knowledge graph configuration.

## Utilities Module

### Logger

Logging utilities for the pipeline.

```python
from utils import setup_logger, get_logger

setup_logger()
logger = get_logger(__name__)
logger.info("Processing started")
```

#### Functions

##### setup_logger() -> None

Initialize logging system.

##### get_logger(name: str) -> Logger

Get logger instance.

### Helpers

Helper functions for configuration and data handling.

```python
from utils import load_config, save_config, clean_text

config = load_config("config.yaml")
save_config(data, "output.json")
```

#### Functions

##### load_config(file_path: str) -> Dict

Load configuration from file.

##### save_config(data: Dict, file_path: str) -> None

Save data to configuration file.

##### clean_text(text: str) -> str

Clean and normalize text.

## Error Handling

The API uses standard Python exceptions:

- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Missing input files
- `ImportError`: Missing optional dependencies
- `RuntimeError`: Processing failures

## Examples

### Basic Usage

```python
from pipeline import AIFSKGPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    input_sources=["data/"],
    input_types=["text"],
    use_llm=False,
    use_vlm=False
)

# Run pipeline
pipeline = AIFSKGPipeline(config)
results = pipeline.run()

print(f"Processed {results['statistics']['data_items_processed']} items")
print(f"Extracted {results['statistics']['total_entities']} entities")
```

### Advanced Configuration

```python
# Advanced configuration with all options
config = PipelineConfig(
    input_sources=["data/texts/", "data/images/", "https://example.com/data.txt"],
    input_types=["text", "image", "structured"],
    use_llm=True,
    use_vlm=True,
    llm_model="gpt-4",
    vlm_model="blip-large",
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.8,
    relation_confidence_threshold=0.7,
    kg_backend="neo4j",
    kg_output_format="gexf",
    save_intermediate_results=True,
    max_workers=8,
    batch_size=20
)
```

### Custom Processing

```python
from data_processing import TextCleaner
from knowledge_extraction import EntityExtractor, RelationExtractor
from knowledge_graph import KnowledgeGraphBuilder

# Manual processing steps
cleaner = TextCleaner()
entity_extractor = EntityExtractor()
relation_extractor = RelationExtractor()
kg_builder = KnowledgeGraphBuilder()

# Process text
clean_text = cleaner.clean_text(raw_text)
entities = entity_extractor.extract_entities(clean_text)
relations = relation_extractor.extract_relations(clean_text, entities)

# Build knowledge graph
kg_builder.build_from_extractions(entities, relations)
kg_builder.export_graph("output.json")
```