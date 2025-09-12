# AI-FS-KG-Gen Usage Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-FS-KG-Gen.git
cd AI-FS-KG-Gen

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Optional Dependencies

For full functionality, install additional dependencies:

```bash
# Install spaCy model for NER
python -m spacy download en_core_web_sm

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Variables

Set up required environment variables:

```bash
# For OpenAI LLM integration
export OPENAI_API_KEY="your-openai-api-key"

# For Neo4j integration (optional)
export KG_DATABASE_URL="bolt://localhost:7687"
export KG_USERNAME="neo4j"
export KG_PASSWORD="your-password"
```

## Quick Start

### Basic Pipeline Example

```python
from pipeline import AIFSKGPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    input_sources=["path/to/your/data"],
    input_types=["text", "image", "structured"],
    use_llm=True,
    use_vlm=True,
    entity_extraction=True,
    relation_extraction=True
)

# Run pipeline
pipeline = AIFSKGPipeline(config)
results = pipeline.run()

print(f"Extracted {results['statistics']['total_entities']} entities")
print(f"Extracted {results['statistics']['total_relations']} relations")
```

### Command Line Usage

```bash
# Basic usage
python -m pipeline.orchestrator --input data/ --output results/

# With configuration file
python -m pipeline.orchestrator --config config.yaml

# Disable certain features
python -m pipeline.orchestrator --input data/ --no-llm --no-vlm
```

## Data Input

### Supported Input Types

#### Text Data
- **File formats**: .txt, .md, .html, .pdf
- **Content**: Research papers, regulations, reports, web articles
- **Example**:
```python
from data_ingestion import TextLoader

loader = TextLoader()
data = loader.load_file("food_safety_report.pdf")
```

#### Image Data
- **File formats**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Content**: Food images, safety violations, laboratory results
- **Example**:
```python
from data_ingestion import ImageLoader

loader = ImageLoader()
data = loader.load_image("contaminated_food.jpg")
```

#### Structured Data
- **File formats**: .csv, .json, .xlsx, .xls, .jsonl
- **Content**: Database exports, inspection records, test results
- **Example**:
```python
from data_ingestion import StructuredDataLoader

loader = StructuredDataLoader()
data = loader.load_file("inspection_records.csv")
```

### Data Sources

- **Local files**: Single files or entire directories
- **Web URLs**: Direct links to documents or images
- **Databases**: Through structured data exports

## Configuration

### Pipeline Configuration

Create a `PipelineConfig` object:

```python
from pipeline import PipelineConfig

config = PipelineConfig(
    # Input configuration
    input_sources=["data/texts/", "data/images/"],
    input_types=["text", "image"],
    
    # Processing configuration
    use_llm=True,
    use_vlm=True,
    llm_model="gpt-3.5-turbo",
    vlm_model="blip-base",
    
    # Extraction configuration
    entity_extraction=True,
    relation_extraction=True,
    entity_confidence_threshold=0.7,
    relation_confidence_threshold=0.6,
    
    # Knowledge graph configuration
    kg_backend="networkx",  # or "neo4j", "rdf"
    kg_output_format="json",  # or "gexf", "rdf"
    
    # Output configuration
    output_dir="results/",
    save_intermediate_results=True,
    
    # Performance configuration
    max_workers=4,
    batch_size=10
)
```

### YAML Configuration

Alternatively, use a YAML configuration file:

```yaml
# config.yaml
input_sources:
  - "data/food_safety_papers/"
  - "data/images/"
input_types:
  - "text"
  - "image"

use_llm: true
use_vlm: true
llm_model: "gpt-3.5-turbo"
vlm_model: "blip-base"

entity_extraction: true
relation_extraction: true
entity_confidence_threshold: 0.7
relation_confidence_threshold: 0.6

kg_backend: "networkx"
kg_output_format: "json"
output_dir: "results/"
save_intermediate_results: true

max_workers: 4
batch_size: 10
```

## Individual Component Usage

### Text Processing

```python
from data_processing import TextCleaner, LLMProcessor

# Clean text
cleaner = TextCleaner()
cleaned_text = cleaner.clean_text("Raw text with noise...")

# Process with LLM
llm = LLMProcessor(model_type="gpt-3.5-turbo")
summary = llm.process_text(cleaned_text, task="summarize")
entities = llm.process_text(cleaned_text, task="extract_entities")
```

### Image Processing

```python
from data_processing import VLMProcessor
from PIL import Image

# Load and process image
vlm = VLMProcessor(model_type="blip-base")
image = Image.open("food_sample.jpg")

caption = vlm.process_image(image, task="caption")
safety_analysis = vlm.process_image(image, task="analyze_food_safety")
```

### Knowledge Extraction

```python
from knowledge_extraction import EntityExtractor, RelationExtractor

# Extract entities
entity_extractor = EntityExtractor(model_type="spacy")
entities = entity_extractor.extract_entities(text)

# Extract relations
relation_extractor = RelationExtractor(model_type="pattern")
relations = relation_extractor.extract_relations(text, entities)
```

### Knowledge Graph Operations

```python
from knowledge_graph import KnowledgeGraphBuilder

# Create knowledge graph
kg = KnowledgeGraphBuilder(backend="networkx")

# Add entities and relations
entity_id = kg.add_entity("E. coli", "pathogen")
kg.add_relation("Ground beef", "contaminated_with", "E. coli")

# Query the graph
entities = kg.query_entities(entity_type="pathogen")
relations = kg.query_relations(predicate="contaminated_with")

# Export the graph
kg.export_graph("food_safety_kg.json", format="json")
```

## Output Formats

### Knowledge Graph Exports

- **JSON**: Human-readable, good for analysis
- **GEXF**: Graph format, compatible with Gephi
- **RDF/Turtle**: Semantic web format, supports SPARQL

### Result Files

The pipeline generates several output files:

- `pipeline_results_TIMESTAMP.json`: Complete pipeline results
- `knowledge_graph_TIMESTAMP.json`: Knowledge graph export
- `entities_TIMESTAMP.json`: Extracted entities
- `relations_TIMESTAMP.json`: Extracted relations

### Example Output Structure

```json
{
  "metadata": {
    "pipeline_id": "2023-12-07T10:30:00",
    "config": {...}
  },
  "statistics": {
    "data_items_processed": 25,
    "total_entities": 150,
    "total_relations": 89,
    "entity_types": ["pathogen", "food_product", "chemical_compound"],
    "relation_types": ["contains", "causes", "prevents"]
  },
  "entities": {
    "pathogen": [
      {
        "text": "E. coli",
        "confidence": 0.95,
        "normalized_text": "e coli"
      }
    ]
  },
  "relations": [
    {
      "subject": "ground beef",
      "predicate": "contaminated_with",
      "object": "e coli",
      "confidence": 0.87
    }
  ]
}
```

## Performance Optimization

### Parallel Processing

```python
config = PipelineConfig(
    max_workers=8,  # Increase for more parallelism
    batch_size=20   # Larger batches for efficiency
)
```

### Memory Management

- Use batch processing for large datasets
- Enable intermediate result saving for checkpointing
- Choose appropriate backend (NetworkX for small, Neo4j for large)

### Model Selection

- Use lightweight models for faster processing
- Cache model outputs when possible
- Consider local models to avoid API rate limits

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Install required models: `python -m spacy download en_core_web_sm`
   - Check GPU memory for large models

2. **API Rate Limits**
   - Set appropriate delays in API calls
   - Use local models as alternatives

3. **Memory Issues**
   - Reduce batch size
   - Use streaming processing for large datasets

4. **Neo4j Connection Errors**
   - Verify database is running
   - Check connection credentials

### Debug Mode

Enable detailed logging:

```python
from utils.logger import setup_logger

setup_logger(log_level="DEBUG")
```

## Advanced Usage

### Custom Entity Types

```python
from knowledge_extraction import EntityExtractor

custom_entities = ["custom_entity_type", "another_type"]
extractor = EntityExtractor(custom_entities=custom_entities)
```

### Custom Relations

```python
from knowledge_extraction import RelationExtractor

custom_relations = ["custom_relation", "another_relation"]
extractor = RelationExtractor(custom_relations=custom_relations)
```

### Graph Analysis

```python
import networkx as nx

# Get the NetworkX graph
graph = kg_builder.graph

# Analyze graph properties
print(f"Graph density: {nx.density(graph)}")
print(f"Number of connected components: {nx.number_connected_components(graph.to_undirected())}")

# Find important nodes
centrality = nx.degree_centrality(graph)
important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
```

## Examples

See the `examples/` directory for:

- `basic_pipeline.py`: Simple pipeline example
- `advanced_features.py`: Advanced configuration examples
- `custom_components.py`: How to extend the system

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review example code
3. Open an issue on GitHub