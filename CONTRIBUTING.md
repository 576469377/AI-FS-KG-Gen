# Contributing to AI-FS-KG-Gen

We welcome contributions to the AI-FS-KG-Gen project! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of knowledge graphs and NLP

### Development Setup

1. **Fork the repository**
   ```bash
   git fork https://github.com/576469377/AI-FS-KG-Gen.git
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-FS-KG-Gen.git
   cd AI-FS-KG-Gen
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run tests**
   ```bash
   python test_basic.py
   python test_fixes.py
   ```

## 🛠️ Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small

### Project Structure

```
AI-FS-KG-Gen/
├── src/                    # Source code
│   ├── data_ingestion/     # Data loading modules
│   ├── data_processing/    # AI processing modules
│   ├── knowledge_extraction/  # Entity and relation extraction
│   ├── knowledge_graph/    # Knowledge graph construction
│   ├── pipeline/          # Pipeline orchestration
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── docs/                  # Documentation
├── examples/              # Example scripts
├── tests/                 # Test files
└── output/               # Generated outputs
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature**
   - Add your code to the appropriate module
   - Follow existing patterns and conventions
   - Add appropriate error handling and logging

3. **Add tests**
   - Create tests for your new functionality
   - Ensure all existing tests still pass

4. **Update documentation**
   - Update relevant docstrings
   - Add examples if applicable
   - Update README.md if needed

5. **Submit a pull request**
   - Provide a clear description of your changes
   - Reference any related issues

## 🧪 Testing

### Running Tests

```bash
# Basic functionality tests
python test_basic.py

# Bug fix verification tests
python test_fixes.py

# Run example pipeline
python examples/basic_pipeline.py
```

### Writing Tests

- Place test files in the root directory or `tests/` folder
- Use descriptive test names
- Test both success and failure cases
- Include edge cases and error conditions

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def extract_entities(self, text: str, confidence_threshold: float = 0.6) -> Dict[str, List[Dict]]:
    """
    Extract entities from text.
    
    Args:
        text: Input text to process
        confidence_threshold: Minimum confidence score for entities
        
    Returns:
        Dictionary mapping entity types to lists of entity dictionaries
        
    Raises:
        ValueError: If text is empty or confidence_threshold is invalid
    """
    pass
```

### Architecture Documentation

When adding new components:
- Update `docs/architecture.md`
- Add component descriptions
- Update the pipeline flowchart if needed
- Include configuration options

## 🔧 Specific Areas for Contribution

### High Priority

1. **Enhanced AI Model Integration**
   - Support for additional LLM providers
   - Vision-language model improvements
   - Custom model integration

2. **Knowledge Graph Backends**
   - Neo4j optimization
   - RDF/SPARQL query support
   - Graph visualization tools

3. **Data Processing**
   - Additional file format support
   - Streaming data processing
   - Parallel processing optimization

### Medium Priority

1. **User Interface**
   - Web-based configuration interface
   - Progress monitoring dashboard
   - Visualization tools

2. **Performance**
   - Caching mechanisms
   - Memory optimization
   - Batch processing improvements

3. **Quality Assurance**
   - Confidence scoring improvements
   - Entity resolution algorithms
   - Duplicate detection

### Documentation Improvements

- Tutorial notebooks
- API documentation
- Use case examples
- Troubleshooting guides

## 🐛 Bug Reports

When reporting bugs:

1. **Check existing issues** first
2. **Provide detailed information**:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

3. **Use the bug report template** (if available)

## 💡 Feature Requests

When requesting features:

1. **Describe the use case**
2. **Explain the expected behavior**
3. **Consider implementation challenges**
4. **Suggest possible approaches**

## 📖 Code Review Process

1. All contributions require code review
2. Maintainers will review pull requests
3. Address feedback promptly
4. Ensure tests pass before merging

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to AI-FS-KG-Gen! 🎉