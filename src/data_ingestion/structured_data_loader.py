"""
Structured data loading for AI-FS-KG-Gen pipeline
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union
from utils.logger import get_logger
from utils.helpers import validate_file_path, generate_hash

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = get_logger(__name__)

class StructuredDataLoader:
    """
    Loader for structured data formats like CSV, JSON, Excel, etc.
    """
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize structured data loader
        
        Args:
            supported_formats: List of supported file formats
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available. Excel file support will be limited.")
            
        self.supported_formats = supported_formats or ['.csv', '.json', '.xlsx', '.xls', '.jsonl', '.tsv']
        self.pandas_available = PANDAS_AVAILABLE
        logger.info(f"StructuredDataLoader initialized with formats: {self.supported_formats}")
    
    def load_csv(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional parameters for pandas.read_csv
        
        Returns:
            Dictionary containing data and metadata
        """
        if not self.pandas_available:
            logger.error("pandas not available. Cannot load CSV files efficiently.")
            raise ImportError("pandas is required for CSV processing. Install with: pip install pandas")
            
        file_path = Path(file_path)
        logger.info(f"Loading CSV file: {file_path}")
        
        try:
            # Default parameters
            csv_params = {
                'encoding': 'utf-8',
                'low_memory': False,
                'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'nan']
            }
            csv_params.update(kwargs)
            
            df = pd.read_csv(file_path, **csv_params)
            
            metadata = {
                'source': str(file_path),
                'format': 'csv',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'file_size': file_path.stat().st_size,
                'hash': generate_hash(str(df.values.tolist()))
            }
            
            return {
                'data': df,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Dictionary containing data and metadata
        """
        file_path = Path(file_path)
        logger.info(f"Loading JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {
                'source': str(file_path),
                'format': 'json',
                'data_type': type(data).__name__,
                'file_size': file_path.stat().st_size,
                'hash': generate_hash(json.dumps(data, sort_keys=True))
            }
            
            if isinstance(data, list):
                metadata['count'] = len(data)
            elif isinstance(data, dict):
                metadata['keys'] = list(data.keys())
            
            return {
                'data': data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
    
    def load_jsonl(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from JSONL (JSON Lines) file
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            Dictionary containing data and metadata
        """
        file_path = Path(file_path)
        logger.info(f"Loading JSONL file: {file_path}")
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            
            metadata = {
                'source': str(file_path),
                'format': 'jsonl',
                'count': len(data),
                'file_size': file_path.stat().st_size,
                'hash': generate_hash(json.dumps(data, sort_keys=True))
            }
            
            return {
                'data': data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {str(e)}")
            raise
    
    def load_excel(self, file_path: str, sheet_name: Optional[Union[str, int]] = None, **kwargs) -> Dict[str, Any]:
        """
        Load data from Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to load
            **kwargs: Additional parameters for pandas.read_excel
        
        Returns:
            Dictionary containing data and metadata
        """
        if not self.pandas_available:
            logger.error("pandas not available. Cannot load Excel files.")
            raise ImportError("pandas is required for Excel processing. Install with: pip install pandas openpyxl")
            
        file_path = Path(file_path)
        logger.info(f"Loading Excel file: {file_path}")
        
        try:
            excel_params = {
                'sheet_name': sheet_name,
                'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'nan']
            }
            excel_params.update(kwargs)
            
            if sheet_name is None:
                # Load all sheets
                data = pd.read_excel(file_path, sheet_name=None, **{k: v for k, v in excel_params.items() if k != 'sheet_name'})
                sheet_names = list(data.keys())
            else:
                # Load specific sheet
                data = pd.read_excel(file_path, **excel_params)
                sheet_names = [sheet_name] if isinstance(sheet_name, str) else [f"Sheet_{sheet_name}"]
            
            metadata = {
                'source': str(file_path),
                'format': 'excel',
                'sheet_names': sheet_names,
                'file_size': file_path.stat().st_size
            }
            
            if isinstance(data, dict):
                # Multiple sheets
                metadata['sheets_info'] = {}
                for sheet, df in data.items():
                    metadata['sheets_info'][sheet] = {
                        'shape': df.shape,
                        'columns': df.columns.tolist()
                    }
                metadata['hash'] = generate_hash(str([df.values.tolist() for df in data.values()]))
            else:
                # Single sheet
                metadata['shape'] = data.shape
                metadata['columns'] = data.columns.tolist()
                metadata['dtypes'] = data.dtypes.to_dict()
                metadata['hash'] = generate_hash(str(data.values.tolist()))
            
            return {
                'data': data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            raise
    
    def load_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load structured data file based on its extension
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parameters for specific loaders
        
        Returns:
            Dictionary containing data and metadata
        """
        file_path = Path(file_path)
        
        if not validate_file_path(file_path, self.supported_formats):
            raise ValueError(f"Invalid file path or unsupported format: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return self.load_csv(file_path, **kwargs)
        elif extension == '.json':
            return self.load_json(file_path)
        elif extension == '.jsonl':
            return self.load_jsonl(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self.load_excel(file_path, **kwargs)
        elif extension == '.tsv':
            return self.load_csv(file_path, sep='\t', **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Load all structured data files from a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
        
        Yields:
            Dictionary containing data and metadata for each file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        logger.info(f"Loading structured data files from directory: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    yield self.load_file(file_path)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
                    continue
    
    def extract_food_safety_records(self, data: Union[Any, List[Dict], Dict], 
                                   food_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract food safety related records from structured data
        
        Args:
            data: Input data (DataFrame, list of dicts, or dict)
            food_columns: Columns that might contain food-related information
        
        Returns:
            List of food safety records
        """
        logger.info("Extracting food safety records from structured data")
        
        if food_columns is None:
            food_columns = [
                'product', 'food', 'ingredient', 'item', 'name', 'title',
                'description', 'category', 'type', 'safety', 'risk',
                'contamination', 'pathogen', 'allergen', 'additive'
            ]
        
        records = []
        
        if self.pandas_available and pd is not None and hasattr(pd, 'DataFrame') and isinstance(data, pd.DataFrame):
            # Find relevant columns
            relevant_cols = []
            for col in data.columns:
                if any(food_term in col.lower() for food_term in food_columns):
                    relevant_cols.append(col)
            
            if not relevant_cols:
                relevant_cols = data.columns.tolist()[:10]  # Take first 10 columns as fallback
            
            for _, row in data.iterrows():
                record = {}
                for col in relevant_cols:
                    if row[col] is not None and str(row[col]).strip() != '' and str(row[col]).lower() not in ['nan', 'na', 'null']:
                        record[col] = str(row[col])
                
                if record:
                    records.append(record)
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Filter for food-related keys
                    record = {}
                    for key, value in item.items():
                        if any(food_term in key.lower() for food_term in food_columns):
                            record[key] = str(value) if value is not None else ""
                    
                    if record:
                        records.append(record)
        
        elif isinstance(data, dict):
            # Extract food-related key-value pairs
            record = {}
            for key, value in data.items():
                if any(food_term in key.lower() for food_term in food_columns):
                    record[key] = str(value) if value is not None else ""
            
            if record:
                records.append(record)
        
        logger.info(f"Extracted {len(records)} food safety records")
        return records

def load_structured_batch(file_paths: List[str], loader: Optional[StructuredDataLoader] = None) -> List[Dict[str, Any]]:
    """
    Load multiple structured data files in batch
    
    Args:
        file_paths: List of file paths
        loader: Optional StructuredDataLoader instance
    
    Returns:
        List of loaded data documents
    """
    if loader is None:
        loader = StructuredDataLoader()
    
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