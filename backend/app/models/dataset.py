from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ColumnInfo(BaseModel):
    """Information about a column in the dataset."""
    name: str
    data_type: str
    unique_values: int
    missing_values: int
    sample_values: List[Any]
    
    # For numerical columns
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    
    # For categorical columns
    top_categories: Optional[Dict[str, int]] = None

class DatasetInfo(BaseModel):
    """Basic information about an uploaded dataset."""
    id: str
    filename: str
    file_type: str
    upload_timestamp: datetime
    num_rows: int
    num_columns: int
    column_names: List[str]

class DatasetPreview(BaseModel):
    """Preview information of a dataset including sample rows and column statistics."""
    id: str
    filename: str
    file_type: str
    upload_timestamp: datetime
    num_rows: int
    num_columns: int
    sample_rows: List[Dict[str, Any]]
    columns: List[ColumnInfo] 