import pandas as pd
import os
import uuid
import json
import tempfile
from datetime import datetime
from fastapi import UploadFile, HTTPException
from typing import Dict, List, Any, Optional

from app.models.dataset import DatasetInfo, DatasetPreview, ColumnInfo

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
STORAGE_PATH = "./storage/datasets"

async def process_upload(file: UploadFile) -> DatasetInfo:
    """
    Process a dataset file upload, save it, and return basic information.
    """
    # Generate a unique ID for the dataset
    dataset_id = str(uuid.uuid4())
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Load the dataset with pandas
        if file_extension in [".csv"]:
            df = pd.read_csv(temp_file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(temp_file_path)
        elif file_extension == ".json":
            df = pd.read_json(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create the storage directory if it doesn't exist
        os.makedirs(STORAGE_PATH, exist_ok=True)
        
        # Save the file to storage
        storage_path = os.path.join(STORAGE_PATH, f"{dataset_id}{file_extension}")
        with open(storage_path, "wb") as f:
            with open(temp_file_path, "rb") as temp:
                f.write(temp.read())
        
        # Create dataset info
        dataset_info = DatasetInfo(
            id=dataset_id,
            filename=file.filename,
            file_type=file_extension.replace(".", ""),
            upload_timestamp=datetime.now(),
            num_rows=df.shape[0],
            num_columns=df.shape[1],
            column_names=df.columns.tolist()
        )
        
        # Save dataset metadata
        metadata_path = os.path.join(STORAGE_PATH, f"{dataset_id}_metadata.json")
        with open(metadata_path, "w") as f:
            f.write(dataset_info.json())
        
        return dataset_info
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def get_preview(dataset_id: str) -> DatasetPreview:
    """
    Get a preview of the dataset including sample rows and column statistics.
    """
    metadata_path = os.path.join(STORAGE_PATH, f"{dataset_id}_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load dataset metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Find the dataset file
    file_extension = f".{metadata['file_type']}"
    dataset_path = os.path.join(STORAGE_PATH, f"{dataset_id}{file_extension}")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Load the dataset
    try:
        if file_extension == ".csv":
            df = pd.read_csv(dataset_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(dataset_path)
        elif file_extension == ".json":
            df = pd.read_json(dataset_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate column information
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "data_type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "missing_values": df[col].isna().sum(),
                "sample_values": df[col].dropna().head(5).tolist()
            }
            
            # Add numerical statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min_value": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max_value": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "std_dev": float(df[col].std()) if not pd.isna(df[col].std()) else None
                })
            
            # Add categorical statistics if applicable
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts().head(5).to_dict()
                col_info["top_categories"] = value_counts
            
            columns.append(ColumnInfo(**col_info))
        
        # Create the preview object
        preview = DatasetPreview(
            id=metadata["id"],
            filename=metadata["filename"],
            file_type=metadata["file_type"],
            upload_timestamp=datetime.fromisoformat(metadata["upload_timestamp"]),
            num_rows=metadata["num_rows"],
            num_columns=metadata["num_columns"],
            sample_rows=df.head(10).to_dict(orient="records"),
            columns=columns
        )
        
        return preview
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing dataset: {str(e)}") 