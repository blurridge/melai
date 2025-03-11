import os
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from fastapi import UploadFile, HTTPException
import tempfile

async def read_data_file(file: UploadFile) -> Tuple[pd.DataFrame, str]:
    """
    Read a data file (CSV, Excel, JSON) and return a pandas DataFrame and the file extension.
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        if file_extension == ".csv":
            df = pd.read_csv(temp_file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(temp_file_path)
        elif file_extension == ".json":
            df = pd.read_json(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        return df, file_extension
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def data_types_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by their data types.
    """
    data_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "other": []
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types["numeric"].append(col)
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Check if it might be boolean
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2 and set(map(str, unique_values)).issubset({'True', 'False', '0', '1', 'yes', 'no', 'y', 'n', 't', 'f'}):
                data_types["boolean"].append(col)
            else:
                data_types["categorical"].append(col)
        elif pd.api.types.is_datetime64_dtype(df[col]):
            data_types["datetime"].append(col)
        else:
            data_types["other"].append(col)
    
    return data_types

def suggest_task_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Suggest a machine learning task type based on the target column.
    """
    if target_column not in df.columns:
        return "unknown"
    
    if pd.api.types.is_numeric_dtype(df[target_column]):
        return "regression"
    
    unique_count = df[target_column].nunique()
    
    if unique_count == 2:
        return "binary_classification"
    elif 2 < unique_count <= 20:
        return "multiclass_classification"
    else:
        return "regression" if pd.api.types.is_numeric_dtype(df[target_column]) else "unknown" 