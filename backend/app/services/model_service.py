import pandas as pd
import numpy as np
import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.models.notebook import NotebookInfo, ModelType, TaskType, ModelParameter
from app.models.training import TrainingRequest, FeatureConfig
from app.services import notebook_service

DATASETS_PATH = "./storage/datasets"
MODELS_PATH = "./storage/models"

async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks) -> NotebookInfo:
    """
    Train a machine learning model and generate a notebook.
    """
    # Check if dataset exists
    metadata_path = os.path.join(DATASETS_PATH, f"{request.dataset_id}_metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load dataset metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Find the dataset file
    file_extension = f".{metadata['file_type']}"
    dataset_path = os.path.join(DATASETS_PATH, f"{request.dataset_id}{file_extension}")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Generate notebook ID
    notebook_id = str(uuid.uuid4())
    
    # Create notebook info
    notebook_info = NotebookInfo(
        id=notebook_id,
        dataset_id=request.dataset_id,
        task_type=request.task_type,
        model_type=request.model_type,
        target_column=request.target_column,
        feature_columns=[f.name for f in request.features if f.use],
        parameters=[
            ModelParameter(name=k, value=v) 
            for k, v in (request.model_parameters or {}).items()
        ],
        creation_timestamp=datetime.now(),
        metrics={},
        download_url=f"/api/notebooks/{notebook_id}"
    )
    
    # Generate notebook in the background
    background_tasks.add_task(
        _generate_notebook_and_train,
        notebook_info=notebook_info,
        dataset_path=dataset_path,
        file_extension=file_extension,
        request=request
    )
    
    return notebook_info

async def _generate_notebook_and_train(
    notebook_info: NotebookInfo,
    dataset_path: str,
    file_extension: str,
    request: TrainingRequest
):
    """
    Generate a notebook for model training and execute it.
    """
    try:
        # Load the dataset
        if file_extension == ".csv":
            df = pd.read_csv(dataset_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(dataset_path)
        elif file_extension == ".json":
            df = pd.read_json(dataset_path)
        
        # Validate feature configurations
        for feature in request.features:
            if feature.use and feature.name not in df.columns:
                raise ValueError(f"Feature '{feature.name}' not found in dataset columns.")
        
        if request.target_column not in df.columns:
            raise ValueError(f"Target column '{request.target_column}' not found in dataset columns.")
        
        # Handle object type columns
        for feature in request.features:
            if feature.use and df[feature.name].dtype == 'object':
                if not feature.transform:
                    # Log warning about object type columns
                    print(f"Warning: Feature '{feature.name}' is an object type without transformation.")
                    # Default to onehot encoding for object types 
                    feature.transform = "onehot"
        
        try:
            # Generate the notebook content
            notebook = notebook_service.generate_notebook(
                df=df,
                notebook_info=notebook_info,
                request=request
            )
            
            # Save the notebook
            notebook_service.save_notebook(notebook_info.id, notebook)
            
            # Update metrics (in a real implementation, we would execute the notebook and extract metrics)
            # For now, we'll simulate some metrics
            metrics = _simulate_metrics(request.task_type)
            
            # Save metrics to notebook info
            _update_notebook_metrics(notebook_info.id, metrics)
        except Exception as e:
            # Log the specific error in notebook generation
            error_message = f"Error in notebook generation: {str(e)}"
            print(error_message)
            
            # Create an error notebook to explain the issue
            _create_error_notebook(notebook_info.id, error_message, request)
            
            # Update notebook info with error status
            _update_notebook_status(notebook_info.id, "failed", error_message)
            
            # Re-raise the exception with more context
            raise Exception(error_message) from e
    
    except Exception as e:
        # Handle all other exceptions
        error_message = f"Error in model training process: {str(e)}"
        print(error_message)
        # We could save error info to a log or database here
        _update_notebook_status(notebook_info.id, "failed", error_message)

def _simulate_metrics(task_type: TaskType) -> Dict[str, float]:
    """
    Simulate metrics for demonstration purposes.
    In a real system, these would come from executing the notebook.
    """
    if task_type == TaskType.REGRESSION:
        return {
            "r2_score": round(np.random.uniform(0.7, 0.95), 4),
            "mean_absolute_error": round(np.random.uniform(0.1, 0.5), 4),
            "mean_squared_error": round(np.random.uniform(0.1, 0.5), 4),
        }
    elif task_type == TaskType.CLASSIFICATION:
        return {
            "accuracy": round(np.random.uniform(0.8, 0.98), 4),
            "precision": round(np.random.uniform(0.8, 0.98), 4),
            "recall": round(np.random.uniform(0.8, 0.98), 4),
            "f1_score": round(np.random.uniform(0.8, 0.98), 4),
        }
    else:  # Clustering
        return {
            "silhouette_score": round(np.random.uniform(0.5, 0.9), 4),
            "inertia": round(np.random.uniform(10, 100), 4),
        } 

# Add helper functions for error handling
def _create_error_notebook(notebook_id: str, error_message: str, request: TrainingRequest):
    """Create a simple notebook explaining the error"""
    import nbformat as nbf
    from nbformat.v4 import new_markdown_cell, new_code_cell
    
    notebook = nbf.v4.new_notebook()
    notebook.cells = []
    
    # Add title
    notebook.cells.append(new_markdown_cell("# Model Training Error Report"))
    
    # Add error information
    notebook.cells.append(new_markdown_cell("## Error Details"))
    notebook.cells.append(new_markdown_cell(f"**Error Message**: {error_message}"))
    
    # Add model configuration
    notebook.cells.append(new_markdown_cell("## Model Configuration"))
    config_text = f"""
    - **Dataset ID**: {request.dataset_id}
    - **Task Type**: {request.task_type}
    - **Model Type**: {request.model_type}
    - **Target Column**: {request.target_column}
    - **Selected Features**: {', '.join([f.name for f in request.features if f.use])}
    """
    notebook.cells.append(new_markdown_cell(config_text))
    
    # Add troubleshooting tips
    notebook.cells.append(new_markdown_cell("## Troubleshooting Tips"))
    tips = """
    1. Check if all selected features are valid columns in your dataset
    2. For object-type features, ensure they are encoded properly (one-hot encoding recommended)
    3. Ensure your target column doesn't contain invalid values
    4. Try reducing the number of features if you're selecting many columns
    5. For numerical features with missing values, consider imputation strategies
    """
    notebook.cells.append(new_markdown_cell(tips))
    
    # Save the error notebook
    notebook_service.save_notebook(notebook_id, notebook)
    
def _update_notebook_metrics(notebook_id: str, metrics: Dict[str, float]):
    """Update notebook with metrics information"""
    try:
        # Load existing metadata
        metadata_path = os.path.join(notebook_service.NOTEBOOKS_PATH, f"{notebook_id}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                notebook_info_dict = json.load(f)
            
            # Update with metrics
            notebook_info_dict["metrics"] = metrics
            
            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(notebook_info_dict, f)
    except Exception as e:
        print(f"Error updating notebook metrics: {str(e)}")

def _update_notebook_status(notebook_id: str, status: str, error_message: str = None):
    """Update notebook status information"""
    try:
        # This would typically update a database record
        # For now, we'll just create a status file
        status_path = f"./storage/notebooks/{notebook_id}_status.json"
        status_data = {
            "status": status,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, 'w') as f:
            json.dump(status_data, f)
            
        # Also update the main metadata file if it exists
        metadata_path = os.path.join(notebook_service.NOTEBOOKS_PATH, f"{notebook_id}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    notebook_info_dict = json.load(f)
                
                # Add status information
                notebook_info_dict["status"] = status
                if error_message:
                    notebook_info_dict["error"] = error_message
                
                # Save updated metadata
                with open(metadata_path, "w") as f:
                    json.dump(notebook_info_dict, f)
            except Exception as inner_e:
                print(f"Error updating metadata file: {str(inner_e)}")
    except Exception as e:
        print(f"Error updating notebook status: {str(e)}") 