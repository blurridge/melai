from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
import shutil
import tempfile
import uuid
import json
import sys
import platform
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.models import dataset as dataset_models
from app.models import notebook as notebook_models
from app.models.training import TrainingRequest
from app.services import data_service, model_service, notebook_service

# Create the FastAPI app
app = FastAPI(
    title="ML App Backend",
    description="API for a no-code ML model building application",
    version="1.0.0"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directories if they don't exist
os.makedirs("./storage/datasets", exist_ok=True)
os.makedirs("./storage/models", exist_ok=True)
os.makedirs("./storage/notebooks", exist_ok=True)


@app.get("/")
def read_root():
    """Basic root endpoint."""
    return {"status": "ok", "message": "ML App Backend is running"}


@app.get("/health")
def health_check():
    """
    Comprehensive health check endpoint that provides system information 
    and verifies storage directories.
    """
    # Check storage directories
    storage_status = {
        "datasets": os.path.exists("./storage/datasets"),
        "models": os.path.exists("./storage/models"),
        "notebooks": os.path.exists("./storage/notebooks")
    }
    
    # Count existing resources
    resource_counts = {
        "datasets": len(os.listdir("./storage/datasets")) if os.path.exists("./storage/datasets") else 0,
        "models": len(os.listdir("./storage/models")) if os.path.exists("./storage/models") else 0,
        "notebooks": len(os.listdir("./storage/notebooks")) if os.path.exists("./storage/notebooks") else 0
    }
    
    # Get system info
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Return comprehensive health status
    return {
        "status": "ok",
        "message": "ML App Backend is operational",
        "api_version": app.version,
        "storage_status": storage_status,
        "resource_counts": resource_counts,
        "system_info": system_info
    }


@app.post("/api/datasets/upload", response_model=dataset_models.DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset (CSV, Excel, JSON) and return basic information about it.
    """
    try:
        return await data_service.process_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/datasets/{dataset_id}", response_model=dataset_models.DatasetPreview)
async def get_dataset_preview(dataset_id: str):
    """
    Get a preview of a dataset with basic statistics.
    """
    try:
        return await data_service.get_preview(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/models/train", response_model=notebook_models.NotebookInfo)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a machine learning model based on the specified configuration
    and return the notebook containing the code.
    """
    try:
        return await model_service.train_model(request, background_tasks)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str):
    """
    Get a Jupyter notebook file by ID.
    """
    try:
        notebook_path = f"./storage/notebooks/{notebook_id}.ipynb"
        if not os.path.exists(notebook_path):
            raise HTTPException(status_code=404, detail="Notebook not found")
        return FileResponse(
            notebook_path, 
            media_type="application/json",
            filename=f"ml_model_{notebook_id}.ipynb"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) 