from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Types of ML models supported."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    KNN = "knn"

class TaskType(str, Enum):
    """Types of ML tasks supported."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"

class ModelParameter(BaseModel):
    """A parameter for a machine learning model."""
    name: str
    value: Any
    description: Optional[str] = None

class NotebookInfo(BaseModel):
    """Information about a generated Jupyter notebook."""
    id: str
    dataset_id: str
    task_type: TaskType
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    parameters: Optional[List[ModelParameter]] = None
    creation_timestamp: datetime
    metrics: Optional[Dict[str, float]] = None
    download_url: str 