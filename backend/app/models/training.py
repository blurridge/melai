from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.models.notebook import TaskType, ModelType

class FeatureConfig(BaseModel):
    """Configuration for a feature in the model."""
    name: str
    use: bool = True
    transform: Optional[str] = None  # One of: None, "log", "scale", "onehot"

class TrainingRequest(BaseModel):
    """Request to train a machine learning model."""
    dataset_id: str
    task_type: TaskType
    model_type: ModelType
    target_column: str
    features: List[FeatureConfig]
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = 42
    model_parameters: Optional[Dict[str, Any]] = None 