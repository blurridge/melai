import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Dataset } from '../App';
import InfoIcon from '../components/InfoIcon';

interface ModelBuilderPageProps {
  currentDataset: Dataset | null;
}

interface ColumnInfo {
  name: string;
  data_type: string;
  unique_values: number;
  missing_values: number;
  sample_values?: any[];
}

interface FeatureConfig {
  name: string;
  use: boolean;
  transform: string | null;
}

interface ModelConfig {
  dataset_id: string;
  task_type: string;
  model_type: string;
  target_column: string;
  features: FeatureConfig[];
  test_size: number;
  random_state: number;
  model_parameters: Record<string, any>;
}

interface NotebookInfo {
  id: string;
  dataset_id: string;
  task_type: string;
  model_type: string;
  target_column: string;
  feature_columns: string[];
  parameters: any[] | null;
  creation_timestamp: string;
  metrics: Record<string, number> | null;
  download_url: string;
}

interface DataPreviewColumn {
  name: string;
  data_type: string;
  unique_values: number;
  missing_values: number;
  sample_values: any[];
}

interface ModelRecommendation {
  modelType: string;
  score: number;
  reasons: string[];
}

const API_URL = 'http://localhost:8000';

const ModelBuilderPage: React.FC<ModelBuilderPageProps> = ({ currentDataset }) => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    dataset_id: datasetId || '',
    task_type: 'regression',
    model_type: 'linear_regression',
    target_column: '',
    features: [],
    test_size: 0.2,
    random_state: 42,
    model_parameters: {},
  });
  
  const [isBuildingModel, setIsBuildingModel] = useState(false);
  const [notebookInfo, setNotebookInfo] = useState<NotebookInfo | null>(null);
  
  // New state for model recommendations and dataset info
  const [datasetSize, setDatasetSize] = useState<number>(0);
  const [modelRecommendations, setModelRecommendations] = useState<ModelRecommendation[]>([]);
  const [showRecommendations, setShowRecommendations] = useState<boolean>(false);
  
  // Function to update recommendations
  const updateRecommendations = () => {
    if (!modelConfig.target_column || columns.length === 0 || datasetSize === 0) {
      return;
    }
    
    const recommendations = generateModelRecommendations(
      modelConfig.task_type,
      columns,
      modelConfig.target_column,
      datasetSize
    );
    
    setModelRecommendations(recommendations);
  };
  
  // Update recommendations when relevant parameters change
  useEffect(() => {
    updateRecommendations();
  }, [modelConfig.task_type, modelConfig.target_column, columns, datasetSize]);
  
  useEffect(() => {
    const fetchDatasetColumns = async () => {
      if (!datasetId) {
        setError('Dataset ID is missing');
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        const response = await axios.get(`${API_URL}/api/datasets/${datasetId}`);
        const dataPreview = response.data;
        
        // Set dataset size
        setDatasetSize(dataPreview.num_rows || 0);
        
        // Extract column information
        const columnInfo = dataPreview.columns.map((col: DataPreviewColumn) => ({
          name: col.name,
          data_type: col.data_type,
          unique_values: col.unique_values,
          missing_values: col.missing_values,
          sample_values: col.sample_values,
        }));
        
        setColumns(columnInfo);
        
        // Initialize feature configs
        const featureConfigs = columnInfo.map((col: ColumnInfo) => ({
          name: col.name,
          use: true,
          transform: null,
        }));
        
        // Set target column to first column as default
        const defaultTargetColumn = columnInfo.length > 0 ? columnInfo[0].name : '';
        
        // Update model config
        setModelConfig(prev => ({
          ...prev,
          target_column: defaultTargetColumn,
          features: featureConfigs,
        }));
        
      } catch (error) {
        console.error('Error fetching dataset columns:', error);
        if (axios.isAxiosError(error) && error.response) {
          setError(error.response.data.detail || 'Error fetching dataset columns');
        } else {
          setError('Network error. Please check your connection.');
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchDatasetColumns();
  }, [datasetId]);
  
  const handleTaskTypeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const taskType = event.target.value;
    let modelType = modelConfig.model_type;
    
    // Update model type based on task type
    if (taskType === 'regression' && modelType === 'logistic_regression') {
      modelType = 'linear_regression';
    } else if (taskType === 'classification' && modelType === 'linear_regression') {
      modelType = 'logistic_regression';
    }
    
    setModelConfig(prev => ({
      ...prev,
      task_type: taskType,
      model_type: modelType,
    }));
  };
  
  const handleTargetColumnChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const targetColumn = event.target.value;
    
    // Update features to exclude target column
    const updatedFeatures = modelConfig.features.map(feature => ({
      ...feature,
      use: feature.name !== targetColumn,
    }));
    
    setModelConfig(prev => ({
      ...prev,
      target_column: targetColumn,
      features: updatedFeatures,
    }));
  };
  
  const handleFeatureToggle = (featureName: string) => {
    try {
      // Get the column info for this feature
      const column = columns.find(col => col.name === featureName);
      
      // If feature is already enabled and it's an object type, prevent disabling to avoid issues
      const feature = modelConfig.features.find(f => f.name === featureName);
      const isCurrentlyEnabled = feature?.use || false;
      
      // Safely check if it's an object type
      const isObject = column?.data_type?.includes('object') || false;
      const isSupported = isObject ? isObjectTypeSupported(column!) : true;
      
      // Show warning only when enabling an unsupported object type
      if (isObject && !isSupported && !isCurrentlyEnabled) {
        // Prevent enabling by showing warning and returning early
        alert(
          "Warning: This feature contains complex object data that may cause issues with the model. " +
          "We recommend excluding this feature from your model to prevent crashes."
        );
        return; // Prevent toggle by returning early
      }
      
      // Toggle the feature safely
      const updatedFeatures = modelConfig.features.map(feature => {
        if (feature.name === featureName) {
          return { 
            ...feature, 
            use: !feature.use,
            // Automatically set one-hot encoding for object types when enabling
            transform: (!feature.use && isObject && isSupported) ? 'onehot' : feature.transform 
          };
        }
        return feature;
      });
      
      setModelConfig(prev => ({
        ...prev,
        features: updatedFeatures,
      }));
    } catch (error) {
      console.error("Error toggling feature:", error);
      // Show error to user instead of crashing
      alert("An error occurred while toggling this feature. Please try again.");
    }
  };
  
  const handleFeatureTransformChange = (featureName: string, transform: string | null) => {
    const updatedFeatures = modelConfig.features.map(feature => {
      if (feature.name === featureName) {
        return { ...feature, transform };
      }
      return feature;
    });
    
    setModelConfig(prev => ({
      ...prev,
      features: updatedFeatures,
    }));
  };
  
  const handleParameterChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = event.target;
    let parsedValue: any = value;
    
    if (type === 'number') {
      parsedValue = parseFloat(value);
    } else if (type === 'checkbox') {
      parsedValue = (event.target as HTMLInputElement).checked;
    }
    
    if (name === 'test_size' || name === 'random_state') {
      setModelConfig(prev => ({
        ...prev,
        [name]: parsedValue,
      }));
    } else {
      setModelConfig(prev => ({
        ...prev,
        model_parameters: {
          ...prev.model_parameters,
          [name]: parsedValue,
        },
      }));
    }
  };
  
  const handleBuildModel = async () => {
    try {
      setIsBuildingModel(true);
      setError(null);
      
      // Filter out any potentially problematic features
      const safeFeatures = modelConfig.features.map(feature => {
        const column = columns.find(col => col.name === feature.name);
        const isObject = column?.data_type?.includes('object') || false;
        
        // If it's an object type feature that's enabled but not properly transformed
        if (feature.use && isObject && !feature.transform) {
          return {
            ...feature,
            transform: 'onehot' // Default to one-hot encoding for object types
          };
        }
        
        return feature;
      });
      
      // Create training request
      const trainingRequest = {
        dataset_id: modelConfig.dataset_id,
        task_type: modelConfig.task_type,
        model_type: modelConfig.model_type,
        target_column: modelConfig.target_column,
        features: safeFeatures,
        test_size: modelConfig.test_size,
        random_state: modelConfig.random_state,
        model_parameters: modelConfig.model_parameters,
      };
      
      // Safety check - ensure we don't have complex objects that can't be serialized
      try {
        JSON.stringify(trainingRequest);
      } catch (error) {
        throw new Error(
          "Unable to process the model configuration. Some features may contain complex data types that cannot be processed. " +
          "Please disable features with object type data and try again."
        );
      }
      
      // Send request to API
      const response = await axios.post(`${API_URL}/api/models/train`, trainingRequest);
      
      // Set notebook info
      setNotebookInfo(response.data);
      
    } catch (error) {
      console.error('Error building model:', error);
      if (axios.isAxiosError(error) && error.response) {
        setError(error.response.data.detail || 'Error building the model');
      } else if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('Network error. Please check your connection.');
      }
    } finally {
      setIsBuildingModel(false);
    }
  };
  
  const handleViewNotebook = () => {
    if (notebookInfo) {
      navigate(`/notebook/${notebookInfo.id}`);
    }
  };
  
  const renderModelParameters = () => {
    const { model_type } = modelConfig;
    
    // Default parameters based on model type
    switch (model_type) {
      case 'linear_regression':
      case 'logistic_regression':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Regularization (C)
              </label>
              <input
                type="number"
                name="C"
                min="0.001"
                step="0.1"
                defaultValue="1.0"
                onChange={handleParameterChange}
                className="input"
              />
              <p className="text-xs text-gray-500 mt-1">
                Inverse of regularization strength (smaller values = stronger regularization)
              </p>
            </div>
          </div>
        );
        
      case 'decision_tree':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Depth
              </label>
              <input
                type="number"
                name="max_depth"
                min="1"
                step="1"
                defaultValue="5"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Samples Split
              </label>
              <input
                type="number"
                name="min_samples_split"
                min="2"
                step="1"
                defaultValue="2"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
          </div>
        );
        
      case 'random_forest':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Trees
              </label>
              <input
                type="number"
                name="n_estimators"
                min="10"
                step="10"
                defaultValue="100"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Depth
              </label>
              <input
                type="number"
                name="max_depth"
                min="1"
                step="1"
                defaultValue="5"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
          </div>
        );
        
      case 'gradient_boosting':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Estimators
              </label>
              <input
                type="number"
                name="n_estimators"
                min="10"
                step="10"
                defaultValue="100"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                name="learning_rate"
                min="0.001"
                step="0.01"
                defaultValue="0.1"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
          </div>
        );
        
      case 'svm':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Kernel
              </label>
              <select
                name="kernel"
                onChange={(e) => 
                  setModelConfig(prev => ({
                    ...prev,
                    model_parameters: { ...prev.model_parameters, kernel: e.target.value }
                  }))
                }
                className="input"
                defaultValue="rbf"
              >
                <option value="linear">Linear</option>
                <option value="poly">Polynomial</option>
                <option value="rbf">RBF (Radial Basis Function)</option>
                <option value="sigmoid">Sigmoid</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                C (Regularization)
              </label>
              <input
                type="number"
                name="C"
                min="0.1"
                step="0.1"
                defaultValue="1.0"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
          </div>
        );
        
      case 'knn':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Neighbors
              </label>
              <input
                type="number"
                name="n_neighbors"
                min="1"
                step="1"
                defaultValue="5"
                onChange={handleParameterChange}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Weights
              </label>
              <select
                name="weights"
                onChange={(e) => 
                  setModelConfig(prev => ({
                    ...prev,
                    model_parameters: { ...prev.model_parameters, weights: e.target.value }
                  }))
                }
                className="input"
                defaultValue="uniform"
              >
                <option value="uniform">Uniform</option>
                <option value="distance">Distance</option>
              </select>
            </div>
          </div>
        );
        
      default:
        return null;
    }
  };
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  
  if (error && !notebookInfo) {
    return (
      <div className="bg-red-100 p-4 rounded-md text-red-700">
        <h2 className="text-lg font-semibold mb-2">Error</h2>
        <p>{error}</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-800">Model Builder</h1>
        <div>
          {currentDataset && (
            <span className="text-gray-500 mr-4">
              Dataset: <span className="font-medium">{currentDataset.filename}</span>
            </span>
          )}
        </div>
      </div>
      
      {notebookInfo ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h2 className="text-2xl font-bold text-green-800 mb-4">
            Model Built Successfully!
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <h3 className="text-lg font-medium mb-2 text-gray-700">Model Details</h3>
              <div className="space-y-2">
                <InfoRow label="Model Type" value={notebookInfo.model_type} />
                <InfoRow label="Task Type" value={notebookInfo.task_type} />
                <InfoRow label="Target Column" value={notebookInfo.target_column} />
                <InfoRow label="Features" value={notebookInfo.feature_columns.join(', ')} />
                <InfoRow 
                  label="Created" 
                  value={new Date(notebookInfo.creation_timestamp).toLocaleString()} 
                />
              </div>
            </div>
            
            {notebookInfo.metrics && Object.keys(notebookInfo.metrics).length > 0 && (
              <div>
                <h3 className="text-lg font-medium mb-2 text-gray-700">Performance Metrics</h3>
                <div className="space-y-2">
                  {Object.entries(notebookInfo.metrics).map(([key, value]) => (
                    <InfoRow 
                      key={key} 
                      label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} 
                      value={value.toString()} 
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="flex justify-end space-x-4">
            <button 
              onClick={handleViewNotebook}
              className="btn btn-primary"
            >
              View Notebook
            </button>
            <a 
              href={`${API_URL}${notebookInfo.download_url}`}
              download
              className="btn btn-secondary"
              target="_blank"
              rel="noopener noreferrer"
            >
              Download Notebook
            </a>
          </div>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">Model Configuration</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                    Task Type
                    <InfoIcon 
                      content="Select the type of machine learning problem you want to solve. Regression predicts continuous values, classification predicts categories, and clustering groups similar data points." 
                      position="right"
                    />
                  </label>
                  <select
                    value={modelConfig.task_type}
                    onChange={handleTaskTypeChange}
                    className="input"
                  >
                    <option value="regression">Regression</option>
                    <option value="classification">Classification</option>
                    <option value="clustering">Clustering</option>
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    {modelConfig.task_type === 'regression' ? 
                      'Regression: Predicts continuous values like price, temperature, or age.' :
                    modelConfig.task_type === 'classification' ? 
                      'Classification: Predicts categories like yes/no, spam/not spam, or types of flowers.' :
                      'Clustering: Groups similar data points together without predefined categories.'}
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                    Model Type
                    <InfoIcon 
                      content="Different algorithms have different strengths. Simple models like linear regression are more interpretable, while complex models like random forests may be more accurate but harder to understand." 
                      position="right"
                    />
                  </label>
                  <div className="flex items-center space-x-2">
                    <select
                      value={modelConfig.model_type}
                      onChange={(e) => setModelConfig(prev => ({ ...prev, model_type: e.target.value }))}
                      className="input flex-grow"
                    >
                      {modelConfig.task_type === 'regression' && (
                        <>
                          <option value="linear_regression">Linear Regression</option>
                          <option value="decision_tree">Decision Tree</option>
                          <option value="random_forest">Random Forest</option>
                          <option value="gradient_boosting">Gradient Boosting</option>
                          <option value="svm">Support Vector Machine (SVM)</option>
                          <option value="knn">K-Nearest Neighbors (KNN)</option>
                        </>
                      )}
                      
                      {modelConfig.task_type === 'classification' && (
                        <>
                          <option value="logistic_regression">Logistic Regression</option>
                          <option value="decision_tree">Decision Tree</option>
                          <option value="random_forest">Random Forest</option>
                          <option value="gradient_boosting">Gradient Boosting</option>
                          <option value="svm">Support Vector Machine (SVM)</option>
                          <option value="knn">K-Nearest Neighbors (KNN)</option>
                        </>
                      )}
                      
                      {modelConfig.task_type === 'clustering' && (
                        <>
                          <option value="kmeans">K-Means</option>
                          <option value="hierarchical">Hierarchical Clustering</option>
                          <option value="dbscan">DBSCAN</option>
                        </>
                      )}
                    </select>
                    
                    {modelConfig.target_column && (
                      <button 
                        className="btn btn-sm btn-outline-primary whitespace-nowrap"
                        onClick={() => setShowRecommendations(!showRecommendations)}
                      >
                        {showRecommendations ? 'Hide' : 'Get'} Recommendations
                      </button>
                    )}
                  </div>
                  
                  {/* Model Recommendations Section */}
                  {showRecommendations && modelRecommendations.length > 0 && (
                    <div className="mt-4 p-4 bg-blue-50 rounded-md border border-blue-200">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-medium text-blue-800">Model Recommendations</h3>
                        <button 
                          onClick={() => setShowRecommendations(false)} 
                          className="text-blue-700 hover:text-blue-900"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </div>
                      
                      <div className="mb-2 text-sm text-blue-700">
                        Based on your dataset characteristics and selected task, here are our model recommendations:
                      </div>
                      
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {modelRecommendations.map(rec => {
                          const displayName = rec.modelType
                            .replace(/_/g, ' ')
                            .replace(/\b\w/g, l => l.toUpperCase());
                          
                          // Calculate a color based on the score
                          const getScoreColor = (score: number) => {
                            if (score >= 80) return 'bg-green-100 text-green-800';
                            if (score >= 60) return 'bg-blue-100 text-blue-800';
                            if (score >= 40) return 'bg-yellow-100 text-yellow-800';
                            return 'bg-red-100 text-red-800';
                          };
                          
                          return (
                            <div 
                              key={rec.modelType}
                              className={`p-3 rounded-md border ${modelConfig.model_type === rec.modelType ? 'border-blue-400 bg-blue-50' : 'border-gray-200 bg-white'}`}
                            >
                              <div className="flex justify-between items-center mb-2">
                                <div className="font-medium">{displayName}</div>
                                <div className={`text-xs px-2 py-1 rounded-full ${getScoreColor(rec.score)}`}>
                                  Score: {rec.score}/100
                                </div>
                              </div>
                              
                              <div className="space-y-1">
                                {rec.reasons.map((reason, idx) => (
                                  <div key={idx} className="text-sm flex items-start">
                                    <span className="text-green-500 mr-1">
                                      {reason.includes('Warning') ? '⚠️' : '✓'}
                                    </span>
                                    <span className={reason.includes('Warning') ? 'text-amber-700' : 'text-gray-700'}>
                                      {reason}
                                    </span>
                                  </div>
                                ))}
                              </div>
                              
                              {modelConfig.model_type !== rec.modelType && (
                                <button
                                  className="mt-2 text-sm text-blue-600 hover:text-blue-800 font-medium"
                                  onClick={() => setModelConfig(prev => ({ ...prev, model_type: rec.modelType }))}
                                >
                                  Select this model
                                </button>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      
                      <div className="mt-3 text-xs text-gray-500">
                        Note: These recommendations are general guidelines based on common machine learning practices. 
                        The best model for your specific data may vary.
                      </div>
                    </div>
                  )}
                  
                  <p className="mt-1 text-xs text-gray-500">
                    {modelConfig.model_type === 'linear_regression' ? 
                      'Linear Regression: Simple model that works well for linear relationships between features and targets.' :
                    modelConfig.model_type === 'logistic_regression' ? 
                      'Logistic Regression: Good for binary or multi-class classification with clear decision boundaries.' :
                    modelConfig.model_type === 'decision_tree' ? 
                      'Decision Tree: Intuitive model that makes decisions based on feature thresholds. Highly interpretable.' :
                    modelConfig.model_type === 'random_forest' ? 
                      'Random Forest: Ensemble of decision trees that often provides higher accuracy and prevents overfitting.' :
                    modelConfig.model_type === 'gradient_boosting' ? 
                      'Gradient Boosting: Advanced ensemble technique that often achieves state-of-the-art performance.' :
                    modelConfig.model_type === 'svm' ? 
                      'Support Vector Machine: Works well for clearly separated classes and can handle non-linear boundaries.' :
                    modelConfig.model_type === 'knn' ? 
                      'K-Nearest Neighbors: Simple algorithm that classifies based on the k-closest training examples.' :
                    modelConfig.model_type === 'kmeans' ? 
                      'K-Means: Groups data into k clusters based on feature similarity.' :
                    modelConfig.model_type === 'hierarchical' ? 
                      'Hierarchical Clustering: Creates a tree of clusters without requiring a pre-specified number.' :
                    modelConfig.model_type === 'dbscan' ? 
                      'DBSCAN: Density-based clustering that works well for irregularly shaped clusters and outlier detection.' :
                      'Select a model type to learn more about it.'
                    }
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                    Target Column (What to Predict)
                    <InfoIcon 
                      content="This is the variable you want to predict or explain. For regression, this should be a numerical value. For classification, this should be a categorical variable with distinct classes. For clustering, this field is often ignored." 
                      position="right"
                    />
                  </label>
                  <div className="relative">
                    <select
                      value={modelConfig.target_column}
                      onChange={handleTargetColumnChange}
                      className="input"
                    >
                      <option value="">Select a target column</option>
                      {columns.map(column => (
                        <option key={column.name} value={column.name}>
                          {column.name} ({column.data_type})
                        </option>
                      ))}
                    </select>
                    <div className="mt-1 text-sm text-indigo-600 font-medium">
                      ℹ️ This is the column your model will learn to predict
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center">
                      Test Size
                      <InfoIcon 
                        content="The portion of your data that will be set aside to evaluate your model. A common split is 80% training, 20% testing. Smaller datasets might need a larger proportion for training." 
                        position="right"
                      />
                    </label>
                    <input
                      type="number"
                      name="test_size"
                      min="0.1"
                      max="0.5"
                      step="0.05"
                      value={modelConfig.test_size}
                      onChange={handleParameterChange}
                      className="input"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Proportion of data to use for testing (0.1 - 0.5)
                    </p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Random State
                    </label>
                    <input
                      type="number"
                      name="random_state"
                      min="0"
                      value={modelConfig.random_state}
                      onChange={handleParameterChange}
                      className="input"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Seed for reproducible results
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">Model Parameters</h2>
              {renderModelParameters()}
            </div>
          </div>
          
          <div className="space-y-6">
            <div className="card">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                  Features
                  <InfoIcon 
                    content="Features are the variables used to make predictions. Select which columns to include in your model. You can also apply transformations to improve model performance." 
                    position="right"
                  />
                </h2>
                <div className="text-sm text-gray-500">
                  {modelConfig.features.filter(f => f.use).length} selected
                </div>
              </div>
              
              <div className="bg-blue-50 p-3 rounded-md mb-4 text-sm text-blue-800 flex items-start">
                <svg className="h-5 w-5 text-blue-500 mr-1 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div>
                  <span className="font-medium">Tip:</span> Select features that are likely to influence what you're trying to predict. Exclude features that would cause data leakage (information that wouldn't be available at prediction time).
                </div>
              </div>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {modelConfig.features.map(feature => (
                  <div 
                    key={feature.name}
                    className={`p-3 rounded-md border 
                      ${feature.name === modelConfig.target_column 
                        ? 'bg-indigo-50 border-indigo-300' 
                        : feature.use 
                          ? 'bg-blue-50 border-blue-200' 
                          : 'bg-white border-gray-200'}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          checked={feature.use}
                          onChange={() => handleFeatureToggle(feature.name)}
                          disabled={feature.name === modelConfig.target_column}
                          className="mr-2 h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                        />
                        <span className={`font-medium ${feature.name === modelConfig.target_column ? 'text-indigo-700' : ''}`}>
                          {feature.name}
                        </span>
                      </div>
                      <span className="text-xs px-2.5 py-0.5 rounded-full text-xs font-medium 
                        ${feature.name === modelConfig.target_column 
                          ? 'bg-indigo-100 text-indigo-800' 
                          : 'bg-gray-200 text-gray-700'}">
                        {feature.name === modelConfig.target_column ? 'Target' : ''}
                      </span>
                    </div>
                    
                    {feature.use && feature.name !== modelConfig.target_column && (
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                          Transform
                          <InfoIcon 
                            content="Apply transformations to improve model performance. Log transform can help with skewed data. Scaling normalizes numerical values. One-hot encoding converts categories to binary features." 
                            position="right"
                            iconClass="ml-1 w-3 h-3"
                          />
                        </label>
                        {/* Check if feature is object type or has complex data and disable transformations */}
                        {columns.find(col => col.name === feature.name)?.data_type.includes('object') ? (
                          <div className="text-xs text-amber-600 mt-1">
                            <svg className="inline-block w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            Complex object type - no transformations available
                          </div>
                        ) : (
                          <select
                            value={feature.transform || ''}
                            onChange={(e) => handleFeatureTransformChange(
                              feature.name, 
                              e.target.value === '' ? null : e.target.value
                            )}
                            className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
                            disabled={feature.name === modelConfig.target_column}
                          >
                            <option value="">None</option>
                            <option value="log">Log Transform</option>
                            <option value="scale">Standardize (Z-score)</option>
                            <option value="onehot">One-Hot Encoding</option>
                          </select>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            <div className="text-right">
              <button
                onClick={handleBuildModel}
                disabled={isBuildingModel}
                className="btn btn-primary py-3 px-6"
              >
                {isBuildingModel ? (
                  <>
                    <span className="animate-spin inline-block h-4 w-4 border-t-2 border-b-2 border-white rounded-full mr-2"></span>
                    Building Model...
                  </>
                ) : 'Build Model and Generate Notebook'}
              </button>
              <p className="mt-2 text-xs text-gray-500">This will train your model and generate a Jupyter notebook with all the code</p>
            </div>
          </div>
        </div>
      )}
      
      {error && notebookInfo && (
        <div className="bg-red-100 p-4 rounded-md text-red-700">
          <h2 className="text-lg font-semibold mb-2">Error</h2>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

// Helper component
const InfoRow = ({ label, value }: { label: string; value: string }) => {
  return (
    <div className="grid grid-cols-2">
      <div className="font-medium text-gray-700">{label}:</div>
      <div className="text-gray-900">{value}</div>
    </div>
  );
};

// Update the isObjectTypeSupported function to be more robust
const isObjectTypeSupported = (column: ColumnInfo): boolean => {
  if (!column) return false;
  
  try {
    // Check if it's a categorical object that can be handled
    if (column.unique_values < 100) {
      return true;
    }
    
    // Check sample values to see if they are simple strings or complex objects
    if (column.sample_values && column.sample_values.length > 0) {
      const sample = column.sample_values[0];
      // Simple string or number values are supported
      if (typeof sample === 'string' || typeof sample === 'number') {
        return true;
      }
      
      // If it's an object, check if it's a simple key-value object
      if (typeof sample === 'object' && sample !== null) {
        // Try to stringify to see if it can be serialized
        JSON.stringify(sample);
        return false; // Even if serializable, complex objects are risky
      }
    }
    
    return false;
  } catch (e) {
    console.error("Error checking object type support:", e);
    return false;
  }
}

// Create a helper function to generate model recommendations
const generateModelRecommendations = (
  taskType: string,
  columns: ColumnInfo[],
  targetColumn: string,
  numRows: number
): ModelRecommendation[] => {
  const recommendations: ModelRecommendation[] = [];
  
  // Get target column info
  const targetInfo = columns.find(col => col.name === targetColumn);
  if (!targetInfo) return [];
  
  // Get feature columns (excluding target)
  const featureColumns = columns.filter(col => col.name !== targetColumn);
  
  // Count different types of features
  const numFeatures = featureColumns.length;
  const numNumericFeatures = featureColumns.filter(col => 
    col.data_type.includes('int') || col.data_type.includes('float')
  ).length;
  const numCategoricalFeatures = featureColumns.filter(col => 
    col.data_type.includes('object') || col.data_type === 'category'
  ).length;
  
  // Check if we have missing values
  const hasMissingValues = featureColumns.some(col => col.missing_values > 0);
  
  // Check dataset size
  const isSmallDataset = numRows < 1000;
  const isMediumDataset = numRows >= 1000 && numRows < 10000;
  const isLargeDataset = numRows >= 10000;
  
  // Check feature count
  const hasHighDimensionality = numFeatures > 20;
  
  // For classification tasks
  if (taskType === 'classification') {
    // Logistic Regression
    const logisticScore = 70 + 
      (numNumericFeatures > numCategoricalFeatures ? 10 : 0) +
      (isSmallDataset ? 10 : 0) +
      (hasMissingValues ? -5 : 5);
    
    recommendations.push({
      modelType: 'logistic_regression',
      score: Math.min(100, Math.max(0, logisticScore)),
      reasons: [
        'Good for binary and multiclass classification',
        'Provides probability estimates',
        'Highly interpretable - easy to understand feature importance',
        numNumericFeatures > numCategoricalFeatures ? 'Works well with your mostly numeric features' : '',
        isSmallDataset ? 'Suitable for smaller datasets like yours' : '',
        hasMissingValues ? 'Note: May need preprocessing for missing values' : 'Handles your complete data well',
      ].filter(Boolean)
    });
    
    // Random Forest
    const rfScore = 75 + 
      (isLargeDataset ? 10 : 0) +
      (hasMissingValues ? 5 : 0) +
      (hasHighDimensionality ? 10 : 0);
    
    recommendations.push({
      modelType: 'random_forest',
      score: Math.min(100, Math.max(0, rfScore)),
      reasons: [
        'Good balance of accuracy and interpretability',
        'Handles a mix of feature types well',
        'Less prone to overfitting than decision trees',
        isLargeDataset ? 'Works well with your large dataset' : '',
        hasMissingValues ? 'Handles missing values better than some models' : '',
        hasHighDimensionality ? 'Effective with your high-dimensional data' : ''
      ].filter(Boolean)
    });
    
    // Gradient Boosting
    const gbScore = 85 + 
      (isSmallDataset ? -5 : 5) +
      (isMediumDataset ? 10 : 0) +
      (hasMissingValues ? -5 : 5);
    
    recommendations.push({
      modelType: 'gradient_boosting',
      score: Math.min(100, Math.max(0, gbScore)),
      reasons: [
        'Often achieves highest accuracy',
        'Feature importance built-in',
        'Handles complex relationships in data',
        !isSmallDataset ? 'Works well with your larger dataset' : 'May overfit on your smaller dataset',
        hasMissingValues ? 'Needs preprocessing for missing values' : ''
      ].filter(Boolean)
    });
    
    // SVM
    const svmScore = 60 + 
      (isSmallDataset ? 15 : 0) +
      (isMediumDataset ? 5 : 0) +
      (isLargeDataset ? -15 : 0) +
      (numNumericFeatures > numCategoricalFeatures ? 10 : 0) +
      (hasHighDimensionality ? -10 : 5);
    
    recommendations.push({
      modelType: 'svm',
      score: Math.min(100, Math.max(0, svmScore)),
      reasons: [
        'Works well with clear margins between classes',
        'Good for medium sized datasets',
        'Effective with non-linear data when using appropriate kernels',
        isSmallDataset ? 'Well-suited for your smaller dataset' : '',
        isLargeDataset ? 'Warning: May be too slow for your large dataset' : '',
        hasHighDimensionality ? 'Warning: May struggle with your high-dimensional data' : ''
      ].filter(Boolean)
    });
    
    // KNN
    const knnScore = 50 + 
      (isSmallDataset ? 20 : 0) +
      (isMediumDataset ? 0 : 0) +
      (isLargeDataset ? -20 : 0) +
      (hasHighDimensionality ? -15 : 5) +
      (hasMissingValues ? -10 : 5);
    
    recommendations.push({
      modelType: 'knn',
      score: Math.min(100, Math.max(0, knnScore)),
      reasons: [
        'Simple and intuitive',
        'No training phase - predictions made directly from data',
        isSmallDataset ? 'Works well with your smaller dataset' : '',
        isLargeDataset ? 'Warning: Too slow for your large dataset' : '',
        hasHighDimensionality ? 'Warning: Performance degrades in high dimensions' : '',
        hasMissingValues ? 'Warning: Sensitive to missing values' : ''
      ].filter(Boolean)
    });
    
    // Decision Tree
    const dtScore = 65 + 
      (numCategoricalFeatures > numNumericFeatures ? 10 : 0) +
      (isSmallDataset ? 5 : 0) +
      (isLargeDataset ? -5 : 0);
    
    recommendations.push({
      modelType: 'decision_tree',
      score: Math.min(100, Math.max(0, dtScore)),
      reasons: [
        'Highly interpretable - can visualize the tree',
        'Handles mixed data types well',
        'No preprocessing needed for feature scaling',
        numCategoricalFeatures > numNumericFeatures ? 'Works well with your categorical features' : '',
        isSmallDataset ? 'Simple enough for your dataset size' : '',
        isLargeDataset ? 'Warning: May overfit on your large dataset' : ''
      ].filter(Boolean)
    });
    
  } else if (taskType === 'regression') {
    // Linear Regression
    const linearScore = 70 + 
      (numNumericFeatures > numCategoricalFeatures ? 15 : -5) +
      (hasHighDimensionality ? -10 : 5) +
      (hasMissingValues ? -10 : 5);
    
    recommendations.push({
      modelType: 'linear_regression',
      score: Math.min(100, Math.max(0, linearScore)),
      reasons: [
        'Simple and highly interpretable',
        'Fast training and prediction times',
        'Statistical insights about feature importance',
        numNumericFeatures > numCategoricalFeatures ? 'Well-suited for your numeric features' : 'Warning: Needs encoding for your categorical features',
        !hasHighDimensionality ? 'Works well with your feature count' : 'Warning: May not capture complex relationships in high-dimensional data',
        !hasMissingValues ? 'Works well with your complete data' : 'Warning: Sensitive to missing values'
      ].filter(Boolean)
    });
    
    // Random Forest
    const rfScore = 80 + 
      (isLargeDataset ? 10 : 0) +
      (hasMissingValues ? 5 : 0) +
      (hasHighDimensionality ? 10 : 0);
    
    recommendations.push({
      modelType: 'random_forest',
      score: Math.min(100, Math.max(0, rfScore)),
      reasons: [
        'Handles non-linear relationships well',
        'Less prone to overfitting than single decision trees',
        'Feature importance built-in',
        isLargeDataset ? 'Works well with your large dataset' : '',
        hasMissingValues ? 'Handles missing values effectively' : '',
        hasHighDimensionality ? 'Effective with your high-dimensional data' : ''
      ].filter(Boolean)
    });
    
    // Gradient Boosting
    const gbScore = 85 + 
      (isSmallDataset ? -5 : 5) +
      (isMediumDataset ? 10 : 0) +
      (hasHighDimensionality ? 5 : 0);
    
    recommendations.push({
      modelType: 'gradient_boosting',
      score: Math.min(100, Math.max(0, gbScore)),
      reasons: [
        'Often achieves state-of-the-art accuracy',
        'Handles complex non-linear relationships',
        'Feature importance built-in',
        !isSmallDataset ? 'Suitable for your dataset size' : 'May overfit on your smaller dataset',
        hasHighDimensionality ? 'Handles your high-dimensional data well' : ''
      ].filter(Boolean)
    });
    
    // SVM
    const svmScore = 60 + 
      (isSmallDataset ? 15 : 0) +
      (isMediumDataset ? 5 : 0) +
      (isLargeDataset ? -20 : 0) +
      (hasHighDimensionality ? -10 : 5);
    
    recommendations.push({
      modelType: 'svm',
      score: Math.min(100, Math.max(0, svmScore)),
      reasons: [
        'Good for complex, non-linear relationships',
        'Works well when number of features exceeds samples',
        isSmallDataset ? 'Well-suited for your smaller dataset' : '',
        isLargeDataset ? 'Warning: Will be very slow on your large dataset' : '',
        hasHighDimensionality ? 'Warning: May be slow with your high-dimensional data' : ''
      ].filter(Boolean)
    });
    
    // Decision Tree
    const dtScore = 60 + 
      (numCategoricalFeatures > numNumericFeatures ? 10 : 0) +
      (isSmallDataset ? 5 : 0) +
      (isLargeDataset ? -10 : 0);
    
    recommendations.push({
      modelType: 'decision_tree',
      score: Math.min(100, Math.max(0, dtScore)),
      reasons: [
        'Highly interpretable',
        'Handles non-linear relationships',
        'No preprocessing needed for feature scaling',
        numCategoricalFeatures > numNumericFeatures ? 'Handles your categorical features well' : '',
        isSmallDataset ? 'Simple enough for your dataset size' : '',
        isLargeDataset ? 'Warning: Likely to overfit on your large dataset' : ''
      ].filter(Boolean)
    });
    
    // KNN
    const knnScore = 45 + 
      (isSmallDataset ? 20 : 0) +
      (isMediumDataset ? 0 : 0) +
      (isLargeDataset ? -20 : 0) +
      (hasHighDimensionality ? -15 : 5);
    
    recommendations.push({
      modelType: 'knn',
      score: Math.min(100, Math.max(0, knnScore)),
      reasons: [
        'Simple and intuitive',
        'Works well for data with clear patterns',
        isSmallDataset ? 'Suitable for your smaller dataset' : '',
        isLargeDataset ? 'Warning: Too slow for your large dataset' : '',
        hasHighDimensionality ? 'Warning: Performance degrades in high dimensions' : ''
      ].filter(Boolean)
    });
  }
  
  // Sort by score descending
  return recommendations.sort((a, b) => b.score - a.score);
};

export default ModelBuilderPage; 