import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

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

const NotebookPage = () => {
  const { notebookId } = useParams<{ notebookId: string }>();
  const navigate = useNavigate();
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notebookInfo, setNotebookInfo] = useState<NotebookInfo | null>(null);
  const [notebookContent, setNotebookContent] = useState<any | null>(null);
  
  useEffect(() => {
    const fetchNotebook = async () => {
      if (!notebookId) {
        setError('Notebook ID is missing');
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        
        // Fetch notebook metadata
        const metadataResponse = await axios.get(`${API_URL}/api/notebooks/${notebookId}/metadata`);
        setNotebookInfo(metadataResponse.data);
        
        // Fetch notebook content
        const contentResponse = await axios.get(`${API_URL}/api/notebooks/${notebookId}`);
        setNotebookContent(contentResponse.data);
        
      } catch (error) {
        console.error('Error fetching notebook:', error);
        if (axios.isAxiosError(error) && error.response) {
          setError(error.response.data.detail || 'Error fetching notebook');
        } else {
          setError('Network error. Please check your connection.');
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchNotebook();
  }, [notebookId]);
  
  const handleDownload = () => {
    if (notebookInfo) {
      window.open(`${API_URL}${notebookInfo.download_url}`, '_blank');
    }
  };
  
  const handleBackToModel = () => {
    if (notebookInfo && notebookInfo.dataset_id) {
      navigate(`/build/${notebookInfo.dataset_id}`);
    } else {
      navigate('/');
    }
  };
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="bg-red-100 p-4 rounded-md text-red-700">
        <h2 className="text-lg font-semibold mb-2">Error</h2>
        <p>{error}</p>
        <button 
          onClick={() => navigate('/')}
          className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
        >
          Back to Home
        </button>
      </div>
    );
  }
  
  if (!notebookInfo) {
    return (
      <div className="bg-yellow-100 p-4 rounded-md text-yellow-700">
        <h2 className="text-lg font-semibold mb-2">Notebook Not Found</h2>
        <p>The requested notebook could not be found.</p>
        <button 
          onClick={() => navigate('/')}
          className="mt-4 bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700"
        >
          Back to Home
        </button>
      </div>
    );
  }
  
  // For the purpose of this demo, we'll display a placeholder view
  // In a real application, we would use a Jupyter notebook viewer component
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-800">
          Notebook: {notebookInfo.model_type} Model
        </h1>
        
        <div className="flex space-x-3">
          <button 
            onClick={handleBackToModel}
            className="btn btn-secondary"
          >
            Back to Model
          </button>
          
          <button 
            onClick={handleDownload}
            className="btn btn-primary"
          >
            Download Notebook
          </button>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Model Information</h2>
        
        <div className="grid grid-cols-2 gap-x-6 gap-y-3">
          <div>
            <span className="font-medium text-gray-700">Model Type:</span>
            <span className="ml-2">{notebookInfo.model_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
          </div>
          
          <div>
            <span className="font-medium text-gray-700">Task:</span>
            <span className="ml-2">{notebookInfo.task_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
          </div>
          
          <div>
            <span className="font-medium text-gray-700">Target Column:</span>
            <span className="ml-2">{notebookInfo.target_column}</span>
          </div>
          
          <div>
            <span className="font-medium text-gray-700">Created On:</span>
            <span className="ml-2">{new Date(notebookInfo.creation_timestamp).toLocaleString()}</span>
          </div>
          
          <div className="col-span-2">
            <span className="font-medium text-gray-700">Features:</span>
            <span className="ml-2">{notebookInfo.feature_columns.join(', ')}</span>
          </div>
        </div>
      </div>
      
      {notebookInfo.metrics && Object.keys(notebookInfo.metrics).length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Performance Metrics</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(notebookInfo.metrics).map(([key, value]) => (
              <div key={key} className="bg-blue-50 p-3 rounded-md">
                <div className="text-sm text-gray-500">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div className="font-medium text-xl mt-1">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Preview</h2>
        
        <div className="bg-gray-800 text-white p-4 rounded-md overflow-auto">
          <p className="text-yellow-300">
            # This is a preview of the generated Jupyter notebook
          </p>
          <p className="mt-2">
            # The full notebook can be downloaded using the button above
          </p>
          <p className="mt-4 text-green-300">
            # It contains all the code needed to:
          </p>
          <p className="text-gray-300 ml-4">
            # 1. Load and explore the dataset
          </p>
          <p className="text-gray-300 ml-4">
            # 2. Preprocess the data
          </p>
          <p className="text-gray-300 ml-4">
            # 3. Train a {notebookInfo.model_type.replace(/_/g, ' ')} model
          </p>
          <p className="text-gray-300 ml-4">
            # 4. Evaluate the model performance
          </p>
          <p className="text-gray-300 ml-4">
            # 5. Visualize the results
          </p>
          <p className="text-gray-300 ml-4">
            # 6. Explain the model using SHAP values
          </p>
        </div>
        
        <div className="mt-4 text-center">
          <button 
            onClick={handleDownload}
            className="btn btn-primary px-8"
          >
            Download Full Notebook
          </button>
        </div>
      </div>
    </div>
  );
};

export default NotebookPage; 