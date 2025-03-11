import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Dataset } from '../App';
import InfoIcon from '../components/InfoIcon';

interface DataUploadPageProps {
  setCurrentDataset: (dataset: Dataset) => void;
}

const API_URL = 'http://localhost:8000';

const DataUploadPage: React.FC<DataUploadPageProps> = ({ setCurrentDataset }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const navigate = useNavigate();
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    // Only accept one file at a time
    if (acceptedFiles.length !== 1) {
      setUploadError('Please upload only one file at a time.');
      return;
    }
    
    const file = acceptedFiles[0];
    
    // Check file type
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    const allowedExtensions = ['csv', 'xlsx', 'xls', 'json'];
    
    if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
      setUploadError(`File type not supported. Please upload CSV, Excel, or JSON files.`);
      return;
    }
    
    // Reset error state
    setUploadError(null);
    setIsUploading(true);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Upload file to API
      const response = await axios.post(`${API_URL}/api/datasets/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Set dataset information
      const dataset = response.data as Dataset;
      setCurrentDataset(dataset);
      
      // Navigate to exploration page
      navigate(`/explore/${dataset.id}`);
    } catch (error) {
      console.error('Upload error:', error);
      if (axios.isAxiosError(error) && error.response) {
        setUploadError(error.response.data.detail || 'Error uploading file. Please try again.');
      } else {
        setUploadError('Network error. Please check your connection and try again.');
      }
    } finally {
      setIsUploading(false);
    }
  }, [navigate, setCurrentDataset]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    disabled: isUploading,
  });
  
  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">Upload Your Dataset</h1>
        <p className="mt-4 text-lg text-gray-500 flex flex-col items-center justify-center">
          Upload a tabular dataset to start building your machine learning model
          <span className="inline-flex items-center mt-2 text-sm bg-blue-50 text-blue-700 px-3 py-1 rounded-full">
            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            No coding skills needed - our platform handles everything automatically
          </span>
        </p>
      </div>
      
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center mb-4">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Step 1: Select Your Data File</h3>
            <InfoIcon 
              content="Upload your dataset in CSV, Excel, or JSON format. The file should have columns (features) and rows (samples/observations). Each feature should be in a separate column." 
              position="right"
            />
          </div>
          
          <div 
            {...getRootProps()} 
            className={`mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-dashed rounded-lg transition-colors duration-200
              ${isDragActive 
                ? 'border-indigo-500 bg-indigo-50' 
                : 'border-gray-300 hover:border-indigo-400'
              } ${isUploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <input {...getInputProps()} />
            
            <div className="space-y-1 text-center">
              {isUploading ? (
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mb-4"></div>
                  <p className="text-indigo-600 font-medium">Uploading file...</p>
                  <p className="text-xs text-gray-500 mt-1">This may take a moment depending on file size</p>
                </div>
              ) : isDragActive ? (
                <div className="flex flex-col items-center py-8">
                  <svg 
                    className="w-12 h-12 text-indigo-500 mb-3" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24" 
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-indigo-600 text-lg font-medium">Drop your file here</p>
                </div>
              ) : (
                <div className="flex flex-col items-center py-6">
                  <svg 
                    className="w-12 h-12 text-gray-400 mb-3" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24" 
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  
                  <p className="text-gray-700 font-medium">
                    Drag and drop your file here, or <span className="text-indigo-600">browse</span>
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supported formats: CSV, Excel, JSON (Max size: 10MB)
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {uploadError && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-500 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg 
                    className="h-5 w-5 text-red-400" 
                    fill="currentColor" 
                    viewBox="0 0 20 20" 
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path 
                      fillRule="evenodd" 
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" 
                      clipRule="evenodd" 
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{uploadError}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-8 bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <div className="flex items-center">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Dataset Requirements</h3>
            <InfoIcon 
              content="Following these guidelines will help the system better understand your data and build more accurate models." 
              position="right"
            />
          </div>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">
            Make sure your data meets these requirements for the best results.
          </p>
        </div>
        <div className="border-t border-gray-200">
          <dl>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                Format
                <InfoIcon content="The structure of your data file" position="right" />
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                CSV, Excel (XLSX/XLS), or JSON files with tabular data. One row per observation and one column per feature.
              </dd>
            </div>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                Headers
                <InfoIcon content="Column names that describe each feature" position="right" />
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                Files should have a header row with column names. For example: "Age", "Income", "Occupation".
              </dd>
            </div>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                Data Types
                <InfoIcon content="The kind of information in each column" position="right" />
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                <ul className="space-y-1 list-disc list-inside text-sm">
                  <li><span className="font-medium">Numerical:</span> Numbers like age, price, or temperature (integers or decimals)</li>
                  <li><span className="font-medium">Categorical:</span> Categories like gender, country, or product type</li>
                  <li><span className="font-medium">Datetime:</span> Date and time values</li>
                  <li><span className="font-medium">Boolean:</span> True/False values</li>
                </ul>
              </dd>
            </div>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                Missing Values
                <InfoIcon content="Cells in your data that have no value" position="right" />
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                Files can have missing values. The system will handle them appropriately during preprocessing.
                Common representations include empty cells, "NA", "N/A", "null", or "NaN".
              </dd>
            </div>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                Best Practices
                <InfoIcon content="Recommendations for better results" position="right" />
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                <ul className="space-y-1 list-disc list-inside text-sm">
                  <li>Ensure your data is clean and consistent</li>
                  <li>Remove duplicate records</li>
                  <li>Make sure column names are descriptive</li>
                  <li>Include a good mix of relevant features</li>
                  <li>For best results, aim for at least 50 rows of data</li>
                </ul>
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
};

export default DataUploadPage; 