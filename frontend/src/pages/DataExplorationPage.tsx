import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Dataset } from '../App';
import InfoIcon from '../components/InfoIcon';
import Tooltip from '../components/Tooltip';
// Import Recharts components
import { 
  BarChart, Bar, 
  PieChart, Pie, Cell, 
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';

interface DataExplorationPageProps {
  currentDataset: Dataset | null;
  setCurrentDataset: (dataset: Dataset) => void;
}

interface ColumnInfo {
  name: string;
  data_type: string;
  unique_values: number;
  missing_values: number;
  sample_values: any[];
  min_value?: number;
  max_value?: number;
  mean?: number;
  median?: number;
  std_dev?: number;
  top_categories?: Record<string, number>;
}

interface DatasetPreview {
  id: string;
  filename: string;
  file_type: string;
  upload_timestamp: string;
  num_rows: number;
  num_columns: number;
  sample_rows: Record<string, any>[];
  columns: ColumnInfo[];
}

const API_URL = 'http://localhost:8000';

// Helper function to safely render values of any type
const safeRenderValue = (value: any): string => {
  if (value === undefined || value === null) {
    return 'N/A';
  }
  
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  }
  
  if (typeof value === 'string') {
    return value;
  }
  
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value).substring(0, 50) + (JSON.stringify(value).length > 50 ? '...' : '');
    } catch (e) {
      return 'Complex object';
    }
  }
  
  return String(value);
};

// Helper component for displaying info cards
const InfoCard = ({ label, value }: { label: string; value: any }) => {
  const displayValue = typeof value === 'string' || typeof value === 'number' 
    ? value 
    : safeRenderValue(value);
    
  return (
    <div className="bg-gray-50 p-3 rounded-md">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="font-medium mt-1">{displayValue}</div>
    </div>
  );
};

// Helper functions for chart data preparation
const prepareHistogramData = (values: any[], bins = 10): any[] => {
  if (!values || values.length === 0) return [];
  
  // Filter out non-numeric or null values
  const numericValues = values.filter(v => typeof v === 'number' && v !== null);
  if (numericValues.length === 0) return [];
  
  // Calculate min and max
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  
  // Create bins
  const binWidth = (max - min) / bins;
  const histData = Array(bins).fill(0).map((_, i) => ({
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
    count: 0,
    binName: `${(min + i * binWidth).toFixed(1)} - ${(min + (i + 1) * binWidth).toFixed(1)}`
  }));
  
  // Count values in each bin
  numericValues.forEach(val => {
    const binIndex = Math.min(Math.floor((val - min) / binWidth), bins - 1);
    histData[binIndex].count++;
  });
  
  return histData;
};

const prepareCategoryData = (categories: Record<string, number>): any[] => {
  if (!categories) return [];
  return Object.entries(categories).map(([name, value]) => ({
    name,
    value
  }));
};

// Colors for charts
const CHART_COLORS = [
  '#8884d8', '#83a6ed', '#8dd1e1', '#82ca9d', '#a4de6c',
  '#d0ed57', '#ffc658', '#ff8042', '#ff6361', '#bc5090'
];

const DataExplorationPage: React.FC<DataExplorationPageProps> = ({ 
  currentDataset, 
  setCurrentDataset 
}) => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataPreview, setDataPreview] = useState<DatasetPreview | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  
  // Add refs to track scroll position
  const scrollPositionRef = useRef<number>(0);
  
  // Modified setSelectedColumn function to preserve scroll
  const handleColumnSelect = (columnName: string) => {
    // Save current scroll position before state update
    scrollPositionRef.current = window.scrollY;
    setSelectedColumn(columnName);
    
    // Use setTimeout to restore scroll position after render
    setTimeout(() => {
      window.scrollTo(0, scrollPositionRef.current);
    }, 0);
  };
  
  useEffect(() => {
    const fetchDatasetPreview = async () => {
      if (!datasetId) {
        setError('Dataset ID is missing');
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        const response = await axios.get(`${API_URL}/api/datasets/${datasetId}`);
        const preview = response.data as DatasetPreview;
        
        setDataPreview(preview);
        
        // If we don't have the dataset info, update it
        if (!currentDataset || currentDataset.id !== preview.id) {
          setCurrentDataset({
            id: preview.id,
            filename: preview.filename,
            file_type: preview.file_type,
            upload_timestamp: preview.upload_timestamp,
            num_rows: preview.num_rows,
            num_columns: preview.num_columns,
            column_names: preview.columns.map(col => col.name),
          });
        }
        
        // Set first column as selected by default
        if (preview.columns.length > 0 && !selectedColumn) {
          setSelectedColumn(preview.columns[0].name);
        }
        
      } catch (error) {
        console.error('Error fetching dataset preview:', error);
        if (axios.isAxiosError(error) && error.response) {
          setError(error.response.data.detail || 'Error fetching dataset preview');
        } else {
          setError('Network error. Please check your connection.');
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchDatasetPreview();
  }, [datasetId, currentDataset, setCurrentDataset, selectedColumn]);
  
  const getSelectedColumnInfo = (): ColumnInfo | null => {
    if (!dataPreview || !selectedColumn) return null;
    return dataPreview.columns.find(col => col.name === selectedColumn) || null;
  };
  
  const handleContinueToModelBuilding = () => {
    if (datasetId) {
      navigate(`/build/${datasetId}`);
    }
  };
  
  // Add explanation for data exploration
  const dataExplorationGuide = (
    <div className="mb-6 bg-blue-50 border-l-4 border-blue-400 p-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-blue-800">Understanding Your Data</h3>
          <div className="mt-2 text-sm text-blue-700">
            <p>Exploring your data helps you understand its characteristics before building a model. Look for patterns, relationships, missing values, and potential issues.</p>
            <ul className="list-disc list-inside mt-1">
              <li>Click on different columns to see their statistics and distributions</li>
              <li>Pay attention to data types, missing values, and unique values</li>
              <li>The data preview at the bottom shows sample rows from your dataset</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
  
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
      </div>
    );
  }
  
  if (!dataPreview) {
    return (
      <div className="bg-yellow-100 p-4 rounded-md text-yellow-700">
        <h2 className="text-lg font-semibold mb-2">No Data Available</h2>
        <p>No preview data available for this dataset.</p>
      </div>
    );
  }
  
  const columnInfo = getSelectedColumnInfo();
  
  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Explore Dataset</h1>
          <p className="mt-2 text-sm text-gray-500">
            Examine your data before building a model to understand its characteristics
          </p>
        </div>
        <button 
          onClick={handleContinueToModelBuilding}
          className="btn btn-primary"
        >
          Continue to Model Building
        </button>
      </div>

      {dataExplorationGuide}
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
          Dataset Information
          <InfoIcon content="Basic information about your uploaded dataset" position="right" />
        </h2>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
          <InfoCard label="Filename" value={dataPreview.filename} />
          <InfoCard label="File Type" value={dataPreview.file_type.toUpperCase()} />
          <Tooltip content="The total number of rows (observations) in your dataset. Each row represents a single data point.">
            <InfoCard label="Rows" value={dataPreview.num_rows.toLocaleString()} />
          </Tooltip>
          <Tooltip content="The total number of columns (features) in your dataset. Each column represents a different attribute.">
            <InfoCard label="Columns" value={dataPreview.num_columns.toLocaleString()} />
          </Tooltip>
          <InfoCard 
            label="Upload Date" 
            value={new Date(dataPreview.upload_timestamp).toLocaleString()} 
          />
        </div>
        
        {/* Dataset column types overview */}
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-3 text-gray-700 flex items-center">
            Column Types Overview
            <InfoIcon content="Summary of data types in your dataset" position="right" />
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={(() => {
                  // Count column types
                  const typeCount: Record<string, number> = {};
                  dataPreview.columns.forEach(col => {
                    const type = col.data_type.includes('int') || col.data_type.includes('float') 
                      ? 'Numeric' 
                      : col.data_type.includes('date') || col.data_type.includes('time')
                        ? 'DateTime'
                        : col.data_type.includes('bool')
                          ? 'Boolean'
                          : 'Categorical';
                    typeCount[type] = (typeCount[type] || 0) + 1;
                  });
                  
                  // Convert to array
                  return Object.entries(typeCount).map(([type, count]) => ({
                    type,
                    count
                  }));
                })()}
                margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" />
                <YAxis />
                <RechartsTooltip formatter={(value) => [`${value} columns`, 'Count']} />
                <Bar dataKey="count" fill="#82ca9d" name="Columns" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-xs text-gray-500 mt-2 text-center">
            Distribution of data types in your dataset
          </div>
        </div>
        
        {/* Missing values overview */}
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-3 text-gray-700 flex items-center">
            Missing Values Overview
            <InfoIcon content="Top 10 columns with missing values" position="right" />
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={dataPreview.columns
                  .filter(col => col.missing_values > 0)
                  .sort((a, b) => b.missing_values - a.missing_values)
                  .slice(0, 10)
                  .map(col => ({
                    name: col.name,
                    missing: col.missing_values,
                    percentage: (col.missing_values / dataPreview.num_rows) * 100
                  }))}
                margin={{ top: 10, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={70} 
                  tick={{ fontSize: 12 }}
                />
                <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                <YAxis 
                  yAxisId="right" 
                  orientation="right" 
                  stroke="#82ca9d" 
                  unit="%" 
                  domain={[0, 100]} 
                />
                <RechartsTooltip 
                  formatter={(value, name) => [
                    name === 'missing' ? `${value} rows` : `${typeof value === 'number' ? value.toFixed(1) : value}%`, 
                    name === 'missing' ? 'Missing Count' : 'Missing Percentage'
                  ]} 
                />
                <Bar yAxisId="left" dataKey="missing" fill="#8884d8" name="Missing Count" />
                <Bar yAxisId="right" dataKey="percentage" fill="#82ca9d" name="Missing Percentage" />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-xs text-gray-500 mt-2 text-center">
            Columns with the most missing values
          </div>
        </div>
      </div>
      
      <div className="grid md:grid-cols-4 gap-6">
        <div className="md:col-span-1 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
            Columns
            <InfoIcon content="The features in your dataset. Click on a column to see its details." position="right" />
          </h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {dataPreview.columns.map(column => (
              <div 
                key={column.name}
                onClick={() => handleColumnSelect(column.name)}
                className={`p-3 rounded-md cursor-pointer hover:bg-blue-50 transition-colors
                  ${selectedColumn === column.name ? 'bg-blue-100 border-l-4 border-blue-500' : ''}`}
              >
                <div className="font-medium truncate">{column.name}</div>
                <div className="text-xs text-gray-500 flex justify-between">
                  <span>{column.data_type}</span>
                  {column.missing_values > 0 && (
                    <span className="text-amber-600">{column.missing_values} missing</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="md:col-span-3 bg-white rounded-lg shadow-md p-6">
          {columnInfo ? (
            <div>
              <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
                Column Details: <span className="text-blue-600 ml-1">{columnInfo.name}</span>
                <InfoIcon content="Detailed statistics and information about this column" position="right" />
              </h2>
              
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                <Tooltip content="The kind of data in this column (numeric, text, date, etc.)">
                  <InfoCard label="Data Type" value={columnInfo.data_type} />
                </Tooltip>
                <Tooltip content="The number of different values in this column. A high count might indicate an ID column.">
                  <InfoCard label="Unique Values" value={columnInfo.unique_values.toLocaleString()} />
                </Tooltip>
                <Tooltip content="The number of missing values. High percentages of missing data might affect model accuracy.">
                  <InfoCard 
                    label="Missing Values" 
                    value={`${columnInfo.missing_values.toLocaleString()} 
                           (${((columnInfo.missing_values / dataPreview.num_rows) * 100).toFixed(2)}%)`} 
                  />
                </Tooltip>
                
                {columnInfo?.min_value !== undefined && (
                  <Tooltip content="The minimum value in this column. Useful to spot outliers or errors.">
                    <InfoCard 
                      label="Minimum" 
                      value={safeRenderValue(columnInfo.min_value)}
                    />
                  </Tooltip>
                )}
                
                {columnInfo?.max_value !== undefined && (
                  <Tooltip content="The maximum value in this column. Useful to spot outliers or errors.">
                    <InfoCard 
                      label="Maximum" 
                      value={safeRenderValue(columnInfo.max_value)}
                    />
                  </Tooltip>
                )}
                
                {columnInfo?.mean !== undefined && (
                  <Tooltip content="The average value. Compare with median to understand distribution.">
                    <InfoCard 
                      label="Mean" 
                      value={safeRenderValue(columnInfo.mean)}
                    />
                  </Tooltip>
                )}
                
                {columnInfo?.median !== undefined && (
                  <Tooltip content="The middle value. Not affected by outliers unlike the mean.">
                    <InfoCard 
                      label="Median" 
                      value={safeRenderValue(columnInfo.median)}
                    />
                  </Tooltip>
                )}
                
                {columnInfo?.std_dev !== undefined && (
                  <Tooltip content="Standard deviation shows how spread out the values are. Lower values indicate data is clustered near the mean.">
                    <InfoCard 
                      label="Std. Deviation" 
                      value={safeRenderValue(columnInfo.std_dev)}
                    />
                  </Tooltip>
                )}
              </div>
              
              {/* Data Visualization Section */}
              <div className="mb-6">
                <h3 className="text-lg font-medium mb-3 text-gray-700 flex items-center">
                  Data Visualization
                  <InfoIcon content="Visual representation of the data distribution to help you understand patterns" position="right" />
                </h3>
                
                {/* Show histogram for numeric columns */}
                {columnInfo.data_type.includes('int') || columnInfo.data_type.includes('float') ? (
                  <div className="bg-white p-3 rounded-md mb-4 border border-gray-200">
                    <h4 className="text-md font-medium mb-2 text-gray-700">Value Distribution</h4>
                    <div className="h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={prepareHistogramData(columnInfo.sample_values)}
                          margin={{ top: 10, right: 30, left: 20, bottom: 40 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="binName" 
                            angle={-45} 
                            textAnchor="end" 
                            height={70} 
                            tick={{ fontSize: 12 }}
                          />
                          <YAxis />
                          <RechartsTooltip 
                            formatter={(value: any, name: string) => [
                              `Count: ${value}`, 
                              "Frequency"
                            ]}
                            labelFormatter={(label) => `Range: ${label}`}
                          />
                          <Bar 
                            dataKey="count" 
                            fill="#8884d8" 
                            name="Frequency" 
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-gray-500 mt-2 text-center">
                      Distribution of values in {columnInfo.name}
                    </div>
                  </div>
                ) : columnInfo.top_categories ? (
                  /* Show pie chart for categorical columns */
                  <div className="bg-white p-3 rounded-md mb-4 border border-gray-200">
                    <h4 className="text-md font-medium mb-2 text-gray-700">Category Distribution</h4>
                    <div className="h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={prepareCategoryData(columnInfo.top_categories)}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {prepareCategoryData(columnInfo.top_categories).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                            ))}
                          </Pie>
                          <RechartsTooltip 
                            formatter={(value: any, name: string, props: any) => [
                              `Count: ${value} (${((value / dataPreview.num_rows) * 100).toFixed(1)}%)`, 
                              props.payload.name
                            ]}
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-gray-500 mt-2 text-center">
                      Distribution of categories in {columnInfo.name}
                    </div>
                  </div>
                ) : null}
                
                {/* Scatter plot for numeric columns to show data points */}
                {(columnInfo.data_type.includes('int') || columnInfo.data_type.includes('float')) && (
                  <div className="bg-white p-3 rounded-md mb-4 border border-gray-200">
                    <h4 className="text-md font-medium mb-2 text-gray-700">Value Scatter</h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart
                          margin={{ top: 10, right: 30, left: 20, bottom: 30 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            type="number" 
                            dataKey="index" 
                            name="Index" 
                            domain={['dataMin - 5', 'dataMax + 5']}
                            label={{ value: 'Data Point Index', position: 'bottom', offset: 10 }}
                          />
                          <YAxis 
                            type="number" 
                            dataKey="value" 
                            name="Value" 
                            domain={['auto', 'auto']}
                            label={{ value: columnInfo.name, angle: -90, position: 'left' }}
                          />
                          <RechartsTooltip 
                            cursor={{ strokeDasharray: '3 3' }}
                            formatter={(value: any) => [value, columnInfo.name]}
                            labelFormatter={(label) => `Point: ${label}`}
                          />
                          <Scatter 
                            name={columnInfo.name} 
                            data={columnInfo.sample_values
                              .filter(v => typeof v === 'number' && !isNaN(v))
                              .map((v, i) => ({ index: i, value: v }))} 
                            fill="#8884d8" 
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-gray-500 mt-2 text-center">
                      Scatter plot of sample values
                    </div>
                  </div>
                )}
              </div>
              
              {columnInfo.top_categories && (
                <div className="mb-6">
                  <h3 className="text-lg font-medium mb-2 text-gray-700 flex items-center">
                    Top Categories
                    <InfoIcon content="Most frequent values in this categorical column" position="right" />
                  </h3>
                  <div className="bg-gray-50 p-3 rounded-md">
                    {Object.entries(columnInfo.top_categories).map(([category, count]) => (
                      <div key={category} className="flex justify-between mb-1">
                        <span className="font-medium">{category}</span>
                        <span>{count.toLocaleString()} ({((count / dataPreview.num_rows) * 100).toFixed(1)}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <h3 className="text-lg font-medium mb-2 text-gray-700 flex items-center">
                Sample Values
                <InfoIcon content="Examples of values in this column to help you understand the data" position="right" />
              </h3>
              <div className="bg-gray-50 p-3 rounded-md overflow-x-auto">
                <div className="space-y-1">
                  {columnInfo.sample_values.map((value, index) => (
                    <div key={index} className="font-mono">{JSON.stringify(value)}</div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-center py-8">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">Select a column to view details</h3>
              <p className="mt-1 text-sm text-gray-500">
                Click on any column name from the list on the left
              </p>
            </div>
          )}
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
          Data Preview
          <InfoIcon content="A sample of rows from your dataset to help you understand the data structure" position="right" />
        </h2>
        
        <div className="mb-4 text-sm text-gray-600 flex items-center">
          <svg className="h-5 w-5 text-indigo-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <span>Showing a sample of your data. Examine the values to ensure they look correct before building a model.</span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-100">
              <tr>
                {dataPreview.columns.map(column => (
                  <th 
                    key={column.name}
                    className="px-4 py-2 text-left text-sm font-semibold text-gray-700"
                  >
                    {column.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {dataPreview.sample_rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-gray-50">
                  {dataPreview.columns.map(column => (
                    <td 
                      key={`${rowIndex}-${column.name}`}
                      className="px-4 py-2 text-sm text-gray-700 max-w-xs truncate"
                    >
                      {JSON.stringify(row[column.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 text-gray-500 text-sm">
          Showing {dataPreview.sample_rows.length} of {dataPreview.num_rows.toLocaleString()} rows
        </div>
      </div>

      <div className="flex justify-end">
        <button 
          onClick={handleContinueToModelBuilding}
          className="btn btn-primary"
        >
          Continue to Model Building →
        </button>
      </div>
    </div>
  );
};

export default DataExplorationPage; 