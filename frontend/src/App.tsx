import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DataUploadPage from './pages/DataUploadPage';
import DataExplorationPage from './pages/DataExplorationPage';
import ModelBuilderPage from './pages/ModelBuilderPage';
import NotebookPage from './pages/NotebookPage';
import NotFoundPage from './pages/NotFoundPage';

// Define the Dataset context types
export interface Dataset {
  id: string;
  filename: string;
  file_type: string;
  upload_timestamp: string;
  num_rows: number;
  num_columns: number;
  column_names: string[];
}

function App() {
  // State for the currently selected dataset
  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(null);

  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-background text-foreground">
        <Navbar />
        <main className="flex-grow py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route 
                path="/upload" 
                element={<DataUploadPage setCurrentDataset={setCurrentDataset} />} 
              />
              <Route 
                path="/explore/:datasetId" 
                element={<DataExplorationPage currentDataset={currentDataset} setCurrentDataset={setCurrentDataset} />} 
              />
              <Route 
                path="/build/:datasetId" 
                element={<ModelBuilderPage currentDataset={currentDataset} />} 
              />
              <Route path="/notebook/:notebookId" element={<NotebookPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </div>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
