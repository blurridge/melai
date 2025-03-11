# ML-AI: No-Code Machine Learning Web Application

A web application that empowers users with no coding experience to build machine learning models using tabulated data and generate Jupyter notebooks containing the code.

## Features

- Upload and analyze CSV, Excel, and JSON datasets
- Automated data exploration and visualization
- Build machine learning models without writing code
- Choose from various algorithms (Linear Regression, Decision Trees, Random Forests, etc.)
- Automated feature engineering and preprocessing
- Model evaluation with performance metrics
- Feature importance analysis and model explainability with SHAP
- Download generated Jupyter Notebooks with complete code

## Technical Overview

### Frontend (React + Vite)

- **React**: Component-based UI library
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Data visualization
- **React Router**: Client-side routing
- **Axios**: HTTP client for API requests

### Backend (FastAPI)

- **FastAPI**: High-performance, async API framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **SHAP**: Model explainability
- **nbformat/nbconvert**: Jupyter notebook generation

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/melai.git
   cd melai
   ```

2. Set up the frontend:
   ```
   cd frontend
   npm install
   ```

3. Set up the backend:
   ```
   cd ../backend
   pip install -r requirements.txt
   ```

### Running the Application

You can run both the frontend and backend servers with a single command:

```
./start.sh
```

Or run them separately:

1. Start the backend server:
   ```
   cd backend
   python main.py
   ```

2. Start the frontend development server:
   ```
   cd frontend
   npm run dev
   ```

3. Open your browser and navigate to: http://localhost:5173

## Usage Guide

1. Upload your dataset (CSV, Excel, or JSON format)
2. Explore the data with automatic visualizations and statistics
3. Select your target variable and features
4. Choose a machine learning model and configure parameters
5. Train the model and view performance metrics
6. Analyze feature importance and model explanations
7. Download the generated Jupyter notebook with all code included

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 