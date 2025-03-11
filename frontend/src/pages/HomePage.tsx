import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:w-full lg:pb-28 xl:pb-32">
            <div className="px-4 pt-10 sm:px-6 sm:pt-12 md:pt-16 lg:px-8">
              <div className="text-center">
                <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block">Build ML Models</span>
                  <span className="block text-indigo-600">Without Writing Code</span>
                </h1>
                <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
                  Upload your data, select your target, and get a machine learning model with just a few clicks.
                  Get a complete Jupyter notebook with all the code included.
                </p>
                <div className="mt-5 max-w-md mx-auto sm:flex sm:justify-center md:mt-8">
                  <div className="rounded-md shadow">
                    <Link
                      to="/upload"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10"
                    >
                      Get Started
                    </Link>
                  </div>
                  <div className="mt-3 rounded-md shadow sm:mt-0 sm:ml-3">
                    <a
                      href="#features"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
                    >
                      Learn More
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              A better way to build machine learning models
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Our platform makes machine learning accessible to everyone, regardless of coding experience.
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
              <FeatureCard
                title="Upload Any Dataset"
                description="Support for CSV, Excel, and JSON formats. Upload your tabular data and start analyzing instantly."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                }
              />
              <FeatureCard
                title="One-Click Model Training"
                description="Select your target variable and features, choose a model type, and train with a single click."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                }
              />
              <FeatureCard
                title="Instant Notebooks"
                description="Get a complete Jupyter notebook with all the code for data exploration, preprocessing, model training, and evaluation."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                }
              />
              <FeatureCard
                title="Feature Importance"
                description="Understand what features matter most in your model with feature importance visualizations."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                }
              />
              <FeatureCard
                title="Model Explainability"
                description="Get SHAP values and other explainability tools to understand how your model makes predictions."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                }
              />
              <FeatureCard
                title="Beautiful Visualizations"
                description="Explore your data with automatic visualizations and gain insights before training."
                icon={
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                  </svg>
                }
              />
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-gray-50 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">Workflow</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              How It Works
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Our simple, step-by-step process takes you from data to insights in minutes.
            </p>
          </div>
          
          <div className="mt-10">
            <Steps />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-indigo-700">
        <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
            <span className="block">Ready to get started?</span>
            <span className="block">Build your first model today.</span>
          </h2>
          <p className="mt-4 text-lg leading-6 text-indigo-200">
            Upload your dataset and train a machine learning model in minutes, no coding required.
          </p>
          <Link
            to="/upload"
            className="mt-8 w-full inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-white hover:bg-indigo-50 sm:w-auto"
          >
            Get Started
          </Link>
        </div>
      </section>
    </div>
  );
};

// Helper Components
const FeatureCard = ({ title, description, icon }: { title: string; description: string; icon: React.ReactNode }) => {
  return (
    <div className="relative">
      <dt>
        <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">
          {icon}
        </div>
        <p className="ml-16 text-lg leading-6 font-medium text-gray-900">{title}</p>
      </dt>
      <dd className="mt-2 ml-16 text-base text-gray-500">{description}</dd>
    </div>
  );
};

const Steps = () => {
  return (
    <ol className="relative border-l border-gray-200 ml-3">
      <StepItem 
        number={1} 
        title="Upload Your Data" 
        description="Upload your CSV, Excel, or JSON file. The platform will automatically analyze the data types and structure." 
      />
      <StepItem 
        number={2} 
        title="Explore and Visualize" 
        description="Explore your dataset with automatic visualizations and statistics to understand patterns and relationships." 
      />
      <StepItem 
        number={3} 
        title="Configure Your Model" 
        description="Select your target variable, choose which features to include, and pick a machine learning algorithm." 
      />
      <StepItem 
        number={4} 
        title="Train and Evaluate" 
        description="Train your model with one click and view performance metrics and visualizations." 
      />
      <StepItem 
        number={5} 
        title="Download Notebook" 
        description="Download a complete Jupyter notebook with all the Python code used to create your model." 
      />
    </ol>
  );
};

const StepItem = ({ number, title, description }: { number: number; title: string; description: string }) => {
  return (
    <li className="mb-10 ml-6">
      <span className="flex absolute -left-3 justify-center items-center w-6 h-6 bg-indigo-200 rounded-full ring-8 ring-white">
        <span className="text-indigo-800 font-bold text-sm">{number}</span>
      </span>
      <h3 className="flex items-center mb-1 text-lg font-semibold text-gray-900">{title}</h3>
      <p className="mb-4 text-base text-gray-500">{description}</p>
    </li>
  );
};

export default HomePage; 