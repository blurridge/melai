const Footer = () => {
  return (
    <footer className="bg-card border-t border-border py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row justify-between items-center">
          <div className="text-center sm:text-left mb-4 sm:mb-0">
            <span className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} ML-AI Platform. All rights reserved.
            </span>
          </div>
          <div className="flex space-x-6">
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
              Terms of Service
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
              Help
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 