const Footer = () => {
  return (
    <footer className="bg-card border-t border-border py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="container flex flex-col items-center justify-between gap-4 px-4 md:flex-row md:gap-0">
          <div className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} MeLAI Platform. All rights reserved.
          </div>
          
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <a href="#" className="text-muted-foreground hover:text-foreground">
              Privacy Policy
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground">
              Terms of Service
            </a>
            <a href="#" className="text-muted-foreground hover:text-foreground">
              Help
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 