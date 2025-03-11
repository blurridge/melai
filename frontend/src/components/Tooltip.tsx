import { useState, ReactNode } from 'react';

interface TooltipProps {
  children: ReactNode;
  content: string;
  position?: 'top' | 'right' | 'bottom' | 'left';
  width?: string;
}

const Tooltip = ({ 
  children, 
  content, 
  position = 'top', 
  width = 'max-w-xs' 
}: TooltipProps) => {
  const [isVisible, setIsVisible] = useState(false);

  const positionClasses = {
    top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
    right: 'left-full top-1/2 transform -translate-y-1/2 ml-2',
    bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 transform -translate-y-1/2 mr-2'
  };

  const arrowClasses = {
    top: 'top-full left-1/2 transform -translate-x-1/2 border-l-transparent border-r-transparent border-b-0 border-t-gray-800',
    right: 'right-full top-1/2 transform -translate-y-1/2 border-t-transparent border-b-transparent border-l-0 border-r-gray-800',
    bottom: 'bottom-full left-1/2 transform -translate-x-1/2 border-l-transparent border-r-transparent border-t-0 border-b-gray-800',
    left: 'left-full top-1/2 transform -translate-y-1/2 border-t-transparent border-b-transparent border-r-0 border-l-gray-800'
  };

  return (
    <div 
      className="relative inline-block" 
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
      onClick={() => setIsVisible(!isVisible)}
    >
      <div className="inline-flex items-center">
        {children}
      </div>
      
      {isVisible && (
        <div 
          className={`absolute z-50 ${positionClasses[position]} ${width} bg-gray-800 text-white text-sm rounded-lg p-3 shadow-lg transition-opacity duration-300`}
          role="tooltip"
        >
          <div className={`absolute w-0 h-0 border-4 ${arrowClasses[position]}`}></div>
          <div className="relative z-20">
            {content}
          </div>
        </div>
      )}
    </div>
  );
};

export default Tooltip; 