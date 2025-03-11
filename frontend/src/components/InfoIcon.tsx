import React from 'react';
import Tooltip from './Tooltip';

interface InfoIconProps {
  content: string;
  position?: 'top' | 'right' | 'bottom' | 'left';
  iconClass?: string;
  tooltipWidth?: string;
}

const InfoIcon: React.FC<InfoIconProps> = ({ 
  content, 
  position = 'top', 
  iconClass = "ml-1.5 w-4 h-4",
  tooltipWidth = "max-w-xs"
}) => {
  return (
    <Tooltip content={content} position={position} width={tooltipWidth}>
      <span className="inline-flex text-gray-400 hover:text-gray-500">
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          className={iconClass} 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
          />
        </svg>
      </span>
    </Tooltip>
  );
};

export default InfoIcon; 