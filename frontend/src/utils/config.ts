/**
 * Application Configuration
 * 
 * This file centralizes access to environment variables and configuration
 * settings for the MeLAI Platform frontend. All environment variables
 * should be accessed through this file to ensure consistent defaults and usage.
 */

// API Configuration
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Application Information
export const APP_TITLE = import.meta.env.VITE_APP_TITLE || 'MeLAI Platform';
export const APP_VERSION = import.meta.env.VITE_APP_VERSION || '1.0.0';

// Feature Flags
export const ENABLE_ADVANCED_FEATURES = 
  (import.meta.env.VITE_ENABLE_ADVANCED_FEATURES || 'false').toLowerCase() === 'true';
export const ENABLE_DEBUG_MODE = 
  (import.meta.env.VITE_ENABLE_DEBUG_MODE || 'false').toLowerCase() === 'true';

// UI Configuration
export const MAX_FILE_SIZE_MB = parseInt(import.meta.env.VITE_MAX_FILE_SIZE || '10', 10);
export const DEFAULT_THEME = import.meta.env.VITE_DEFAULT_THEME || 'system';

// Logging utility that only logs in debug mode
export const debugLog = (...args: any[]) => {
  if (ENABLE_DEBUG_MODE) {
    console.log('[DEBUG]', ...args);
  }
};

// Export configuration as a single object
export const config = {
  API_URL,
  APP_TITLE,
  APP_VERSION,
  ENABLE_ADVANCED_FEATURES,
  ENABLE_DEBUG_MODE,
  MAX_FILE_SIZE_MB,
  DEFAULT_THEME,
}; 