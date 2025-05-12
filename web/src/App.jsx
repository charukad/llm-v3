import React, { useEffect, useState } from 'react';
import MathQueryPanel from './components/math/MathQueryPanel';
import IntegratedResponse from './components/response/IntegratedResponse';
import apiService from './services/apiService';
import preferencesService from './services/preferencesService';
import responseService from './services/responseService';
import './styles/MathInputPanel.css';

/**
 * Main Application Component
 * Integrates all components and services for the Mathematical Multimodal LLM system
 */
const App = () => {
  // State for the application
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(null);
  // Skip authentication check and set to true by default
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  
  // Load dark mode preference
  useEffect(() => {
    const isDarkMode = preferencesService.isDarkMode();
    setDarkMode(isDarkMode);
    
    // Apply dark mode class to body
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
    
    // Load MathJax if not already loaded
    if (!window.MathJax) {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
      script.async = true;
      document.head.appendChild(script);
    }
  }, []);
  
  // Handle query submission
  const handleQuerySubmit = (query) => {
    // Store the current query
    setCurrentQuery(query);
    
    // Clear previous response
    setResponse(null);
    setLoading(true);
    setError('');
  };
  
  // Handle query response
  const handleQueryResponse = (responseData) => {
    setLoading(false);
    
    // Format the response
    const formattedResponse = responseService.formatResponse({ data: responseData });
    setResponse(formattedResponse);
  };
  
  // Handle query error
  const handleQueryError = (errorMessage) => {
    setLoading(false);
    setError(errorMessage);
  };
  
  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    
    // Update preference
    preferencesService.setValue('display', 'darkMode', newDarkMode);
    
    // Apply dark mode class to body
    if (newDarkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  };
  
  // Handle login
  const handleLogin = async (credentials) => {
    try {
      const response = await apiService.auth.login(credentials);
      if (response.success) {
        setIsAuthenticated(true);
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  };
  
  // Handle logout
  const handleLogout = () => {
    apiService.auth.logout();
    setIsAuthenticated(false);
  };
  
  // Login form (simple implementation - would be replaced with a proper component)
  const renderLoginForm = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loginError, setLoginError] = useState('');
    
    const handleSubmit = async (e) => {
      e.preventDefault();
      const result = await handleLogin({ username, password });
      if (!result.success) {
        setLoginError(result.error);
      }
    };
    
    return (
      <div className="login-container">
        <h2>Login to Mathematical LLM System</h2>
        {loginError && <div className="login-error">{loginError}</div>}
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input 
              type="text" 
              id="username" 
              value={username} 
              onChange={(e) => setUsername(e.target.value)} 
              required 
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input 
              type="password" 
              id="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
            />
          </div>
          <button type="submit" className="login-button">Login</button>
        </form>
      </div>
    );
  };
  
  return (
    <div className={`app-container ${darkMode ? 'dark-mode' : ''}`}>
      {/* Header */}
      <header className="app-header">
        <h1>Mathematical Multimodal LLM System</h1>
        <div className="header-controls">
          <button 
            className="dark-mode-toggle"
            onClick={toggleDarkMode}
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
          </button>
          
          {isAuthenticated && (
            <button 
              className="logout-button"
              onClick={handleLogout}
              aria-label="Logout"
            >
              Logout
            </button>
          )}
        </div>
      </header>
      
      {/* Main content */}
      <main className="app-main">
        {!isAuthenticated ? (
          renderLoginForm()
        ) : (
          <>
            <section className="query-section">
              <h2>Mathematical Query</h2>
              <MathQueryPanel
                onSubmit={handleQuerySubmit}
                onResponse={handleQueryResponse}
                onError={handleQueryError}
                showPreferences={true}
              />
            </section>
            
            <section className="response-section">
              <h2>Response</h2>
              {currentQuery && (
                <div className="current-query">
                  <h3>Current Query:</h3>
                  <p>{currentQuery.query}</p>
                </div>
              )}
              <IntegratedResponse
                response={response}
                loading={loading}
                error={error}
              />
            </section>
          </>
        )}
      </main>
      
      {/* Footer */}
      <footer className="app-footer">
        <p>© 2025 Mathematical Multimodal LLM System</p>
      </footer>
    </div>
  );
};

export default App;
