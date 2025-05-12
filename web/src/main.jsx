import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/MathInputPanel.css';

// Create root element for React
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the App component
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
