/**
 * Test script for the Enhanced Math Editor component and its subcomponents
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MathJaxContext } from 'better-react-mathjax';
import EnhancedMathEditor from '../web/src/components/math/EnhancedMathEditor';

// Configuration for MathJax
const mathJaxConfig = {
  loader: { load: ['input/tex', 'output/svg'] },
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  svg: {
    fontCache: 'global'
  }
};

// Wrap the test component with MathJax provider
const TestComponent = () => {
  const handleChange = (latex) => {
    console.log('LaTeX changed:', latex);
  };

  const handleSubmit = (latex) => {
    console.log('Submitted LaTeX:', latex);
    alert(`Submitted LaTeX: ${latex}`);
  };

  return (
    <MathJaxContext config={mathJaxConfig}>
      <div style={{ padding: '20px' }}>
        <h1>Enhanced Math Editor Test</h1>
        <EnhancedMathEditor
          initialValue="\frac{x^2}{2} + C"
          onChange={handleChange}
          onSubmit={handleSubmit}
          accessibilityMode={true}
        />
      </div>
    </MathJaxContext>
  );
};

// Main test function
const runTest = () => {
  const container = document.createElement('div');
  document.body.appendChild(container);
  
  render(<TestComponent />, container);
  
  console.log('Test component rendered. Please interact with the Math Editor to test functionality.');
};

// Execute test
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', runTest);
} else {
  console.error('This script needs to run in a browser environment');
}
