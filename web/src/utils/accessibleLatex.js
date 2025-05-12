/**
 * Utility functions for creating accessible descriptions of LaTeX equations
 */

// Simple LaTeX to spoken description conversion
export const latexToSpeech = (latex) => {
  if (!latex) return 'Empty equation';
  
  let description = latex;
  
  // Common LaTeX commands with their spoken equivalents
  const replacements = [
    // Fractions
    { pattern: /\\frac\{([^}]*)\}\{([^}]*)\}/g, replacement: 'the fraction with numerator $1 and denominator $2' },
    
    // Powers
    { pattern: /([a-zA-Z0-9])(\^)(\{[^}]*\}|\^[a-zA-Z0-9])/g, replacement: '$1 raised to the power of $3' },
    { pattern: /\^(\{[^}]*\})/g, replacement: ' raised to the power of $1' },
    { pattern: /\^([a-zA-Z0-9])/g, replacement: ' raised to the power of $1' },
    
    // Subscripts
    { pattern: /([a-zA-Z0-9])(_)(\{[^}]*\}|_[a-zA-Z0-9])/g, replacement: '$1 subscript $3' },
    { pattern: /_(\{[^}]*\})/g, replacement: ' subscript $1' },
    { pattern: /_([a-zA-Z0-9])/g, replacement: ' subscript $1' },
    
    // Square root
    { pattern: /\\sqrt\{([^}]*)\}/g, replacement: 'square root of $1' },
    { pattern: /\\sqrt\[([^]]*)\]\{([^}]*)\}/g, replacement: '$1 root of $2' },
    
    // Greek letters
    { pattern: /\\alpha/g, replacement: 'alpha' },
    { pattern: /\\beta/g, replacement: 'beta' },
    { pattern: /\\gamma/g, replacement: 'gamma' },
    { pattern: /\\delta/g, replacement: 'delta' },
    { pattern: /\\epsilon/g, replacement: 'epsilon' },
    { pattern: /\\zeta/g, replacement: 'zeta' },
    { pattern: /\\eta/g, replacement: 'eta' },
    { pattern: /\\theta/g, replacement: 'theta' },
    { pattern: /\\iota/g, replacement: 'iota' },
    { pattern: /\\kappa/g, replacement: 'kappa' },
    { pattern: /\\lambda/g, replacement: 'lambda' },
    { pattern: /\\mu/g, replacement: 'mu' },
    { pattern: /\\nu/g, replacement: 'nu' },
    { pattern: /\\xi/g, replacement: 'xi' },
    { pattern: /\\pi/g, replacement: 'pi' },
    { pattern: /\\rho/g, replacement: 'rho' },
    { pattern: /\\sigma/g, replacement: 'sigma' },
    { pattern: /\\tau/g, replacement: 'tau' },
    { pattern: /\\upsilon/g, replacement: 'upsilon' },
    { pattern: /\\phi/g, replacement: 'phi' },
    { pattern: /\\chi/g, replacement: 'chi' },
    { pattern: /\\psi/g, replacement: 'psi' },
    { pattern: /\\omega/g, replacement: 'omega' },
    
    // Capital Greek letters
    { pattern: /\\Gamma/g, replacement: 'capital Gamma' },
    { pattern: /\\Delta/g, replacement: 'capital Delta' },
    { pattern: /\\Theta/g, replacement: 'capital Theta' },
    { pattern: /\\Lambda/g, replacement: 'capital Lambda' },
    { pattern: /\\Xi/g, replacement: 'capital Xi' },
    { pattern: /\\Pi/g, replacement: 'capital Pi' },
    { pattern: /\\Sigma/g, replacement: 'capital Sigma' },
    { pattern: /\\Upsilon/g, replacement: 'capital Upsilon' },
    { pattern: /\\Phi/g, replacement: 'capital Phi' },
    { pattern: /\\Psi/g, replacement: 'capital Psi' },
    { pattern: /\\Omega/g, replacement: 'capital Omega' },
    
    // Integrals
    { pattern: /\\int_\{([^}]*)\}\^\{([^}]*)\}/g, replacement: 'the integral from $1 to $2 of' },
    { pattern: /\\int/g, replacement: 'the integral of' },
    
    // Summation
    { pattern: /\\sum_\{([^}]*)\}\^\{([^}]*)\}/g, replacement: 'the sum from $1 to $2 of' },
    { pattern: /\\sum/g, replacement: 'the sum of' },
    
    // Product
    { pattern: /\\prod_\{([^}]*)\}\^\{([^}]*)\}/g, replacement: 'the product from $1 to $2 of' },
    { pattern: /\\prod/g, replacement: 'the product of' },
    
    // Limits
    { pattern: /\\lim_\{([^}]*)\}/g, replacement: 'the limit as $1 of' },
    { pattern: /\\lim/g, replacement: 'the limit of' },
    
    // Common functions
    { pattern: /\\sin/g, replacement: 'sine' },
    { pattern: /\\cos/g, replacement: 'cosine' },
    { pattern: /\\tan/g, replacement: 'tangent' },
    { pattern: /\\arcsin/g, replacement: 'arc sine' },
    { pattern: /\\arccos/g, replacement: 'arc cosine' },
    { pattern: /\\arctan/g, replacement: 'arc tangent' },
    { pattern: /\\sinh/g, replacement: 'hyperbolic sine' },
    { pattern: /\\cosh/g, replacement: 'hyperbolic cosine' },
    { pattern: /\\tanh/g, replacement: 'hyperbolic tangent' },
    { pattern: /\\log/g, replacement: 'logarithm' },
    { pattern: /\\ln/g, replacement: 'natural logarithm' },
    { pattern: /\\exp/g, replacement: 'exponential function' },
    
    // Brackets
    { pattern: /\\left\(/g, replacement: 'open parenthesis' },
    { pattern: /\\right\)/g, replacement: 'close parenthesis' },
    { pattern: /\\left\[/g, replacement: 'open bracket' },
    { pattern: /\\right\]/g, replacement: 'close bracket' },
    { pattern: /\\left\\{/g, replacement: 'open curly brace' },
    { pattern: /\\right\\}/g, replacement: 'close curly brace' },
    
    // Operators and relations
    { pattern: /\\times/g, replacement: 'times' },
    { pattern: /\\div/g, replacement: 'divided by' },
    { pattern: /\\cdot/g, replacement: 'dot' },
    { pattern: /\\leq/g, replacement: 'less than or equal to' },
    { pattern: /\\geq/g, replacement: 'greater than or equal to' },
    { pattern: /\\neq/g, replacement: 'not equal to' },
    { pattern: /\\approx/g, replacement: 'approximately equal to' },
    { pattern: /\\equiv/g, replacement: 'equivalent to' },
    { pattern: /\\in/g, replacement: 'element of' },
    { pattern: /\\subset/g, replacement: 'subset of' },
    { pattern: /\\subseteq/g, replacement: 'subset or equal to' },
    { pattern: /\\cup/g, replacement: 'union' },
    { pattern: /\\cap/g, replacement: 'intersection' },
    
    // Matrices
    { pattern: /\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}/gs, replacement: 'a matrix with elements $1' },
    { pattern: /\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}/gs, replacement: 'the expression $1' },
    { pattern: /\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}/gs, replacement: 'the determinant of matrix with elements $1' },
    
    // Clean up curly braces
    { pattern: /\{/g, replacement: '' },
    { pattern: /\}/g, replacement: '' },
    
    // Clean up backslashes
    { pattern: /\\/g, replacement: '' }
  ];
  
  // Apply all replacements
  for (const { pattern, replacement } of replacements) {
    description = description.replace(pattern, replacement);
  }
  
  // Clean up multiple spaces
  description = description.replace(/\s+/g, ' ').trim();
  
  return description;
};

/**
 * Convert LaTeX to MathML for better screen reader support
 * This is a simplified implementation, a real one would use a proper LaTeX to MathML converter
 */
export const latexToMathML = (latex) => {
  // Note: This is a placeholder function. In a real implementation,
  // you would use a library like MathJax or a custom parser to generate MathML.
  // For demonstration purposes, we return a simple MathML structure.
  
  return `<math xmlns="http://www.w3.org/1998/Math/MathML">
    <mrow>
      <mi>Placeholder MathML for LaTeX: ${latex.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</mi>
    </mrow>
  </math>`;
};

/**
 * Get ARIA attributes for a LaTeX expression
 */
export const getLatexAriaAttributes = (latex) => {
  const description = latexToSpeech(latex);
  
  return {
    'role': 'math',
    'aria-label': description
  };
};
