/**
 * Symbol Palette Component
 * 
 * A palette of mathematical symbols organized by category
 * for easy insertion into the math editor.
 */

import React, { useState } from 'react';
import './SymbolPalette.css';

const SymbolPalette = ({ onSymbolSelect, compact = false }) => {
  const [activeCategory, setActiveCategory] = useState('basic');
  
  // Define symbols categorized
  const symbolCategories = {
    basic: [
      { symbol: '+', latex: '+', description: 'plus' },
      { symbol: '-', latex: '-', description: 'minus' },
      { symbol: '×', latex: '\\times ', description: 'times' },
      { symbol: '÷', latex: '\\div ', description: 'divided by' },
      { symbol: '·', latex: '\\cdot ', description: 'dot' },
      { symbol: '=', latex: '=', description: 'equals' },
      { symbol: '≠', latex: '\\neq ', description: 'not equal to' },
      { symbol: '<', latex: '<', description: 'less than' },
      { symbol: '>', latex: '>', description: 'greater than' },
      { symbol: '≤', latex: '\\leq ', description: 'less than or equal to' },
      { symbol: '≥', latex: '\\geq ', description: 'greater than or equal to' },
      { symbol: '≈', latex: '\\approx ', description: 'approximately equal to' },
      { symbol: '∝', latex: '\\propto ', description: 'proportional to' },
      { symbol: '±', latex: '\\pm ', description: 'plus or minus' },
      { symbol: '∓', latex: '\\mp ', description: 'minus or plus' },
    ],
    
    greek: [
      { symbol: 'α', latex: '\\alpha ', description: 'alpha' },
      { symbol: 'β', latex: '\\beta ', description: 'beta' },
      { symbol: 'γ', latex: '\\gamma ', description: 'gamma' },
      { symbol: 'Γ', latex: '\\Gamma ', description: 'capital gamma' },
      { symbol: 'δ', latex: '\\delta ', description: 'delta' },
      { symbol: 'Δ', latex: '\\Delta ', description: 'capital delta' },
      { symbol: 'ε', latex: '\\epsilon ', description: 'epsilon' },
      { symbol: 'ζ', latex: '\\zeta ', description: 'zeta' },
      { symbol: 'η', latex: '\\eta ', description: 'eta' },
      { symbol: 'θ', latex: '\\theta ', description: 'theta' },
      { symbol: 'Θ', latex: '\\Theta ', description: 'capital theta' },
      { symbol: 'κ', latex: '\\kappa ', description: 'kappa' },
      { symbol: 'λ', latex: '\\lambda ', description: 'lambda' },
      { symbol: 'Λ', latex: '\\Lambda ', description: 'capital lambda' },
      { symbol: 'μ', latex: '\\mu ', description: 'mu' },
      { symbol: 'ν', latex: '\\nu ', description: 'nu' },
      { symbol: 'ξ', latex: '\\xi ', description: 'xi' },
      { symbol: 'Ξ', latex: '\\Xi ', description: 'capital xi' },
      { symbol: 'π', latex: '\\pi ', description: 'pi' },
      { symbol: 'Π', latex: '\\Pi ', description: 'capital pi' },
      { symbol: 'ρ', latex: '\\rho ', description: 'rho' },
      { symbol: 'σ', latex: '\\sigma ', description: 'sigma' },
      { symbol: 'Σ', latex: '\\Sigma ', description: 'capital sigma' },
      { symbol: 'τ', latex: '\\tau ', description: 'tau' },
      { symbol: 'φ', latex: '\\phi ', description: 'phi' },
      { symbol: 'Φ', latex: '\\Phi ', description: 'capital phi' },
      { symbol: 'χ', latex: '\\chi ', description: 'chi' },
      { symbol: 'ψ', latex: '\\psi ', description: 'psi' },
      { symbol: 'Ψ', latex: '\\Psi ', description: 'capital psi' },
      { symbol: 'ω', latex: '\\omega ', description: 'omega' },
      { symbol: 'Ω', latex: '\\Omega ', description: 'capital omega' },
    ],
    
    calculus: [
      { symbol: '∫', latex: '\\int ', description: 'integral' },
      { symbol: '∬', latex: '\\iint ', description: 'double integral' },
      { symbol: '∭', latex: '\\iiint ', description: 'triple integral' },
      { symbol: '∮', latex: '\\oint ', description: 'contour integral' },
      { symbol: '∂', latex: '\\partial ', description: 'partial derivative' },
      { symbol: '∇', latex: '\\nabla ', description: 'nabla' },
      { symbol: 'lim', latex: '\\lim_{} ', description: 'limit' },
      { symbol: '→', latex: '\\to ', description: 'to' },
      { symbol: '∞', latex: '\\infty ', description: 'infinity' },
      { symbol: 'dx', latex: '\\,dx', description: 'd x' },
      { symbol: '∫_a^b', latex: '\\int_{a}^{b} ', description: 'definite integral from a to b' },
      { symbol: 'dy/dx', latex: '\\frac{dy}{dx}', description: 'derivative of y with respect to x' },
    ],
    
    sets: [
      { symbol: '∈', latex: '\\in ', description: 'element of' },
      { symbol: '∉', latex: '\\notin ', description: 'not an element of' },
      { symbol: '∋', latex: '\\ni ', description: 'contains as an element' },
      { symbol: '⊂', latex: '\\subset ', description: 'subset of' },
      { symbol: '⊃', latex: '\\supset ', description: 'superset of' },
      { symbol: '⊆', latex: '\\subseteq ', description: 'subset or equal to' },
      { symbol: '⊇', latex: '\\supseteq ', description: 'superset or equal to' },
      { symbol: '∪', latex: '\\cup ', description: 'union' },
      { symbol: '∩', latex: '\\cap ', description: 'intersection' },
      { symbol: '∅', latex: '\\emptyset ', description: 'empty set' },
      { symbol: 'ℕ', latex: '\\mathbb{N} ', description: 'natural numbers' },
      { symbol: 'ℤ', latex: '\\mathbb{Z} ', description: 'integers' },
      { symbol: 'ℚ', latex: '\\mathbb{Q} ', description: 'rational numbers' },
      { symbol: 'ℝ', latex: '\\mathbb{R} ', description: 'real numbers' },
      { symbol: 'ℂ', latex: '\\mathbb{C} ', description: 'complex numbers' },
    ],
    
    logic: [
      { symbol: '∧', latex: '\\wedge ', description: 'logical and' },
      { symbol: '∨', latex: '\\vee ', description: 'logical or' },
      { symbol: '¬', latex: '\\neg ', description: 'logical not' },
      { symbol: '⇒', latex: '\\Rightarrow ', description: 'implies' },
      { symbol: '⇔', latex: '\\Leftrightarrow ', description: 'if and only if' },
      { symbol: '∀', latex: '\\forall ', description: 'for all' },
      { symbol: '∃', latex: '\\exists ', description: 'there exists' },
      { symbol: '∄', latex: '\\nexists ', description: 'there does not exist' },
      { symbol: '⊢', latex: '\\vdash ', description: 'proves' },
      { symbol: '⊨', latex: '\\models ', description: 'models' },
    ],
    
    arrows: [
      { symbol: '→', latex: '\\rightarrow ', description: 'right arrow' },
      { symbol: '←', latex: '\\leftarrow ', description: 'left arrow' },
      { symbol: '↑', latex: '\\uparrow ', description: 'up arrow' },
      { symbol: '↓', latex: '\\downarrow ', description: 'down arrow' },
      { symbol: '↔', latex: '\\leftrightarrow ', description: 'left right arrow' },
      { symbol: '⇒', latex: '\\Rightarrow ', description: 'right double arrow' },
      { symbol: '⇐', latex: '\\Leftarrow ', description: 'left double arrow' },
      { symbol: '⇑', latex: '\\Uparrow ', description: 'up double arrow' },
      { symbol: '⇓', latex: '\\Downarrow ', description: 'down double arrow' },
      { symbol: '⇔', latex: '\\Leftrightarrow ', description: 'left right double arrow' },
    ],
    
    misc: [
      { symbol: '∠', latex: '\\angle ', description: 'angle' },
      { symbol: '°', latex: '^{\\circ} ', description: 'degree' },
      { symbol: '⊥', latex: '\\perp ', description: 'perpendicular' },
      { symbol: '∥', latex: '\\parallel ', description: 'parallel' },
      { symbol: '≅', latex: '\\cong ', description: 'congruent to' },
      { symbol: '∼', latex: '\\sim ', description: 'similar to' },
      { symbol: '≈', latex: '\\approx ', description: 'approximately equal to' },
      { symbol: '…', latex: '\\ldots ', description: 'low dots' },
      { symbol: '⋯', latex: '\\cdots ', description: 'centered dots' },
      { symbol: '⋮', latex: '\\vdots ', description: 'vertical dots' },
      { symbol: '⋱', latex: '\\ddots ', description: 'diagonal dots' },
    ],
  };
  
  // Filter categories for compact mode
  const availableCategories = compact 
    ? ['basic', 'greek', 'calculus', 'sets'] 
    : Object.keys(symbolCategories);
  
  // Handle symbol selection
  const handleSymbolClick = (latex) => {
    if (onSymbolSelect) {
      onSymbolSelect(latex);
    }
  };
  
  return (
    <div className={`symbol-palette ${compact ? 'compact' : ''}`}>
      <div className="category-tabs">
        {availableCategories.map((category) => (
          <button
            key={category}
            className={`category-tab ${activeCategory === category ? 'active' : ''}`}
            onClick={() => setActiveCategory(category)}
            type="button"
            aria-label={`${category} symbols`}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>
      
      <div className="symbols-container">
        {symbolCategories[activeCategory].map((symbol, index) => (
          <button
            key={index}
            className="symbol-btn"
            onClick={() => handleSymbolClick(symbol.latex)}
            title={symbol.description}
            aria-label={symbol.description}
            type="button"
          >
            {symbol.symbol}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SymbolPalette;
