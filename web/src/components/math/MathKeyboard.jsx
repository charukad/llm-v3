/**
 * Math Keyboard Component
 * 
 * A specialized keyboard for mathematical input with common symbols,
 * operators, and structures used in LaTeX.
 */

import React from 'react';
import './MathKeyboard.css';

const MathKeyboard = ({ onInput, compact = false }) => {
  // Define keyboard layouts by category
  const basicSymbols = [
    { label: '+', value: '+' },
    { label: '-', value: '-' },
    { label: '×', value: '\\times ' },
    { label: '÷', value: '\\div ' },
    { label: '=', value: '=' },
    { label: '≠', value: '\\neq ' },
    { label: '<', value: '<' },
    { label: '>', value: '>' },
    { label: '≤', value: '\\leq ' },
    { label: '≥', value: '\\geq ' },
  ];

  const greekLetters = [
    { label: 'α', value: '\\alpha ' },
    { label: 'β', value: '\\beta ' },
    { label: 'γ', value: '\\gamma ' },
    { label: 'Γ', value: '\\Gamma ' },
    { label: 'δ', value: '\\delta ' },
    { label: 'Δ', value: '\\Delta ' },
    { label: 'π', value: '\\pi ' },
    { label: 'θ', value: '\\theta ' },
    { label: 'λ', value: '\\lambda ' },
    { label: 'μ', value: '\\mu ' },
    { label: 'σ', value: '\\sigma ' },
    { label: 'Σ', value: '\\Sigma ' },
  ];

  const structures = [
    { label: 'a^b', value: '^{}' },
    { label: 'a_b', value: '_{}' },
    { label: 'a/b', value: '\\frac{}{}' },
    { label: '√', value: '\\sqrt{}' },
    { label: '∛', value: '\\sqrt[3]{}' },
    { label: '∫', value: '\\int_{}}^{}' },
    { label: '∑', value: '\\sum_{}^{}' },
    { label: '∏', value: '\\prod_{}^{}' },
    { label: 'lim', value: '\\lim_{}' },
  ];

  const functions = [
    { label: 'sin', value: '\\sin ' },
    { label: 'cos', value: '\\cos ' },
    { label: 'tan', value: '\\tan ' },
    { label: 'ln', value: '\\ln ' },
    { label: 'log', value: '\\log ' },
    { label: 'exp', value: '\\exp ' },
    { label: 'max', value: '\\max ' },
    { label: 'min', value: '\\min ' },
  ];

  const matrices = [
    { label: '[ ]', value: '\\begin{bmatrix} & \\\\ & \\end{bmatrix}' },
    { label: '( )', value: '\\begin{pmatrix} & \\\\ & \\end{pmatrix}' },
    { label: '| |', value: '\\begin{vmatrix} & \\\\ & \\end{vmatrix}' },
  ];

  const brackets = [
    { label: '( )', value: '()' },
    { label: '[ ]', value: '[]' },
    { label: '{ }', value: '\\{\\}' },
    { label: '| |', value: '|.|' },
    { label: '⌈ ⌉', value: '\\lceil \\rceil' },
    { label: '⌊ ⌋', value: '\\lfloor \\rfloor' },
  ];

  // Define categories
  const categories = [
    { name: 'Basic', symbols: basicSymbols },
    { name: 'Greek', symbols: greekLetters },
    { name: 'Structures', symbols: structures },
    { name: 'Functions', symbols: functions },
    { name: 'Matrices', symbols: matrices },
    { name: 'Brackets', symbols: brackets },
  ];

  // Filter categories for compact mode
  const visibleCategories = compact ? categories.slice(0, 4) : categories;

  // Handle keyboard input
  const handleButtonClick = (value) => {
    if (onInput) {
      onInput(value);
    }
  };

  return (
    <div className={`math-keyboard ${compact ? 'compact' : ''}`}>
      <div className="keyboard-tabs">
        {visibleCategories.map((category, index) => (
          <div key={index} className="keyboard-section">
            <h4 className="section-title">{category.name}</h4>
            <div className="symbol-grid">
              {category.symbols.map((symbol, symbolIndex) => (
                <button
                  key={symbolIndex}
                  className="symbol-button"
                  onClick={() => handleButtonClick(symbol.value)}
                  title={symbol.value}
                  type="button"
                >
                  {symbol.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="keyboard-actions">
        <button 
          type="button" 
          className="keyboard-action-button"
          onClick={() => handleButtonClick('UNDO')}
        >
          Undo
        </button>
        <button 
          type="button" 
          className="keyboard-action-button"
          onClick={() => handleButtonClick('REDO')}
        >
          Redo
        </button>
        <button 
          type="button" 
          className="keyboard-action-button"
          onClick={() => handleButtonClick(' ')}
        >
          Space
        </button>
        <button 
          type="button" 
          className="keyboard-action-button wide"
          onClick={() => handleButtonClick('\n')}
        >
          Enter
        </button>
      </div>
    </div>
  );
};

export default MathKeyboard;
