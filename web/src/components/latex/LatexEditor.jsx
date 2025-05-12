import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import LatexRenderer from './LatexRenderer';
import './LatexEditor.css';

/**
 * LaTeX Editor Component
 * 
 * A dual-pane editor that allows users to write LaTeX code and see
 * the rendered output in real-time, with helpful features like
 * autocompletion and common symbol insertion.
 */
const LatexEditor = ({
  initialValue = '',
  onChange = () => {},
  onSave = () => {},
  height = '300px',
  renderEngine = 'mathjax',
  displayMode = true,
  showSymbolToolbar = true,
  autoRender = true
}) => {
  const [latex, setLatex] = useState(initialValue);
  const [renderResult, setRenderResult] = useState(initialValue);
  const [error, setError] = useState(null);
  const editorRef = useRef(null);
  const symbolButtonsRef = useRef([]);

  // Update rendered output when LaTeX changes (if autoRender is true)
  useEffect(() => {
    if (autoRender) {
      const timer = setTimeout(() => {
        try {
          setRenderResult(latex);
          setError(null);
        } catch (err) {
          setError(err.message);
        }
      }, 500); // Debounce rendering for better performance
      
      return () => clearTimeout(timer);
    }
  }, [latex, autoRender]);

  // Handle LaTeX changes
  const handleChange = (e) => {
    const newValue = e.target.value;
    setLatex(newValue);
    onChange(newValue);
  };

  // Handle manual render button click
  const handleRender = () => {
    try {
      setRenderResult(latex);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  // Handle save button click
  const handleSave = () => {
    onSave(latex);
  };

  // Insert symbol at cursor position
  const insertSymbol = (symbol) => {
    if (!editorRef.current) return;
    
    const textarea = editorRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    
    // Insert symbol at cursor position
    const newValue = latex.substring(0, start) + symbol + latex.substring(end);
    
    // Update state
    setLatex(newValue);
    
    // Update cursor position after symbol insertion
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + symbol.length, start + symbol.length);
    }, 0);
    
    // Trigger onChange callback
    onChange(newValue);
  };

  // Common LaTeX symbols organized by category
  const symbolCategories = [
    {
      name: 'Greek Letters',
      symbols: [
        { symbol: '\\alpha', display: 'α' },
        { symbol: '\\beta', display: 'β' },
        { symbol: '\\gamma', display: 'γ' },
        { symbol: '\\delta', display: 'δ' },
        { symbol: '\\epsilon', display: 'ε' },
        { symbol: '\\zeta', display: 'ζ' },
        { symbol: '\\eta', display: 'η' },
        { symbol: '\\theta', display: 'θ' },
        { symbol: '\\lambda', display: 'λ' },
        { symbol: '\\mu', display: 'μ' },
        { symbol: '\\pi', display: 'π' },
        { symbol: '\\sigma', display: 'σ' },
        { symbol: '\\phi', display: 'φ' },
        { symbol: '\\omega', display: 'ω' }
      ]
    },
    {
      name: 'Operators',
      symbols: [
        { symbol: '+', display: '+' },
        { symbol: '-', display: '-' },
        { symbol: '\\cdot', display: '·' },
        { symbol: '\\times', display: '×' },
        { symbol: '\\div', display: '÷' },
        { symbol: '=', display: '=' },
        { symbol: '\\neq', display: '≠' },
        { symbol: '\\approx', display: '≈' },
        { symbol: '\\lt', display: '<' },
        { symbol: '\\gt', display: '>' },
        { symbol: '\\leq', display: '≤' },
        { symbol: '\\geq', display: '≥' }
      ]
    },
    {
      name: 'Calculus',
      symbols: [
        { symbol: '\\sum', display: 'Σ' },
        { symbol: '\\prod', display: 'Π' },
        { symbol: '\\int', display: '∫' },
        { symbol: '\\partial', display: '∂' },
        { symbol: '\\nabla', display: '∇' },
        { symbol: '\\oint', display: '∮' },
        { symbol: '\\lim', display: 'lim' },
        { symbol: '\\infty', display: '∞' },
        { symbol: '\\frac{}{} ', display: 'a/b' }
      ]
    },
    {
      name: 'Sets & Logic',
      symbols: [
        { symbol: '\\in', display: '∈' },
        { symbol: '\\subset', display: '⊂' },
        { symbol: '\\cup', display: '∪' },
        { symbol: '\\cap', display: '∩' },
        { symbol: '\\emptyset', display: '∅' },
        { symbol: '\\forall', display: '∀' },
        { symbol: '\\exists', display: '∃' },
        { symbol: '\\neg', display: '¬' },
        { symbol: '\\wedge', display: '∧' },
        { symbol: '\\vee', display: '∨' }
      ]
    },
    {
      name: 'Structures',
      symbols: [
        { symbol: '\\sqrt{} ', display: '√' },
        { symbol: '\\sqrt[n]{} ', display: 'ⁿ√' },
        { symbol: '^{}', display: 'x²' },
        { symbol: '_{}', display: 'x₂' },
        { symbol: '\\binom{}{} ', display: '(ⁿₖ)' },
        { symbol: '\\left( \\right)', display: '( )' },
        { symbol: '\\left[ \\right]', display: '[ ]' },
        { symbol: '\\left\\{ \\right\\}', display: '{ }' },
        { symbol: '\\begin{matrix} \\end{matrix}', display: 'mat' },
        { symbol: '\\begin{cases} \\end{cases}', display: '{ }' }
      ]
    }
  ];

  return (
    <div className="latex-editor" style={{ height }}>
      <div className="latex-editor-container">
        {/* Editor Pane */}
        <div className="editor-pane">
          <div className="pane-header">LaTeX Code</div>
          <textarea
            ref={editorRef}
            className="latex-textarea"
            value={latex}
            onChange={handleChange}
            placeholder="Enter LaTeX code here..."
            spellCheck="false"
          />
          
          {/* Bottom Toolbar */}
          <div className="editor-toolbar">
            {!autoRender && (
              <button 
                className="render-button" 
                onClick={handleRender}
                title="Render LaTeX"
              >
                Render
              </button>
            )}
            <button 
              className="save-button" 
              onClick={handleSave}
              title="Save LaTeX"
            >
              Save
            </button>
          </div>
        </div>
        
        {/* Preview Pane */}
        <div className="preview-pane">
          <div className="pane-header">Preview</div>
          <div className="latex-preview">
            {error ? (
              <div className="render-error">Error: {error}</div>
            ) : (
              <LatexRenderer
                latex={renderResult}
                displayMode={displayMode}
                engine={renderEngine}
              />
            )}
          </div>
        </div>
      </div>
      
      {/* Symbol Toolbar */}
      {showSymbolToolbar && (
        <div className="symbol-toolbar">
          {symbolCategories.map((category, catIndex) => (
            <div key={catIndex} className="symbol-category">
              <div className="category-name">{category.name}</div>
              <div className="symbol-buttons">
                {category.symbols.map((symbolObj, symIndex) => (
                  <button
                    key={symIndex}
                    ref={el => {
                      if (el) symbolButtonsRef.current[catIndex * 100 + symIndex] = el;
                    }}
                    className="symbol-button"
                    onClick={() => insertSymbol(symbolObj.symbol)}
                    title={symbolObj.symbol}
                  >
                    {symbolObj.display}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

LatexEditor.propTypes = {
  initialValue: PropTypes.string,
  onChange: PropTypes.func,
  onSave: PropTypes.func,
  height: PropTypes.string,
  renderEngine: PropTypes.oneOf(['mathjax', 'katex']),
  displayMode: PropTypes.bool,
  showSymbolToolbar: PropTypes.bool,
  autoRender: PropTypes.bool
};

export default LatexEditor;
