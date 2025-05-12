import React, { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import './LatexRenderer.css';

/**
 * LatexRenderer Component
 * Renders LaTeX mathematical expressions using MathJax or KaTeX
 * 
 * @param {Object} props - Component props
 * @param {string} props.latex - LaTeX expression to render
 * @param {boolean} props.isBlock - Whether to render as block (display math) or inline
 * @param {string} props.className - Additional CSS class
 * @param {boolean} props.renderImmediately - Whether to render immediately
 * @param {Object} ref - Forwarded ref
 */
const LatexRenderer = forwardRef(({
  latex = '',
  isBlock = false,
  className = '',
  renderImmediately = true
}, ref) => {
  const containerRef = useRef(null);
  
  // Effect to render LaTeX when the component mounts or when latex changes
  useEffect(() => {
    if (!latex || !containerRef.current) return;
    
    // Skip immediate rendering if not requested
    if (!renderImmediately) return;
    
    // Get the rendering engine from user preferences or default to MathJax
    const renderingEngine = localStorage.getItem('mathLlmEquationRenderer') || 'mathjax';
    
    if (renderingEngine === 'katex' && window.katex) {
      try {
        // Use KaTeX for rendering
        window.katex.render(latex, containerRef.current, {
          displayMode: isBlock,
          throwOnError: false,
          output: 'html'
        });
      } catch (error) {
        console.error('KaTeX rendering error:', error);
        containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
      }
    } else if (window.MathJax) {
      try {
        // Use MathJax for rendering
        // Clear previous content
        containerRef.current.innerHTML = '';
        
        // Create element with LaTeX
        const mathElement = document.createElement(isBlock ? 'div' : 'span');
        mathElement.textContent = isBlock ? `\\[${latex}\\]` : `\\(${latex}\\)`;
        containerRef.current.appendChild(mathElement);
        
        // Process the new element with MathJax
        window.MathJax.typesetPromise([containerRef.current]).catch((error) => {
          console.error('MathJax rendering error:', error);
          containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
        });
      } catch (error) {
        console.error('MathJax setup error:', error);
        containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
      }
    } else {
      // Fallback if neither KaTeX nor MathJax is available
      containerRef.current.innerHTML = `<code>${latex}</code>`;
      console.warn('Neither MathJax nor KaTeX is available for rendering LaTeX');
    }
  }, [latex, isBlock, renderImmediately]);
  
  // Manual render method for external triggering
  const render = () => {
    // Re-trigger the same effect logic
    if (!latex || !containerRef.current) return;
    
    // Same rendering logic as in the effect
    const renderingEngine = localStorage.getItem('mathLlmEquationRenderer') || 'mathjax';
    
    if (renderingEngine === 'katex' && window.katex) {
      try {
        window.katex.render(latex, containerRef.current, {
          displayMode: isBlock,
          throwOnError: false
        });
      } catch (error) {
        console.error('KaTeX rendering error:', error);
        containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
      }
    } else if (window.MathJax) {
      try {
        containerRef.current.innerHTML = '';
        
        const mathElement = document.createElement(isBlock ? 'div' : 'span');
        mathElement.textContent = isBlock ? `\\[${latex}\\]` : `\\(${latex}\\)`;
        containerRef.current.appendChild(mathElement);
        
        window.MathJax.typesetPromise([containerRef.current]).catch((error) => {
          console.error('MathJax rendering error:', error);
          containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
        });
      } catch (error) {
        console.error('MathJax setup error:', error);
        containerRef.current.textContent = `Error rendering LaTeX: ${latex}`;
      }
    } else {
      containerRef.current.innerHTML = `<code>${latex}</code>`;
      console.warn('Neither MathJax nor KaTeX is available for rendering LaTeX');
    }
  };
  
  // Expose the render method to parent components via ref
  useImperativeHandle(ref, () => ({
    render,
    getContainer: () => containerRef.current
  }));
  
  // Calculate CSS classes
  const containerClasses = `latex-renderer ${isBlock ? 'block-math' : 'inline-math'} ${className}`;
  
  // Render the component
  return (
    <div 
      ref={containerRef} 
      className={containerClasses}
      data-latex={latex}
      role="math"
      aria-label={`Math formula: ${latex}`}
    >
      {!renderImmediately && (
        <code className="latex-placeholder">{latex}</code>
      )}
    </div>
  );
});

export default LatexRenderer;
