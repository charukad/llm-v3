/**
 * Configuration utilities for LaTeX rendering engines (MathJax and KaTeX)
 */

/**
 * Initialize MathJax with optimal configuration for mathematical notation
 * 
 * @param {Object} options - Custom configuration options
 * @returns {Promise} - Promise that resolves when MathJax is loaded and configured
 */
export const initMathJax = (options = {}) => {
  return new Promise((resolve, reject) => {
    // Skip if MathJax is already loaded
    if (window.MathJax) {
      resolve(window.MathJax);
      return;
    }
    
    // Default configuration
    window.MathJax = {
      tex: {
        inlineMath: [['\\(', '\\)']],
        displayMath: [['$$', '$$']],
        processEscapes: true,
        processEnvironments: true,
        packages: ['base', 'ams', 'noerrors', 'noundefined', 'autoload', 'color'],
        ...options.tex
      },
      options: {
        enableMenu: false,
        renderActions: {
          addMenu: [], // Disable the MathJax menu
          checkLoading: []
        },
        ...options.options
      },
      startup: {
        pageReady: () => {
          resolve(window.MathJax);
        },
        ...options.startup
      },
      svg: {
        fontCache: 'global',
        ...options.svg
      }
    };
    
    // Load MathJax script
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
    script.async = true;
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

/**
 * Initialize KaTeX with optimal configuration for mathematical notation
 * 
 * @param {Object} options - Custom configuration options
 * @returns {Promise} - Promise that resolves when KaTeX is loaded
 */
export const initKaTeX = (options = {}) => {
  return new Promise((resolve, reject) => {
    // Skip if KaTeX is already loaded
    if (window.katex) {
      resolve(window.katex);
      return;
    }
    
    // Load KaTeX script
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js';
    script.integrity = 'sha384-cpW21h6RZv/phavutF+AuVYrL+9wFuDWWD1Oc/ghPGPs3kVK5+9VYfNbNLe5vJVQ'; 
    script.crossOrigin = 'anonymous';
    script.async = true;
    
    script.onload = () => {
      // Load KaTeX CSS
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css';
      link.integrity = 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntxDrHA7ycNz5ZgWWIr2MWLxiFjr0Pf+Tx'; 
      link.crossOrigin = 'anonymous';
      document.head.appendChild(link);
      
      // Load auto-render extension if requested
      if (options.autoRender) {
        const autoRenderScript = document.createElement('script');
        autoRenderScript.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js';
        autoRenderScript.integrity = 'sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/9aIgS'; 
        autoRenderScript.crossOrigin = 'anonymous';
        autoRenderScript.onload = () => {
          resolve(window.katex);
        };
        autoRenderScript.onerror = reject;
        document.head.appendChild(autoRenderScript);
      } else {
        resolve(window.katex);
      }
    };
    
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

/**
 * Initialize the preferred LaTeX rendering engine
 * 
 * @param {string} engine - Either 'mathjax' or 'katex'
 * @param {Object} options - Custom configuration options
 * @returns {Promise} - Promise that resolves when the engine is loaded
 */
export const initLatexEngine = (engine = 'mathjax', options = {}) => {
  if (engine === 'mathjax') {
    return initMathJax(options);
  } else if (engine === 'katex') {
    return initKaTeX(options);
  } else {
    return Promise.reject(new Error(`Unsupported LaTeX engine: ${engine}`));
  }
};

/**
 * Get the optimal LaTeX engine based on device capabilities
 * 
 * @returns {string} - The recommended engine ('mathjax' or 'katex')
 */
export const getOptimalLatexEngine = () => {
  // KaTeX is faster but has fewer features
  // MathJax is more comprehensive but slower
  
  // Simple heuristic: use KaTeX on mobile devices for better performance
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
  
  return isMobile ? 'katex' : 'mathjax';
};
