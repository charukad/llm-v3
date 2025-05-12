import React from 'react';
import PropTypes from 'prop-types';
import './LatexDocument.css';

/**
 * LaTeX Document Component
 * 
 * A component that renders a complete mathematical document with proper
 * LaTeX formatting, optimized for printing and PDF export.
 */
const LatexDocument = ({
  title,
  author,
  date = new Date().toLocaleDateString(),
  abstract = '',
  sections = [],
  bibliography = [],
  documentClass = 'article',
  fontSize = '11pt',
  paperSize = 'a4paper',
  margin = '1in',
  displayMode = false,
  onExport
}) => {
  // Generate LaTeX preamble
  const generatePreamble = () => {
    return `\\documentclass[${fontSize},${paperSize}]{${documentClass}}

% Packages
\\usepackage{amsmath,amssymb,amsfonts}
\\usepackage{mathtools}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{booktabs}
\\usepackage{geometry}

% Document settings
\\geometry{margin=${margin}}
\\title{${title}}
\\author{${author}}
\\date{${date}}

\\begin{document}

\\maketitle

${abstract ? `\\begin{abstract}\n${abstract}\n\\end{abstract}\n` : ''}`;
  };

  // Generate LaTeX for sections
  const generateSections = () => {
    let content = '';
    
    for (const section of sections) {
      if (section.type === 'section') {
        content += `\\section{${section.title}}\n${section.content}\n\n`;
      } else if (section.type === 'subsection') {
        content += `\\subsection{${section.title}}\n${section.content}\n\n`;
      } else if (section.type === 'equation') {
        content += `\\begin{${section.numbered ? 'equation' : 'equation*'}}\n${section.content}\n\\end{${section.numbered ? 'equation' : 'equation*'}}\n\n`;
      } else if (section.type === 'figure') {
        content += `\\begin{figure}[htbp]
\\centering
\\includegraphics[width=${section.width || '0.7\\textwidth'}]{${section.path}}
\\caption{${section.caption}}
\\label{fig:${section.label || 'figure'}}
\\end{figure}\n\n`;
      } else if (section.type === 'table') {
        content += `\\begin{table}[htbp]
\\centering
\\caption{${section.caption}}
\\label{tab:${section.label || 'table'}}
\\begin{tabular}{${section.alignment || 'c'.repeat(section.data[0]?.length || 1)}}
\\toprule
${section.data.map(row => row.join(' & ')).join(' \\\\\n')}
\\bottomrule
\\end{tabular}
\\end{table}\n\n`;
      } else if (section.type === 'list') {
        content += `\\begin{${section.ordered ? 'enumerate' : 'itemize'}}\n`;
        for (const item of section.items) {
          content += `\\item ${item}\n`;
        }
        content += `\\end{${section.ordered ? 'enumerate' : 'itemize'}}\n\n`;
      } else {
        // Default to paragraph
        content += `${section.content}\n\n`;
      }
    }
    
    return content;
  };

  // Generate LaTeX for bibliography
  const generateBibliography = () => {
    if (!bibliography || bibliography.length === 0) return '';
    
    let content = '\\begin{thebibliography}{99}\n';
    
    for (const [index, reference] of bibliography.entries()) {
      content += `\\bibitem{ref${index + 1}} ${reference}\n`;
    }
    
    content += '\\end{thebibliography}\n';
    
    return content;
  };

  // Generate complete LaTeX document
  const generateFullDocument = () => {
    return `${generatePreamble()}
${generateSections()}
${generateBibliography()}
\\end{document}`;
  };

  // Handle export button click
  const handleExport = () => {
    const fullDocument = generateFullDocument();
    
    if (onExport) {
      onExport(fullDocument);
    } else {
      // Default export behavior: download as .tex file
      const blob = new Blob([fullDocument], { type: 'application/x-tex' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title.replace(/\s+/g, '_').toLowerCase()}.tex`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className={`latex-document ${displayMode ? 'display-mode' : ''}`}>
      <div className="document-header">
        <h1 className="document-title">{title}</h1>
        <div className="document-meta">
          <span className="document-author">{author}</span>
          <span className="document-date">{date}</span>
        </div>
        {abstract && (
          <div className="document-abstract">
            <h2>Abstract</h2>
            <p>{abstract}</p>
          </div>
        )}
      </div>
      
      <div className="document-content">
        {sections.map((section, index) => (
          <div 
            key={index} 
            className={`document-section ${section.type}`}
            id={`section-${index}`}
          >
            {section.type === 'section' && (
              <h2 className="section-title">{section.title}</h2>
            )}
            
            {section.type === 'subsection' && (
              <h3 className="subsection-title">{section.title}</h3>
            )}
            
            {section.type === 'equation' && (
              <div className="equation-container">
                <div className="equation-content">
                  {section.content}
                </div>
                {section.numbered && (
                  <div className="equation-number">({index + 1})</div>
                )}
              </div>
            )}
            
            {section.type === 'figure' && (
              <figure className="figure-container">
                <img 
                  src={section.src} 
                  alt={section.caption}
                  style={{ maxWidth: section.width || '70%' }}
                />
                <figcaption>
                  <span className="figure-label">Figure {index + 1}:</span> {section.caption}
                </figcaption>
              </figure>
            )}
            
            {section.type === 'table' && (
              <div className="table-container">
                <table>
                  <caption>
                    <span className="table-label">Table {index + 1}:</span> {section.caption}
                  </caption>
                  <tbody>
                    {section.data.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex}>{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            
            {section.type === 'list' && (
              <div className="list-container">
                {section.ordered ? (
                  <ol>
                    {section.items.map((item, itemIndex) => (
                      <li key={itemIndex}>{item}</li>
                    ))}
                  </ol>
                ) : (
                  <ul>
                    {section.items.map((item, itemIndex) => (
                      <li key={itemIndex}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            
            {(section.type !== 'section' && 
              section.type !== 'subsection' && 
              section.type !== 'equation' &&
              section.type !== 'figure' &&
              section.type !== 'table' &&
              section.type !== 'list') && (
              <div className="paragraph-content">
                {section.content}
              </div>
            )}
          </div>
        ))}
      </div>
      
      {bibliography.length > 0 && (
        <div className="document-bibliography">
          <h2>References</h2>
          <ol className="bibliography-list">
            {bibliography.map((reference, index) => (
              <li key={index} className="bibliography-item" id={`ref-${index + 1}`}>
                {reference}
              </li>
            ))}
          </ol>
        </div>
      )}
      
      <div className="document-footer">
        <button className="export-button" onClick={handleExport}>
          Export as LaTeX
        </button>
      </div>
    </div>
  );
};

LatexDocument.propTypes = {
  title: PropTypes.string.isRequired,
  author: PropTypes.string.isRequired,
  date: PropTypes.string,
  abstract: PropTypes.string,
  sections: PropTypes.arrayOf(
    PropTypes.shape({
      type: PropTypes.oneOf([
        'section', 
        'subsection', 
        'paragraph', 
        'equation', 
        'figure', 
        'table', 
        'list'
      ]).isRequired,
      title: PropTypes.string,
      content: PropTypes.string,
      numbered: PropTypes.bool,
      src: PropTypes.string,
      caption: PropTypes.string,
      width: PropTypes.string,
      alignment: PropTypes.string,
      data: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.string)),
      items: PropTypes.arrayOf(PropTypes.string),
      ordered: PropTypes.bool
    })
  ).isRequired,
  bibliography: PropTypes.arrayOf(PropTypes.string),
  documentClass: PropTypes.oneOf(['article', 'report', 'book', 'letter']),
  fontSize: PropTypes.oneOf(['10pt', '11pt', '12pt']),
  paperSize: PropTypes.oneOf(['a4paper', 'letterpaper']),
  margin: PropTypes.string,
  displayMode: PropTypes.bool,
  onExport: PropTypes.func
};

export default LatexDocument;
