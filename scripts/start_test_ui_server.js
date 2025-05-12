/**
 * Simple HTTP server for testing the mathematical UI components
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.jsx': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  
  // Check if it's a file request
  let filePath = req.url;
  if (filePath === '/') {
    filePath = '/web/public/test-math-interface.html';
  }
  
  // Remove query parameters if any
  filePath = filePath.split('?')[0];
  
  // Resolve to absolute path
  let resolvedPath;
  if (filePath.startsWith('/web/')) {
    resolvedPath = path.join(__dirname, '..', filePath);
  } else if (filePath.startsWith('/scripts/')) {
    resolvedPath = path.join(__dirname, '..', filePath);
  } else {
    resolvedPath = path.join(__dirname, '..', 'web', 'public', filePath);
  }
  
  // Get file extension for content type
  const ext = path.extname(resolvedPath);
  const contentType = MIME_TYPES[ext] || 'text/plain';
  
  // Read file
  fs.readFile(resolvedPath, (err, data) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // 404 - File not found
        console.error(`File not found: ${resolvedPath}`);
        res.writeHead(404);
        res.end('File not found');
      } else {
        // 500 - Internal server error
        console.error(`Error reading file: ${err}`);
        res.writeHead(500);
        res.end('Internal server error');
      }
      return;
    }
    
    // Successful response
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Test UI available at http://localhost:${PORT}/web/public/test-math-interface.html`);
  console.log(`Press Ctrl+C to stop the server`);
});
