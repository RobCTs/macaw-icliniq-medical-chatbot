// Importing Express (web application framework, for Node.js). For web and mobile applications.
const express = require('express');
//utilities for working with file and directory paths.
const path = require('path');
const app = express();
// where server listens for requests
const port = 3000;

// Middleware to serve static files
app.use(express.static('public'));

// Route to serve the homepage; any file in the 'public' folder will be accessible from the web server.
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Route to serve the about page
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});