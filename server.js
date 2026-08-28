const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml'
};

const handler = (req, res) => {
    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    const ext = path.extname(filePath);
    let contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    if (req.url === '/api/diagnose-roboflow') {
        handleRoboflowProxy(req, res);
        return;
    }

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 
                    'Content-Type': 'text/html',
                    'Cache-Control': 'no-store, no-cache, must-revalidate'
                });
                res.end('<h1>404 Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`);
            }
        } else {
            res.writeHead(200, { 
                'Content-Type': contentType,
                'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            });
            res.end(content, 'utf-8');
        }
    });
};

async function handleRoboflowProxy(req, res) {
    if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Method Not Allowed' }));
        return;
    }

    let body = '';
    req.on('data', chunk => {
        body += chunk;
    });

    req.on('end', async () => {
        try {
            const parsed = JSON.parse(body);
            const base64Image = parsed.image; // raw base64 data
            if (!base64Image) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Missing image parameter' }));
                return;
            }

            console.log("Roboflow proxy: sending request to serverless workflows...");
            const response = await fetch('https://serverless.roboflow.com/markdaluson30-gmail-com/workflows/general-segmentation-api-2', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    api_key: "HyFt0Ki9VlE29y54x3W6",
                    inputs: {
                        image: {
                            type: "base64",
                            value: base64Image
                        },
                        classes: "solution"
                    }
                })
            });

            const result = await response.json();
            res.writeHead(response.status, { 
                'Content-Type': 'application/json',
                'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate'
            });
            res.end(JSON.stringify(result));
        } catch (err) {
            console.error("Roboflow proxy failed:", err);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: err.message }));
        }
    });
}

if (process.env.NODE_ENV !== 'production' && !process.env.VERCEL) {
    const server = http.createServer(handler);
    server.listen(PORT, () => {
        console.log(`Server running at http://localhost:${PORT}/`);
    });
}
module.exports = handler;
