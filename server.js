const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;

// Trained Roboflow YOLOv5 detector for the 6 rice pest classes
// (Common Rice Pests (Philippines) v11 - mAP50 97.5%, precision 96.9%, recall 97.2%)
const ROBOFLOW_MODEL_URL = 'https://serverless.roboflow.com/common-rice-pests-philippines/11';
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "HyFt0Ki9VlE29y54x3W6";

// Pre-trained non-YOLO comparison model: ResNet18 image classifier (palay-peste/rice-pest-classifierr v3)
const RESNET_MODEL_URL = 'https://serverless.roboflow.com/rice-pest-classifierr/3';

const PROXY_ROUTES = {
    '/api/diagnose-roboflow': { label: 'hosted YOLOv5 pest model', url: () => `${ROBOFLOW_MODEL_URL}?api_key=${ROBOFLOW_API_KEY}&confidence=40&overlap=20` },
    '/api/classify-resnet': { label: 'hosted ResNet18 pest classifier', url: () => `${RESNET_MODEL_URL}?api_key=${ROBOFLOW_API_KEY}` }
};

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.onnx': 'application/octet-stream'
};

const handler = (req, res) => {
    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    const ext = path.extname(filePath);
    let contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    if (PROXY_ROUTES[req.url]) {
        handleRoboflowProxy(req, res, PROXY_ROUTES[req.url]);
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

async function handleRoboflowProxy(req, res, route) {
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

            console.log(`Roboflow proxy: sending request to ${route.label}...`);
            const response = await fetch(route.url(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: base64Image
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
