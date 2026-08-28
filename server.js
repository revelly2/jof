const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
    let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
    const ext = path.extname(filePath);
    let contentType = MIME_TYPES[ext] || 'application/octet-stream';
    
    // Intercept API routes for Roboflow proxy, local model training, and dataset scanning
    if (req.url === '/api/diagnose-roboflow') {
        handleRoboflowProxy(req, res);
        return;
    }
    
    if (req.url === '/api/dataset/list') {
        handleDatasetList(req, res);
        return;
    }
    
    if (req.url.startsWith('/api/train/')) {
        handleApiRoute(req, res);
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
});

// ==========================================
// BACKEND TRAINER PROCESS CONFIGURATION
// ==========================================
const { spawn } = require('child_process');

let trainingState = {
    status: 'idle', // 'idle', 'training', 'success', 'error'
    progress: 0,
    currentEpoch: 0,
    totalEpochs: 50,
    latestLog: '',
    error: null
};

let trainingProcess = null;

function processTrainingLogLine(line) {
    // Strip carriage returns, line feeds and ANSI color/escape sequences
    const cleanLine = line
        .replace(/[\r\n]/g, '')
        .replace(/[\u001b\u009b]\[[()#?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '')
        .trim();
        
    if (!cleanLine) return;
    
    trainingState.latestLog = cleanLine;
    
    // Parse progress. YOLO prints epoch ratio (e.g. 1/50) and iteration ratio (e.g. 197/329)
    const matches = cleanLine.match(/\b(\d+)\/(\d+)\b/g);
    
    let curEpoch = trainingState.currentEpoch;
    let totalEpochs = trainingState.totalEpochs;
    let innerPct = 0;
    
    if (matches && matches.length > 0) {
        // First match is epoch progress
        const epochParts = matches[0].split('/');
        curEpoch = parseInt(epochParts[0], 10);
        totalEpochs = parseInt(epochParts[1], 10);
        
        // Second match is iteration progress within the current epoch
        if (matches.length >= 2) {
            const iterParts = matches[1].split('/');
            const curIter = parseInt(iterParts[0], 10);
            const totalIter = parseInt(iterParts[1], 10);
            if (totalIter > 0) {
                innerPct = (curIter / totalIter) * 100;
            }
        } else {
            // Fallback: search for percentage symbol
            const pctMatch = cleanLine.match(/(\d+)%/);
            if (pctMatch) {
                innerPct = parseInt(pctMatch[1], 10);
            }
        }
        
        if (totalEpochs > 0 && curEpoch <= totalEpochs) {
            trainingState.currentEpoch = curEpoch;
            trainingState.totalEpochs = totalEpochs;
            
            // Overall training progress = previous completed epochs + fraction of current epoch
            const rawProgress = ((curEpoch - 1) / totalEpochs) * 100 + (innerPct / totalEpochs);
            trainingState.progress = Math.min(100, Math.max(0, Math.round(rawProgress)));
        }
    }
}

function handleApiRoute(req, res) {
    res.writeHead(200, { 
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*'
    });
    
    if (req.url === '/api/train/start' && req.method === 'POST') {
        if (trainingState.status === 'training') {
            res.end(JSON.stringify({ error: 'Training is already running' }));
            return;
        }

        // Reset state
        trainingState.status = 'training';
        trainingState.progress = 0;
        trainingState.currentEpoch = 0;
        trainingState.latestLog = 'Initializing local training environment...';
        trainingState.error = null;

        // Spawn python train.py using py launcher
        trainingProcess = spawn('py', ['train.py'], { cwd: __dirname });

        const handleLogData = (data) => {
            const lines = data.toString().split(/[\r\n]+/);
            lines.forEach(line => {
                processTrainingLogLine(line);
            });
        };

        trainingProcess.stdout.on('data', handleLogData);
        trainingProcess.stderr.on('data', handleLogData);

        trainingProcess.on('close', (code) => {
            if (code === 0) {
                trainingState.status = 'success';
                trainingState.progress = 100;
                trainingState.latestLog = 'Training completed successfully! Model exported to ONNX format.';
            } else {
                if (trainingState.status === 'training') {
                    trainingState.status = 'error';
                    trainingState.error = `Process exited with code ${code}`;
                    trainingState.latestLog = `Error: Training process terminated with exit code ${code}. Check if ultralytics is installed.`;
                }
            }
            trainingProcess = null;
        });

        trainingProcess.on('error', (err) => {
            trainingState.status = 'error';
            trainingState.error = err.message;
            trainingState.latestLog = `Error spawning training process: ${err.message}`;
            trainingProcess = null;
        });

        res.end(JSON.stringify({ status: 'started' }));
        return;
    }
    
    if (req.url === '/api/train/status' && req.method === 'GET') {
        res.end(JSON.stringify(trainingState));
        return;
    }
    
    if (req.url === '/api/train/stop' && (req.method === 'POST' || req.method === 'GET')) {
        if (trainingProcess) {
            trainingProcess.kill();
            trainingProcess = null;
        }
        trainingState.status = 'idle';
        trainingState.progress = 0;
        trainingState.currentEpoch = 0;
        trainingState.latestLog = 'Training reset.';
        trainingState.error = null;
        res.end(JSON.stringify({ status: 'stopped' }));
        return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Endpoint not found' }));
}

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

function listDatasetImages() {
    const datasetDir = path.join(__dirname, 'dataset');
    if (!fs.existsSync(datasetDir)) {
        return [];
    }
    
    const allFiles = [];
    const walk = (dir) => {
        if (!fs.existsSync(dir)) return;
        const list = fs.readdirSync(dir);
        for (const file of list) {
            const fullPath = path.join(dir, file);
            const stat = fs.statSync(fullPath);
            if (stat.isDirectory()) {
                walk(fullPath);
            } else {
                allFiles.push(fullPath);
            }
        }
    };
    
    walk(datasetDir);
    
    const imgExtensions = ['.bmp', '.jpg', '.jpeg', '.png'];
    const images = [];
    
    for (const filePath of allFiles) {
        const ext = path.extname(filePath).toLowerCase();
        if (imgExtensions.includes(ext)) {
            const relPath = path.relative(__dirname, filePath).replace(/\\/g, '/');
            const baseName = path.basename(filePath, ext);
            
            // Try to find matching label file in the corresponding labels directory
            const dirName = path.dirname(filePath);
            const parentDir = path.basename(dirName); // Should be 'images'
            let groundTruthClassId = null;
            
            if (parentDir === 'images') {
                const labelDir = path.join(path.dirname(dirName), 'labels');
                const labelPath = path.join(labelDir, baseName + '.txt');
                if (fs.existsSync(labelPath)) {
                    try {
                        const content = fs.readFileSync(labelPath, 'utf8').trim();
                        if (content) {
                            const firstLine = content.split('\n')[0].trim();
                            const classId = parseInt(firstLine.split(/\s+/)[0], 10);
                            if (!isNaN(classId)) {
                                groundTruthClassId = classId;
                            }
                        }
                    } catch (e) {
                        console.error(`Error reading label file ${labelPath}:`, e);
                    }
                }
            }
            
            images.push({
                path: relPath,
                name: baseName,
                groundTruthClassId: groundTruthClassId
            });
        }
    }
    
    return images;
}

function handleDatasetList(req, res) {
    res.writeHead(200, {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*'
    });
    try {
        const images = listDatasetImages();
        res.end(JSON.stringify({ images }));
    } catch (err) {
        console.error("Failed to list dataset:", err);
        res.end(JSON.stringify({ error: err.message }));
    }
}

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});
