// Trained Roboflow YOLOv5 detector for the 6 rice pest classes
// (Common Rice Pests (Philippines) v11 - mAP50 97.5%, precision 96.9%, recall 97.2%)
const ROBOFLOW_MODEL_URL = 'https://serverless.roboflow.com/common-rice-pests-philippines/11';
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "HyFt0Ki9VlE29y54x3W6";

module.exports = async function handler(req, res) {
    // Vercel Serverless Function entry point
    if (req.method !== 'POST') {
        res.status(405).json({ error: 'Method Not Allowed' });
        return;
    }

    try {
        // Vercel automatically parses JSON bodies
        const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
        const base64Image = body.image;
        
        if (!base64Image) {
            res.status(400).json({ error: 'Missing image parameter' });
            return;
        }

        console.log("Roboflow proxy: sending request to hosted YOLOv5 pest model...");
        const response = await fetch(`${ROBOFLOW_MODEL_URL}?api_key=${ROBOFLOW_API_KEY}&confidence=40&overlap=20`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: base64Image
        });

        const result = await response.json();
        res.status(response.status).json(result);
    } catch (err) {
        console.error("Roboflow proxy failed:", err);
        res.status(500).json({ error: err.message });
    }
};
