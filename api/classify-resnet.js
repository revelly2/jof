// Pre-trained non-YOLO comparison model: ResNet18 image classifier
// (Roboflow Universe palay-peste/rice-pest-classifierr v3 - names the pest in the photo, no bounding boxes)
const RESNET_MODEL_URL = 'https://serverless.roboflow.com/rice-pest-classifierr/3';
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

        console.log("Roboflow proxy: sending request to hosted ResNet18 pest classifier...");
        const response = await fetch(`${RESNET_MODEL_URL}?api_key=${ROBOFLOW_API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: base64Image
        });

        const result = await response.json();
        res.status(response.status).json(result);
    } catch (err) {
        console.error("ResNet18 proxy failed:", err);
        res.status(500).json({ error: err.message });
    }
};
