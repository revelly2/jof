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
        res.status(response.status).json(result);
    } catch (err) {
        console.error("Roboflow proxy failed:", err);
        res.status(500).json({ error: err.message });
    }
};
