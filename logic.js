/* ==========================================================================
   RICEGUARD AI - LAYER 2: LOGIC LAYER (logic.js)
   Heuristic Classifier, Dosing Math, and Weather Advisor Logic
   ========================================================================== */

import { PEST_DATASET } from './data.js';

/**
 * Runs client-side pixel signature classification against the dataset color profiles.
 * Implements saturation thresholding to skip background mud, soil, and reflections.
 * @param {Uint8ClampedArray} pixelData - Canvas image raw pixel values
 * @returns {Object} Detected pest ID and diagnosis confidence percentage
 */
export function classifyImageColorSignature(pixelData, width = 600, height = 400, engineMode = 'ensemble') {
    let rgbMatches = {};
    let contrastMatches = {};
    let spatialCoordinates = {}; // stores { key: [{x, y}, ...] }
    
    // Initialize collections
    Object.keys(PEST_DATASET).forEach(key => {
        rgbMatches[key] = 0;
        contrastMatches[key] = 0;
        spatialCoordinates[key] = [];
    });

    // Subsample pixels for fast client-side performance (step by 80 pixels)
    // 80 pixels step is ~3000-5000 checks per image, extremely fast and statistically robust
    for (let i = 0; i < pixelData.length; i += 80) {
        const r = pixelData[i];
        const g = pixelData[i + 1];
        const b = pixelData[i + 2];
        const pixelIdx = i / 4;
        const x = pixelIdx % width;
        const y = Math.floor(pixelIdx / width);

        // Saturation Thresholding: Skip desaturated background noise (gray mud/water/sky reflections)
        const maxVal = Math.max(r, g, b);
        const minVal = Math.min(r, g, b);
        const saturation = maxVal - minVal;
        const intensity = (r + g + b) / 3;
        
        if (saturation < 25 || intensity < 30) {
            // Permit Leaf Folder color profiles (which are papery white / desaturated) through the saturation filter
            const isLeafFolder = Math.sqrt(Math.pow(r - 215, 2) + Math.pow(g - 225, 2) + Math.pow(b - 205, 2)) < 22;
            if (!isLeafFolder) {
                continue; 
            }
        }

        // 1. RGB Engine: Find the closest matching class profile (exclusive nearest neighbor)
        let bestRGBKey = null;
        let minRGBDist = Infinity;
        
        // 2. Contrast/Chromaticity Engine: Find the closest matching class profile
        let bestContrastKey = null;
        let minContrastDist = Infinity;

        Object.keys(PEST_DATASET).forEach(key => {
            const pest = PEST_DATASET[key];
            const target = pest.colorProfile;

            // Compute RGB Euclidean distance
            const distRGB = Math.sqrt(
                Math.pow(r - target.r, 2) +
                Math.pow(g - target.g, 2) +
                Math.pow(b - target.b, 2)
            );
            if (distRGB < target.tolerance && distRGB < minRGBDist) {
                minRGBDist = distRGB;
                bestRGBKey = key;
            }

            // Compute Chromaticity coordinates and distance
            const sumP = r + g + b;
            const sumT = target.r + target.g + target.b;
            if (sumP > 0 && sumT > 0) {
                const pr = r / sumP;
                const pg = g / sumP;
                const pb = b / sumP;

                const tr = target.r / sumT;
                const tg = target.g / sumT;
                const tb = target.b / sumT;

                const distChromatic = Math.sqrt(
                    Math.pow(pr - tr, 2) +
                    Math.pow(pg - tg, 2) +
                    Math.pow(pb - tb, 2)
                );
                
                if (distChromatic < 0.075 && distChromatic < minContrastDist) {
                    minContrastDist = distChromatic;
                    bestContrastKey = key;
                }
            }
        });

        // Assign to the single closest matched profiles exclusively
        if (bestRGBKey) {
            rgbMatches[bestRGBKey]++;
        }
        if (bestContrastKey) {
            contrastMatches[bestContrastKey]++;
            spatialCoordinates[bestContrastKey].push({ x, y });
        }
    }

    // 3. Spatial Morphology Engine
    let spatialScores = {};
    Object.keys(PEST_DATASET).forEach(key => {
        const coords = spatialCoordinates[key];
        const N = coords.length;
        
        if (N < 8) {
            spatialScores[key] = 0;
            return;
        }

        // Compute mean centroids
        let sumX = 0, sumY = 0;
        coords.forEach(pt => {
            sumX += pt.x;
            sumY += pt.y;
        });
        const meanX = sumX / N;
        const meanY = sumY / N;
        const cx = meanX / width;
        const cy = meanY / height;

        // Compute standard deviation (spread)
        let sumX2 = 0, sumY2 = 0;
        coords.forEach(pt => {
            sumX2 += Math.pow(pt.x - meanX, 2);
            sumY2 += Math.pow(pt.y - meanY, 2);
        });
        const sx = Math.sqrt(sumX2 / N) / width;
        const sy = Math.sqrt(sumY2 / N) / height;

        // Score based on pest template guidelines
        let score = 0;
        if (key === 'brown_planthopper') {
            // Clusters at the bottom/base of stems
            const locationScore = Math.max(0, Math.min(1, (cy - 0.45) / 0.35));
            const clusterScore = Math.max(0, 1 - (sx + sy) / 0.4);
            score = (locationScore * 0.6) + (clusterScore * 0.4);
        } else if (key === 'green_leafhopper') {
            // Scattered on leaves, upper part
            const locationScore = Math.max(0, Math.min(1, (0.75 - cy) / 0.5));
            const spreadScore = Math.min(1, (sx + sy) / 0.35);
            score = (locationScore * 0.5) + (spreadScore * 0.5);
        } else if (key === 'leaf_folder') {
            // Long folded narrow streaks
            const ratio = sy / (sx + 1e-5);
            const streakScore = Math.min(1, Math.max(0, (ratio - 1.2) / 2.0));
            score = streakScore;
        } else if (key === 'rice_bug') {
            // On grain panicles, middle-upper height
            const locationScore = Math.max(0, Math.min(1, (cy - 0.15) / 0.5));
            const clusterScore = Math.max(0, 1 - (sx + sy) / 0.3);
            score = (locationScore * 0.5) + (clusterScore * 0.5);
        } else if (key === 'stem_borer') {
            // Highly localized decay/bore hole
            const tightScore = Math.max(0, 1 - (sx + sy) / 0.22);
            score = tightScore;
        } else if (key === 'whorl_maggot') {
            // Margin stripes, linear streaks
            const ratio = sy / (sx + 1e-5);
            const streakScore = Math.min(1, Math.max(0, (ratio - 1.0) / 2.0));
            score = streakScore;
        }

        spatialScores[key] = Math.max(0.1, Math.min(1.0, score));
    });

    // Normalize RGB and Contrast counts into 0-1 scores
    let rgbScores = {};
    let contrastScores = {};
    
    let sumRGB = Object.values(rgbMatches).reduce((a, b) => a + b, 0);
    let sumContrast = Object.values(contrastMatches).reduce((a, b) => a + b, 0);

    Object.keys(PEST_DATASET).forEach(key => {
        rgbScores[key] = sumRGB > 0 ? rgbMatches[key] / sumRGB : 0;
        contrastScores[key] = sumContrast > 0 ? contrastMatches[key] / sumContrast : 0;
    });

    // 4. Ensemble Blended Engine
    let ensembleScores = {};
    Object.keys(PEST_DATASET).forEach(key => {
        // Blending model: 35% RGB, 35% Chromatic Contrast, 30% Spatial Morphology
        // Only give a spatial score if there is a chromatic connection (contrastMatches > 0)
        const hasColorConnection = contrastMatches[key] > 2;
        const currentSpatialScore = hasColorConnection ? spatialScores[key] : 0;
        
        ensembleScores[key] = (rgbScores[key] * 0.35) + (contrastScores[key] * 0.35) + (currentSpatialScore * 0.30);
    });

    // Select winner based on active engine
    let finalScores = {};
    if (engineMode === 'rgb') finalScores = rgbScores;
    else if (engineMode === 'contrast') finalScores = contrastScores;
    else if (engineMode === 'spatial') finalScores = spatialScores;
    else finalScores = ensembleScores; // default: ensemble

    let detectedPestKey = 'brown_planthopper';
    let maxScore = -1;
    Object.keys(finalScores).forEach(key => {
        if (finalScores[key] > maxScore) {
            maxScore = finalScores[key];
            detectedPestKey = key;
        }
    });

    // Calculate dynamic confidence score (scaled to 65% - 98% range)
    let confidenceScore = 70;
    if (maxScore > 0) {
        if (engineMode === 'ensemble') {
            confidenceScore = Math.round(maxScore * 100);
            confidenceScore = Math.max(68, Math.min(98, confidenceScore));
        } else {
            // For simple engines, base relative to sum
            const sumScores = Object.values(finalScores).reduce((a, b) => a + b, 0);
            confidenceScore = sumScores > 0 ? Math.round((maxScore / sumScores) * 100) : 70;
            confidenceScore = Math.max(65, Math.min(95, confidenceScore));
        }
    }

    return {
        detectedPestId: detectedPestKey,
        confidence: confidenceScore,
        pestMatches: rgbMatches, // Backwards compatibility with drawing highlight circles
        breakdown: {
            engineUsed: engineMode,
            rgbScores,
            contrastScores,
            spatialScores,
            ensembleScores
        }
    };
}

/**
 * Calculates chemical spray requirement and tank dilution instructions.
 * Aligns with official IRRI & PhilRice Hectare/Knapsack spray rates.
 * @param {number} farmArea - Total land area in square meters
 * @param {number} infectedArea - Target spray area in square meters
 * @param {Object} pest - Targeted pest record from dataset
 * @returns {Object} Calculated metrics
 */
export function calculateDosing(farmArea, infectedArea, pest) {
    // 1 Hectare = 10,000 square meters
    const infectedHectares = infectedArea / 10000;
    
    const totalPesticide = infectedHectares * pest.dosagePerHa; // ml or grams
    const totalWater = infectedHectares * pest.waterPerHa;     // Liters
    const knapsackTanks = totalWater / 16;                     // 16L tanks
    
    let chemicalPerTank = 0;
    if (knapsackTanks > 0) {
        chemicalPerTank = totalPesticide / knapsackTanks;
    }

    const dilutionRatio = totalWater > 0 ? (totalPesticide / totalWater) : 0; // units/Liter

    return {
        farmAreaHectares: farmArea / 10000,
        infectedHectares: infectedHectares,
        totalPesticide: parseFloat(totalPesticide.toFixed(1)),
        totalWater: parseFloat(totalWater.toFixed(1)),
        knapsackTanks: parseFloat(knapsackTanks.toFixed(1)),
        chemicalPerTank: parseFloat(chemicalPerTank.toFixed(1)),
        dilutionRatio: parseFloat(dilutionRatio.toFixed(2)),
        unit: (pest.id === 'stem_borer' || pest.id === 'whorl_maggot') ? 'g' : 'ml'
    };
}

/**
 * Evaluates spraying safety threshold based on wind, rain and temperature.
 * Aligns with PhilRice safe weather recommendations.
 * @param {number} wind - Wind speed in km/h
 * @param {number} rain - Rain probability in percentage
 * @param {number} temp - Temperature in degrees Celsius
 * @returns {Object} Status tag and detailed alert warning description
 */
export function assessWeatherSafety(wind, rain, temp) {
    let status = 'safe';
    let msg = 'Safe to Spray';
    let headerMsg = 'Weather: Safe to Spray';

    if (wind > 10 || rain > 40 || temp > 35) {
        status = 'caution';
        headerMsg = 'Weather: Check Spray Advisories';
        
        if (wind > 10) msg = 'Caution: Wind Drift Risk';
        else if (rain > 40) msg = 'Caution: Rain Washoff Risk';
        else if (temp > 35) msg = 'Caution: Evaporation / Stress Risk';
    }
    
    if (wind > 15 || rain > 70) {
        status = 'danger';
        headerMsg = 'Weather: Critical Spray Danger';
        
        if (wind > 15) msg = 'Warning: High Spray Drift Hazard';
        else if (rain > 70) msg = 'Warning: High Washoff Danger';
    }

    return { status, msg, headerMsg };
}

/* ==========================================
   YOLOv8 DEEP LEARNING UTILITIES
   ========================================== */

/**
 * Prepares canvas image data for YOLOv8 model inference.
 * Resizes the image to 640x640 and formats pixel channels to planar Float32Array [1, 3, 640, 640].
 * @param {HTMLCanvasElement} canvas - Original input image canvas
 * @returns {Float32Array} Normalized input tensor data
 */
export function preprocessCanvasForYolo(canvas) {
    const size = 640;
    const offscreen = document.createElement('canvas');
    offscreen.width = size;
    offscreen.height = size;
    const ctx = offscreen.getContext('2d');
    ctx.drawImage(canvas, 0, 0, size, size);
    
    const imgData = ctx.getImageData(0, 0, size, size);
    const data = imgData.data;
    
    // planar float array: CHW format (Channel, Height, Width)
    const floatArray = new Float32Array(3 * size * size);
    
    for (let i = 0; i < size * size; i++) {
        floatArray[i] = data[i * 4] / 255.0;               // Red Channel
        floatArray[size * size + i] = data[i * 4 + 1] / 255.0; // Green Channel
        floatArray[2 * size * size + i] = data[i * 4 + 2] / 255.0; // Blue Channel
    }
    
    return floatArray;
}

/**
 * Parses raw YOLOv8 ONNX model output data and filters predictions via NMS.
 * @param {Float32Array} outputData - Raw output tensor data [1, 4+C, 8400]
 * @param {number} imgWidth - Target original image width for scaling
 * @param {number} imgHeight - Target original image height for scaling
 * @param {number} confThreshold - Minimum confidence score filter
 * @param {number} numClasses - Number of classes (5 in our dataset)
 * @returns {Array} List of suppressed bounding box detections
 */
export function postprocessYoloOutput(outputData, imgWidth, imgHeight, confThreshold = 0.4, numClasses = 6) {
    const numAnchors = 8400;
    const boxes = [];
    
    // Map class indices to dataset IDs
    const classNames = ['brown_planthopper', 'green_leafhopper', 'leaf_folder', 'rice_bug', 'stem_borer', 'whorl_maggot'];
    
    for (let col = 0; col < numAnchors; col++) {
        let maxScore = -1;
        let classId = -1;
        
        for (let cl = 0; cl < numClasses; cl++) {
            const score = outputData[(4 + cl) * numAnchors + col];
            if (score > maxScore) {
                maxScore = score;
                classId = cl;
            }
        }
        
        if (maxScore > confThreshold) {
            const cx = outputData[0 * numAnchors + col];
            const cy = outputData[1 * numAnchors + col];
            const w = outputData[2 * numAnchors + col];
            const h = outputData[3 * numAnchors + col];
            
            // Re-scale coordinates from 640x640 grid back to original crop aspect ratio
            const x1 = (cx - w / 2) * (imgWidth / 640);
            const y1 = (cy - h / 2) * (imgHeight / 640);
            const boxW = w * (imgWidth / 640);
            const boxH = h * (imgHeight / 640);
            
            boxes.push({
                x: Math.max(0, x1),
                y: Math.max(0, y1),
                w: boxW,
                h: boxH,
                score: parseFloat(maxScore.toFixed(3)),
                classId: classId,
                pestId: classNames[classId]
            });
        }
    }
    
    // Apply Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes
    return applyNMS(boxes, 0.45);
}

function applyNMS(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];
    const active = new Array(boxes.length).fill(true);
    
    for (let i = 0; i < boxes.length; i++) {
        if (!active[i]) continue;
        
        selected.push(boxes[i]);
        
        for (let j = i + 1; j < boxes.length; j++) {
            if (!active[j]) continue;
            
            if (calculateIOU(boxes[i], boxes[j]) > iouThreshold) {
                active[j] = false;
            }
        }
    }
    return selected;
}

function calculateIOU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);
    
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;
    
    return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

/**
 * Simulates YOLOv8 bounding boxes when running in local development simulation mode.
 * Places realistic coordinates around leaf presets and uploaded leaf features.
 * @param {string} pestId - Predicted/selected pest target identifier
 * @param {number} imgWidth - Target image canvas width
 * @param {number} imgHeight - Target image canvas height
 * @returns {Array} Mock bounding boxes matching YOLOv8 output
 */
export function simulateYoloDetections(pestId, imgWidth, imgHeight) {
    const boxes = [];
    const scoreBase = 0.85 + Math.random() * 0.12;
    
    if (pestId === 'brown_planthopper') {
        boxes.push(
            { x: imgWidth * 0.44, y: imgHeight * 0.7, w: imgWidth * 0.14, h: imgHeight * 0.18, score: parseFloat(scoreBase.toFixed(2)), pestId },
            { x: imgWidth * 0.38, y: imgHeight * 0.78, w: imgWidth * 0.12, h: imgHeight * 0.15, score: parseFloat((scoreBase - 0.06).toFixed(2)), pestId }
        );
    } else if (pestId === 'green_leafhopper') {
        boxes.push(
            { x: imgWidth * 0.42, y: imgHeight * 0.32, w: imgWidth * 0.12, h: imgHeight * 0.14, score: parseFloat(scoreBase.toFixed(2)), pestId },
            { x: imgWidth * 0.50, y: imgHeight * 0.48, w: imgWidth * 0.10, h: imgHeight * 0.12, score: parseFloat((scoreBase - 0.08).toFixed(2)), pestId }
        );
    } else if (pestId === 'leaf_folder') {
        boxes.push(
            { x: imgWidth * 0.4, y: imgHeight * 0.15, w: imgWidth * 0.09, h: imgHeight * 0.62, score: parseFloat(scoreBase.toFixed(2)), pestId }
        );
    } else if (pestId === 'rice_bug') {
        boxes.push(
            { x: imgWidth * 0.46, y: imgHeight * 0.22, w: imgWidth * 0.14, h: imgHeight * 0.15, score: parseFloat(scoreBase.toFixed(2)), pestId },
            { x: imgWidth * 0.40, y: imgHeight * 0.40, w: imgWidth * 0.12, h: imgHeight * 0.14, score: parseFloat((scoreBase - 0.07).toFixed(2)), pestId }
        );
    } else if (pestId === 'stem_borer') {
        boxes.push(
            { x: imgWidth * 0.44, y: imgHeight * 0.52, w: imgWidth * 0.15, h: imgHeight * 0.16, score: parseFloat(scoreBase.toFixed(2)), pestId }
        );
    } else if (pestId === 'whorl_maggot') {
        boxes.push(
            { x: imgWidth * 0.48, y: imgHeight * 0.10, w: imgWidth * 0.08, h: imgHeight * 0.60, score: parseFloat(scoreBase.toFixed(2)), pestId },
            { x: imgWidth * 0.42, y: imgHeight * 0.25, w: imgWidth * 0.07, h: imgHeight * 0.45, score: parseFloat((scoreBase - 0.09).toFixed(2)), pestId }
        );
    }
    
    return boxes;
}
