/* ==========================================================================
   RICEGUARD AI - LAYER 1: DATA & STORAGE LAYER (data.js)
   Pest Database & LocalStorage Serialization State Manager
   ========================================================================== */

// 1. Scientific Rice Disease & Pest Database (IRRI, PhilRice, & FAO Sourced)
export const PEST_DATASET = {
    brown_planthopper: {
        id: 'brown_planthopper',
        name: 'Brown Planthopper (BPH)',
        scientificName: 'Nilaparvata lugens',
        type: 'Insecticide Recommendation',
        chemical: 'Imidacloprid / Pymetrozine',
        dosagePerHa: 250,     // ml or g of formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Yellowing and drying of leaves',
            'Wilting of tillers near the water line',
            'Hopperburn (patches of dried brown crops)',
            'Insects clustered at the base of stems'
        ],
        description: 'BPH damages rice by sucking plant sap and transmitting grassy stunt viruses. High populations cause plants to dry out completely, producing a symptom known as "hopperburn". Check leaf sheaths at the water level.',
        safeInterval: 'Re-entry interval: 12 hours. Pre-harvest interval: 21 days.',
        colorProfile: { r: 165, g: 105, b: 55, tolerance: 38 } // Warm orange-brown planthopper insects
    },
    green_leafhopper: {
        id: 'green_leafhopper',
        name: 'Green Leafhopper (GLH)',
        scientificName: 'Nephotettix virescens',
        type: 'Insecticide Recommendation',
        chemical: 'Imidacloprid / Buprofezin',
        dosagePerHa: 300,     // ml of formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Yellowing of leaves starting from tips',
            'Stunted growth and reduced tillering',
            'Plants appearing bunched or grassy',
            'Insects jumping when crop is disturbed'
        ],
        description: 'GLH is a major sap-sucking pest of rice. More importantly, it serves as the primary vector for Rice Tungro Virus. Early management is critical to prevent virus outbreaks.',
        safeInterval: 'Re-entry interval: 12 hours. Pre-harvest interval: 14 days.',
        colorProfile: { r: 85, g: 165, b: 70, tolerance: 30 } // Vibrant green leafhopper insects
    },
    leaf_folder: {
        id: 'leaf_folder',
        name: 'Rice Leaf Folder',
        scientificName: 'Cnaphalocrocis medinalis',
        type: 'Insecticide Recommendation',
        chemical: 'Flubendiamide / Spinosad',
        dosagePerHa: 200,     // ml of formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Leaves folded longitudinally in tubes',
            'White transparent streaks on leaves',
            'Webbing holding leaf margins together',
            'Larval feeding scars removing chlorophyll'
        ],
        description: 'Leaf folder larvae fold rice leaves to create protective chambers. They scrape the green leaf tissue, leaving white streaks. This reduces photosynthesis and hinders grain filling. Spray only if flag leaf damage exceeds 15% threshold.',
        safeInterval: 'Re-entry interval: 12 hours. Pre-harvest interval: 7 days.',
        colorProfile: { r: 215, g: 225, b: 205, tolerance: 22 } // Papery white scraped streaks
    },
    rice_bug: {
        id: 'rice_bug',
        name: 'Rice Bug',
        scientificName: 'Leptocorisa oratorius',
        type: 'Insecticide Recommendation',
        chemical: 'Cypermethrin',
        dosagePerHa: 350,     // ml of formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Brownish spots on developing grains',
            'Empty, shriveled or deformed grains',
            'Foul odor emitted in infected fields',
            'Slender green bugs visible on panicles'
        ],
        description: 'Rice bugs suck sap from developing rice grains during the milky stage, leading to empty or partially filled grains ("pecky rice"). Field sanitation and synchronized planting minimize damage.',
        safeInterval: 'Re-entry interval: 24 hours. Pre-harvest interval: 14 days.',
        colorProfile: { r: 110, g: 125, b: 75, tolerance: 32 } // Olive-green rice bugs
    },
    stem_borer: {
        id: 'stem_borer',
        name: 'Yellow Stem Borer',
        scientificName: 'Scirpophaga incertulas',
        type: 'Insecticide Recommendation',
        chemical: 'Chlorantraniliprole',
        dosagePerHa: 150,     // grams of granular formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Deadhearts (withered central leaf tillers)',
            'Whiteheads (empty, bleached panicles)',
            'Bore holes at the lower stem nodes',
            'Frass or caterpillar larvae inside stems'
        ],
        description: 'Stem Borer larvae bore into the rice stem, severing the vascular system. Early attacks cause withered tillers ("deadhearts"). Attacks during heading cause empty, white panicles ("whiteheads"). Apply treatments early based on moth counts.',
        safeInterval: 'Re-entry interval: 24 hours. Pre-harvest interval: 14 days.',
        colorProfile: { r: 135, g: 90, b: 40, tolerance: 30 } // Darker yellow-brown woody decay nodes
    },
    whorl_maggot: {
        id: 'whorl_maggot',
        name: 'Rice Whorl Maggot',
        scientificName: 'Hydrellia philippina',
        type: 'Insecticide Recommendation',
        chemical: 'Carbofuran',
        dosagePerHa: 200,     // grams of granular formulation per hectare
        waterPerHa: 400,      // Liters of spray solution per hectare
        symptoms: [
            'Linear yellow-white scars on leaf margins',
            'Shriveled or distorted unopened leaf tips',
            'Stunted tillers and plant growth retardation',
            'Small white maggots visible inside leaf whorls'
        ],
        description: 'Whorl Maggots feed on the leaf margins before they unfold, leaving visible white-yellow streaks. Damage is most severe during the vegetative phase in irrigated fields.',
        safeInterval: 'Re-entry interval: 48 hours. Pre-harvest interval: 21 days.',
        colorProfile: { r: 175, g: 185, b: 110, tolerance: 30 } // Yellow-white linear scars
    }
};

// 2. Application Global State Store
export const state = {
    selectedPest: null,
    scanImageData: null, // Clean cached upload image data
    farmGridData: Array(100).fill(false), // 10x10 grid (false=healthy, true=infected)
    isPaintingOnGrid: false,
    currentWeatherData: { wind: 4.5, rain: 15, temp: 28.5 },
    activeEngine: 'roboflow', // Options: 'roboflow', 'yolov8'
    yoloModelStatus: 'loading', 
    detectedBoxes: [] 
};

// 3. LocalStorage Storage Management Functions
export const reportsStore = {
    save(report) {
        const history = this.getAll();
        history.unshift(report);
        localStorage.setItem('rice_guard_reports', JSON.stringify(history));
    },
    getAll() {
        return JSON.parse(localStorage.getItem('rice_guard_reports') || '[]');
    },
    delete(reportId) {
        let history = this.getAll();
        history = history.filter(r => r.id !== reportId);
        localStorage.setItem('rice_guard_reports', JSON.stringify(history));
        return history;
    },
    getById(reportId) {
        const history = this.getAll();
        return history.find(r => r.id === reportId);
    }
};

export const trackerStore = {
    get(id) {
        return localStorage.getItem(`tracker_${id}`) === 'true';
    },
    set(id, val) {
        localStorage.setItem(`tracker_${id}`, val ? 'true' : 'false');
    }
};
