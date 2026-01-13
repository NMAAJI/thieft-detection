// Configuration for local and Railway deployment
const CONFIG = {
    // Change this to your Railway URL after deployment
    // For local: 'http://localhost:5000'
    // For Railway: 'https://your-app.up.railway.app'
    SERVER_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:5000' 
        : window.location.origin,
    
    // Auto-detect protocol (http/https)
    USE_HTTPS: window.location.protocol === 'https:'
};

// Export for use in HTML
window.APP_CONFIG = CONFIG;
