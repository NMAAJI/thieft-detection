// Configuration for deployment
const CONFIG = {
    // SERVER_URL: use page origin; page-level SERVER_URL override may exist
    SERVER_URL: window.location.origin,
    USE_HTTPS: window.location.protocol === 'https:'
};

window.APP_CONFIG = CONFIG;
