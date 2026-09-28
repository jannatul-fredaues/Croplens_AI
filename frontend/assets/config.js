// Single place to point the frontend at the backend. Edit the production
// URL after you deploy the API (Day 6) - everything else reads from here.
window.CROPLENS_CONFIG = {
  API_BASE_URL:
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://REPLACE-WITH-YOUR-DEPLOYED-API-URL", // TODO: set after Day 6 deploy
};
