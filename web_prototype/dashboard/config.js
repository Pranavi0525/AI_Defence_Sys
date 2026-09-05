// web_prototype/dashboard/config.js
//
// Static, no-build-step configuration for the AI Defense Lab dashboard.
//
// This file exists so the deployed API base URL can be set WITHOUT a
// build/bundle step: after deploying this static site (e.g. to Vercel),
// edit the line below to point at your deployed Render backend, e.g.
//
//     window.AI_DEFENSE_API_BASE = "https://ai-defense-api.onrender.com";
//
// Leave it as an empty string for local development -- the dashboard
// falls back to the on-page input field (which itself falls back to
// whatever was last saved in this browser's localStorage, or otherwise
// stays blank and asks the user to enter a backend URL explicitly).
//
// Intentionally NOT localhost by default: shipping "http://localhost:8000"
// as the default here would silently point a deployed frontend at the
// visitor's own machine, which is never correct after deployment.
window.AI_DEFENSE_API_BASE = "";
