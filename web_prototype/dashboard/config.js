// web_prototype/dashboard/config.js
//
// Static, no-build-step configuration for the AI Defense System dashboard.
//
// This file exists so the deployed API base URL can be set WITHOUT a
// build/bundle step. It is set here to the canonical production backend.
// The on-page API URL field still lets anyone override this (saved to
// this browser's localStorage) -- e.g. for local development against
// http://localhost:8000 -- and that override always wins over this file.
//
// Intentionally NOT localhost by default: shipping "http://localhost:8000"
// as the default here would silently point a deployed frontend at the
// visitor's own machine, which is never correct after deployment.
window.AI_DEFENSE_API_BASE = "https://ai-defense-api.onrender.com";
