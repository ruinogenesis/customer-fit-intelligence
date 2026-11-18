// The staff dashboard (T401): React + TypeScript, built by Vite for Firebase
// Hosting (dist/), tested by Vitest in a simulated browser (jsdom).
import react from "@vitejs/plugin-react";
import type { ProxyOptions } from "vite";
import { defineConfig } from "vitest/config";

// index.html names the commit it was built from; CI sets it, locally "local".
process.env.VITE_COMMIT ??= "local";

// npm run dev:cloud: the API and the Firebase config come from dev Hosting,
// as they would on the deployed site (T407).
const DEV_HOSTING = { target: "https://cfi-dev-th26.web.app", changeOrigin: true };

// npm run dev: the API run locally on port 8000, against the emulators.
const LOCAL_API = "http://127.0.0.1:8000";

export default defineConfig(({ mode }) => {
  const proxy: Record<string, string | ProxyOptions> =
    mode === "cloud"
      ? { "/api": DEV_HOSTING, "/__/firebase": DEV_HOSTING }
      : { "/api": LOCAL_API };
  return {
    plugins: [react()],
    server: { proxy },
    test: {
      environment: "jsdom",
      include: ["tests/**/*.test.{ts,tsx}"],
    },
  };
});
