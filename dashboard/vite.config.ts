// The staff dashboard (T401): React + TypeScript, built by Vite for Firebase
// Hosting (dist/), tested by Vitest in a simulated browser (jsdom).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// index.html names the commit it was built from; CI sets it, locally "local".
process.env.VITE_COMMIT ??= "local";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
