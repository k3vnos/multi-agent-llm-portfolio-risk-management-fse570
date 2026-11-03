/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local only: dev server on 127.0.0.1, API proxied to the read-only FastAPI server on 127.0.0.1:8000.
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, proxy: { "/replay": api, "/runs": api, "/health": api } },
  test: { globals: true, environment: "jsdom", setupFiles: ["./src/test/setup.ts"], include: ["src/**/*.test.{ts,tsx}"] },
});
