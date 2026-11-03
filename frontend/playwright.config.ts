import { defineConfig } from "@playwright/test";

// Browser tests against the locally running product (./run.sh start). Uses the installed Google Chrome:
// no browser download.
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:8000", channel: "chrome", viewport: { width: 1440, height: 900 },
         screenshot: "only-on-failure" },
  reporter: [["list"]],
  outputDir: "./e2e-results",
});
