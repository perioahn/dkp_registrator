import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "e2e",
  testMatch: "*.spec.ts",
  workers: 1,
  timeout: 45000,
  use: {
    baseURL: "http://127.0.0.1:8792",
    viewport: { width: 1366, height: 768 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command:
      process.platform === "win32"
        ? "py -3.13 e2e/server.py"
        : "python e2e/server.py",
    url: "http://127.0.0.1:8792/api/state",
    timeout: 90000,
    reuseExistingServer: false,
  },
});
