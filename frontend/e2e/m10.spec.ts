import { expect, test, type Page } from "@playwright/test";

// M10: Portfolio Lens, Agent Trace, live "Ask the agents" (real local model; skipped if Ollama is not running) and
// the full COVID-19 user journey: select portfolio -> choose date -> inspect risk -> run agents -> trace.

const errors: string[] = [];
test.beforeEach(async ({ page }) => {
  errors.length = 0;
  page.on("console", (m) => { if (m.type() === "error" && !/status of 4\d\d/.test(m.text())) errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
});
test.afterEach(() => { expect(errors, errors.join("\n")).toEqual([]); });

async function open(page: Page, hash: string, ready: string) {
  await page.goto(`/?t=${Date.now()}${hash}`);
  await expect(page.getByText(ready).first()).toBeVisible({ timeout: 20_000 });
}

async function modelReady(page: Page) {
  const r = await (await page.request.get("/agents/status")).json();
  return r.available as boolean;
}

test("portfolio editor: example, validation, weights must total 100%", async ({ page }) => {
  await open(page, "#/portfolio?date=2020-03-16", "Build a portfolio");
  await expect(page.getByTestId("exploratory-banner")).toContainText("separate from the project's locked research evaluation");
  await page.getByRole("button", { name: /Classic 60\/40/ }).click();
  await expect(page).toHaveURL(/pf=SPY:60,IEF:40/);
  await expect(page.getByTestId("weight-total")).toHaveText("Total 100%");
  await page.getByLabel("Weight 1").fill("50");
  await expect(page.getByTestId("editor-problems")).toContainText("must total 100%");
  await expect(page.getByRole("button", { name: "Analyse this portfolio" })).toBeDisabled();
  await page.getByRole("button", { name: "Scale to 100%" }).click();
  await expect(page.getByTestId("weight-total")).toHaveText("Total 100%");
  await page.getByRole("button", { name: "+ Add fund" }).click();
  await page.getByLabel("Fund 3").selectOption("GLD");
  await page.getByLabel("Weight 3").fill("10");
  await page.getByLabel("Weight 1").fill("45.5");
  await page.getByLabel("Weight 2").fill("44.5");
  await page.getByRole("button", { name: "Analyse this portfolio" }).click();
  await expect(page).toHaveURL(/pf=SPY:45.5,IEF:44.5,GLD:10/);
  await expect(page.getByTestId("risk-contributions")).toContainText("GLD");
});

test("portfolio numbers on screen equal the point-in-time API", async ({ page, request }) => {
  await open(page, "#/lens?date=2020-03-16&pf=SPY:60,IEF:40", "Which holdings carry the risk?");
  const a = await (await request.get("/portfolio/analyze?spec=SPY:60,IEF:40&date=2020-03-16")).json();
  expect(a.data_through).toBe("2020-03-16");
  await expect(page.getByTestId("m-vol")).toContainText(`${a.metrics.vol_63d.toFixed(1)}%`);
  await expect(page.getByTestId("m-var")).toContainText(`${a.metrics.var95_hist.toFixed(2)}%`);
  await expect(page.getByTestId("m-dd")).toContainText(`${a.metrics.max_drawdown_252d.toFixed(1)}%`);
  await expect(page.getByTestId("risk-sentences")).toContainText(`${a.holdings[0].risk_share.toFixed(0)}%`);
  await expect(page.getByTestId("portfolio-network")).toContainText("not causal effects");
  await expect(page.getByTestId("ask-agents")).toContainText("Include my portfolio");
});

test("saved agent trace shows inputs, confirmed and rejected statements, and synthesis", async ({ page, request }) => {
  const saved = await (await request.get("/agents/saved/2020-01-10")).json();
  test.skip(!saved.record, "saved v4 explanations not generated yet");
  await page.route("**/agents/status", (r) => r.fulfill({ json: { available: false, model: "qwen2.5:7b", reason: "test: model off", busy: false, notice: "", fallback: "" } }));
  await open(page, "#/agents?date=2020-01-10", "Ask the agents about");
  await expect(page.getByTestId("model-status")).toContainText("not running");
  await expect(page.getByRole("button", { name: /Ask the agents/ })).toBeDisabled();
  const tr = page.getByTestId("saved-explanation").getByTestId("agent-trace");
  await expect(tr.getByTestId("agent-flow")).toContainText("Validator");
  for (const a of ["graph", "macro", "news"]) await expect(tr.getByTestId(`trace-${a}`)).toBeVisible();
  const nConfirmed = saved.record.specialists.reduce((n: number, s: { claims: unknown[] }) => n + s.claims.length, 0);
  await expect(tr.getByTestId("confirmed-claim")).toHaveCount(nConfirmed);
  await expect(tr.getByTestId("synthesis-text")).not.toBeEmpty();
});

test("news agent abstains explicitly when there is no news coverage", async ({ page, request }) => {
  const saved = await (await request.get("/agents/saved/2023-03-13")).json();
  test.skip(!saved.record, "saved v4 explanations not generated yet");
  await page.route("**/agents/status", (r) => r.fulfill({ json: { available: false, model: "qwen2.5:7b", reason: "off", busy: false, notice: "", fallback: "" } }));
  await open(page, "#/explain?date=2023-03-13", "Ask the agents about");
  await expect(page.getByTestId("abstain-news").first()).toContainText("abstains");
});

test("live run: real progress from the local model, validated trace, score unchanged", async ({ page, request }) => {
  test.skip(!(await modelReady(page)), "Ollama is not running locally");
  test.setTimeout(300_000);
  const before = (await (await request.get("/replay/snapshot/2020-02-24")).json()).model;
  await open(page, "#/explain?date=2020-02-24", "Ask the agents about");
  await expect(page.getByTestId("live-notice")).toContainText("not real-time market monitoring");
  await page.getByRole("button", { name: /Ask the agents/ }).click();
  await expect(page.getByTestId("agent-progress")).toBeVisible();
  await expect(page.getByTestId("progress-graph")).toHaveAttribute("data-state", /running|done/, { timeout: 30_000 });
  await expect(page.getByTestId("live-result")).toBeVisible({ timeout: 240_000 });
  const res = page.getByTestId("live-result");
  for (const a of ["graph", "macro", "news"]) await expect(res.getByTestId(`trace-${a}`)).toBeVisible();
  await expect(res.getByTestId("trace-synthesis")).toBeVisible();
  for (const a of ["graph", "macro", "news", "synthesis"]) await expect(page.getByTestId(`progress-${a}`)).toHaveAttribute("data-state", "done");
  const after = (await (await request.get("/replay/snapshot/2020-02-24")).json()).model;
  expect(after).toEqual(before);
});

test("live run can be cancelled", async ({ page }) => {
  test.skip(!(await modelReady(page)), "Ollama is not running locally");
  test.setTimeout(120_000);
  await open(page, "#/agents?date=2019-12-03", "Ask the agents about");
  await page.getByRole("button", { name: /Ask the agents/ }).click();
  await expect(page.getByTestId("progress-graph")).toHaveAttribute("data-state", "running", { timeout: 30_000 });
  await page.getByRole("button", { name: "■ Cancel" }).click();
  await expect(page.getByTestId("job-stopped")).toContainText("cancelled", { timeout: 60_000 });
  await expect(page.getByTestId("live-result")).toHaveCount(0);
});

test("COVID-19 journey: portfolio -> date -> risk -> agents -> trace", async ({ page }) => {
  const live = await modelReady(page);
  test.setTimeout(300_000);
  await open(page, "#/home?date=2019-12-31", "What the data showed that day");
  await page.getByRole("button", { name: "My portfolio" }).click();
  await page.getByRole("button", { name: /Classic 60\/40/ }).click();
  await page.getByTestId("simple-replay").getByRole("button", { name: /COVID-19 crash/ }).click();
  await expect(page.getByTestId("replay-date")).toContainText("2020");
  await page.getByLabel("Replay date").fill("2020-03-16");
  await expect(page.getByTestId("portfolio-metrics")).toContainText("Mar 16, 2020");
  await expect(page.getByTestId("m-vol")).toContainText("unusually high");
  await expect(page.getByTestId("risk-sentences")).toContainText("SPY");
  await page.getByRole("button", { name: "Ask the AI agents" }).click();
  await expect(page).toHaveURL(/#\/explain\?date=2020-03-16&pf=SPY:60,IEF:40/);
  await expect(page.getByTestId("ask-agents")).toContainText("Include my portfolio");
  if (!live) return;
  await page.getByRole("button", { name: /Ask the agents \(/ }).click();
  await expect(page.getByTestId("live-result")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByTestId("live-result")).toContainText("portfolio SPY:60,IEF:40");
  await expect(page.getByTestId("live-result").getByTestId("agent-flow")).toContainText("about your portfolio");
});
