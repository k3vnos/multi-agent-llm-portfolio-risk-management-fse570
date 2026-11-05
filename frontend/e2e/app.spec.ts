import { expect, test, type Page } from "@playwright/test";

// End-to-end checks against the locally running product (./run.sh start). Numbers on screen are compared with
// the read-only API, which is itself tested against the saved M5 artifacts.

const errors: string[] = [];
test.beforeEach(async ({ page }) => {
  errors.length = 0;
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
});
test.afterEach(() => { expect(errors, errors.join("\n")).toEqual([]); });

async function open(page: Page, hash: string, ready: string) {
  await page.goto(`/?t=${Date.now()}${hash}`);
  await expect(page.getByText(ready).first()).toBeVisible({ timeout: 20_000 });
}

test("monitor shows the replay snapshot for the chosen date", async ({ page, request }) => {
  await open(page, "#/monitor?date=2020-02-24", "What moved this month");
  const snap = await (await request.get("/replay/snapshot/2020-02-24")).json();
  await expect(page.locator("header").getByText("HISTORICAL REPLAY")).toBeVisible();
  await expect(page.getByTestId("kpi-sii")).toHaveText(snap.sii.value.toFixed(2));
  await expect(page.getByTestId("kpi-status")).toContainText("Model alert");
  await expect(page.getByTestId("stress-chart").locator("svg").first()).toBeVisible();
  await expect(page.getByTestId("briefing")).toContainText("not a forecast");
  await expect(page.locator("body")).not.toContainText(/next refresh|live data feed|[0-9a-f]{40}/);
});

test("changing the replay date updates every panel consistently", async ({ page, request }) => {
  await open(page, "#/monitor?date=2020-02-24", "What moved this month");
  await page.getByLabel("Replay date").fill("2019-06-03");
  const snap = await (await request.get("/replay/snapshot/2019-06-03")).json();
  await expect(page.getByTestId("kpi-sii")).toHaveText(snap.sii.value.toFixed(2));
  await expect(page.getByText("Viewing")).toContainText("Jun 3, 2019");
  await page.getByRole("button", { name: "1w ▶" }).click();
  await expect(page).toHaveURL(/date=2019-06-10/);
});

test("episode replay and play/pause move the date forward", async ({ page }) => {
  await open(page, "#/monitor?date=2019-01-02", "What moved this month");
  await page.getByRole("button", { name: /COVID-19 market stress/ }).click();
  await expect(page).toHaveURL(/date=2020-01-1\d|date=2020-01-2\d|date=2020-02/, { timeout: 5_000 });
  await page.getByRole("button", { name: /Pause/ }).click();
  const url = page.url();
  await page.waitForTimeout(1500);
  expect(page.url()).toBe(url);
});

test("timeline separates hindsight from point-in-time data", async ({ page }) => {
  await open(page, "#/timeline?date=2020-02-24", "Historical episodes");
  await expect(page.getByText(/Hindsight view/)).toHaveCount(0);
  await page.getByRole("button", { name: "Show what happened next" }).click();
  await expect(page.getByText(/none of this was known on/)).toBeVisible();
  await page.getByRole("button", { name: /COVID-19 market stress/ }).last().click();
  await expect(page).toHaveURL(/date=2020-02-21/);
});

test("network: nodes and arrows are inspectable, modes switch", async ({ page }) => {
  await open(page, "#/network?date=2020-02-24", "Largest changes");
  await page.locator('[data-node="HYG"]').click();
  await expect(page.getByTestId("inspector")).toContainText("High-yield corporate bonds");
  await page.getByTestId("change-list").locator("button").first().click();
  await expect(page.getByTestId("inspector")).toContainText("not a causal effect");
  await page.getByRole("button", { name: "Current links" }).click();
  await expect(page.getByText("Positive dependence")).toBeVisible();
  await page.getByRole("button", { name: "Matrix" }).click();
  await expect(page.getByTestId("matrix")).toBeVisible();
});

test("alert explorer shows evidence and keeps the outcome as hindsight", async ({ page }) => {
  await open(page, "#/alerts?date=2020-01-10&alert=bayes_logit:combined@2020-01-10", "Retrospective outcome");
  await expect(page.getByTestId("alert-detail")).toContainText("0.445");
  await expect(page.getByText("Known only with hindsight")).toBeVisible();
  await page.getByRole("button", { name: "Network →" }).click();
  await expect(page).toHaveURL(/#\/network\?date=2020-01-10/);
});

test("AI analysis: deterministic briefing, unverified LLM draft, explicit missing news", async ({ page }) => {
  await open(page, "#/agents?date=2020-01-10", "M6 free-text LLM draft");
  await expect(page.getByTestId("synthesis")).toBeVisible();
  await expect(page.getByTestId("llm-draft")).toContainText("UNVERIFIED LLM DRAFT");
  await expect(page.getByTestId("llm-draft")).toContainText("unfaithful");
  await open(page, "#/agents?date=2023-03-08", "M6 free-text LLM draft");
  await expect(page.getByTestId("news-unavailable")).toContainText("not neutral");
});

test("research section reports the locked results exactly", async ({ page, request }) => {
  await open(page, "#/research", "Leakage audit, reproducibility");
  const r = await (await request.get("/replay/research")).json();
  const vix = r.primary.data.methods.find((m: { method: string }) => m.method === "vix");
  await expect(page.getByTestId("conclusion")).toContainText("did not give earlier or better warning");
  await expect(page.getByTestId("track-record").locator('[data-method="vix"]')).toContainText(vix.roc_auc.toFixed(3));
  await expect(page.getByTestId("track-record").locator('[data-method="inst_composite"]')).toContainText("0.601");
  await expect(page.getByTestId("audit")).not.toContainText("fail");
  await expect(page.getByText("bit-identical")).toBeVisible();
});

test("guided demo walks through all seven steps on real pages", async ({ page }) => {
  await open(page, "#/monitor", "What moved this month");
  await page.getByRole("button", { name: "▶ Guided demo" }).click();
  const expected = ["monitor", "monitor", "timeline", "network", "lens", "agents", "research"];
  for (let i = 0; i < expected.length; i++) {
    await expect(page.getByTestId("tour")).toContainText(`step ${i + 1} of 7`);
    await expect(page).toHaveURL(new RegExp(`#/${expected[i]}`));
    if (i < expected.length - 1) await page.getByTestId("tour").getByRole("button", { name: "Next", exact: true }).click();
  }
  await page.getByTestId("tour").getByRole("button", { name: "Finish" }).click();
  await expect(page.getByTestId("tour")).toHaveCount(0);
});
