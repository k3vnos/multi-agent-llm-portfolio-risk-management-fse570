import { expect, test, type Page } from "@playwright/test";

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

test("Simple Mode is the default and answers the three questions with the saved numbers", async ({ page, request }) => {
  await page.goto(`/?t=${Date.now()}`);
  await expect(page.getByTestId("three-questions")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("button", { name: "Simple", pressed: true })).toBeVisible();
  for (const q of ["1 · What is happening?", "2 · Why alert or no alert?", "3 · What evidence supports it?"])
    await expect(page.getByText(q)).toBeVisible();
  const meta = await (await request.get("/replay/meta")).json();
  const s = await (await request.get(`/replay/snapshot/${meta.end}`)).json();
  await expect(page.getByTestId("why-alert")).toContainText(s.model.score.toFixed(2));
  await expect(page.getByTestId("why-alert")).toContainText(s.model.threshold.toFixed(2));
  await expect(page.getByText("Historical replay · past data, not live")).toBeVisible();
  await expect(page.getByTestId("chart-explanation")).toContainText("trading days");
});

test("COVID date: alert explained with score above the alert line", async ({ page }) => {
  await open(page, "#/home?date=2020-02-24", "What the data showed that day");
  await expect(page.getByTestId("why-alert")).toHaveText("Its score was 0.83, above its alert line of 0.32 — so it raised an alert.");
  await expect(page.getByTestId("evidence-rows")).toContainText("8 headlines");
});

test("switching modes keeps the date and lands on the matching page", async ({ page }) => {
  await open(page, "#/home?date=2020-02-24", "What the data showed that day");
  await page.getByRole("button", { name: "Advanced" }).click();
  await expect(page).toHaveURL(/#\/monitor\?date=2020-02-24/);
  await expect(page.getByText("What moved this month")).toBeVisible();
  await page.getByRole("button", { name: "Relationship network" }).click();
  await page.getByRole("button", { name: "Simple" }).click();
  await expect(page).toHaveURL(/#\/relations\?date=2020-02-24/);
});

test("relationships: top 5 changes by default, all on request, inspect a change", async ({ page }) => {
  await open(page, "#/relations?date=2020-02-24", "How to read this");
  await expect(page.getByTestId("top-changes").locator("li")).toHaveCount(5);
  const arrowsTop = await page.locator("[data-edge]").count();
  expect(arrowsTop).toBe(5);
  await page.getByRole("button", { name: "All changes" }).click();
  expect(await page.locator("[data-edge]").count()).toBeGreaterThan(5);
  await page.getByRole("button", { name: "Top 5 changes" }).click();
  await page.getByTestId("top-changes").locator("button").first().click();
  await expect(page.getByTestId("simple-inspector")).toContainText("Estimated strength");
});

test("AI explanation: concise agents, details and unverified draft behind expanders", async ({ page }) => {
  await open(page, "#/explain?date=2020-01-10", "Synthesis agent");
  await expect(page.getByTestId("agent-summaries")).toContainText("Graph agent");
  await expect(page.getByTestId("details-llm")).not.toContainText("Unverified draft written");
  await page.getByRole("button", { name: "Show the AI-written draft (unverified)" }).click();
  await expect(page.getByTestId("details-llm")).toContainText("73%");
  await page.getByRole("button", { name: /Show the detailed evidence/ }).click();
  await expect(page.getByTestId("details-evidence")).toContainText("available 2020-01-10");
  await open(page, "#/explain?date=2023-03-08", "Synthesis agent");
  await expect(page.getByTestId("agent-summaries")).toContainText("missing data, not 'no news'");
});

test("what we found: plain result and locked numbers", async ({ page, request }) => {
  await open(page, "#/findings", "Important limitations");
  await expect(page.getByTestId("finding-plain")).toContainText("the answer was no");
  const r = await (await request.get("/replay/research")).json();
  for (const m of ["vix", "bayes_logit:controls", "bayes_logit:combined", "inst_composite", "bayes_logit:instability"]) {
    const v = r.primary.data.methods.find((x: { method: string }) => x.method === m).roc_auc.toFixed(2);
    await expect(page.getByTestId("finding-bars").locator(`[data-method="${m}"]`)).toContainText(v);
  }
});

test("simple guided demo walks seven plain-language steps", async ({ page }) => {
  await page.goto(`/?t=${Date.now()}`);
  await expect(page.getByTestId("three-questions")).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "▶ Guided demo" }).click();
  const pages = ["home", "home", "home", "relations", "portfolio", "explain", "findings"];
  for (let i = 0; i < pages.length; i++) {
    await expect(page.getByTestId("tour")).toContainText(`step ${i + 1} of 7`);
    await expect(page).toHaveURL(new RegExp(`#/${pages[i]}`));
    if (i < pages.length - 1) await page.getByTestId("tour").getByRole("button", { name: "Next", exact: true }).click();
  }
  await expect(page.getByTestId("tour")).toContainText("did not give earlier or better warnings");
});
