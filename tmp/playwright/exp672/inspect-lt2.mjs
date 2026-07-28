import fs from "node:fs/promises";
import crypto from "node:crypto";
import { chromium } from "playwright";

const urls = process.argv.slice(2);
if (!urls.length) throw new Error("provide one or more URLs");
const browser = await chromium.launch({ headless: true, channel: "chromium" });
const records = [];
try {
  for (const requestedUrl of urls) {
    const context = await browser.newContext({ serviceWorkers: "block" });
    const diagnostics = { pageErrors: [], requestFailures: [], blockedMethods: [] };
    context.on("weberror", e => diagnostics.pageErrors.push(String(e.error()).slice(0, 500)));
    context.on("requestfailed", r => diagnostics.requestFailures.push({ method: r.method(), url: r.url(), error: r.failure()?.errorText }));
    await context.route("**/*", async route => {
      const method = route.request().method();
      if (method !== "GET" && method !== "HEAD") {
        diagnostics.blockedMethods.push({ method, url: route.request().url() });
        return route.abort("blockedbyclient");
      }
      return route.continue();
    });
    const page = await context.newPage();
    let response = null;
    try {
      response = await page.goto(requestedUrl, { waitUntil: "domcontentloaded", timeout: 45000 });
      await page.locator("body").waitFor({ state: "visible", timeout: 15000 });
      const rawBody = response ? await response.body() : Buffer.alloc(0);
      const data = await page.evaluate(() => {
        const body = (document.body?.innerText || "").replace(/\s+/g, " ").trim();
        const links = [...document.querySelectorAll("a[href]")]
          .map(a => ({ text: (a.textContent || "").replace(/\s+/g, " ").trim().slice(0, 120), href: a.href }))
          .filter(x => x.href.includes("github.com/chili-lab/LT2"));
        const patterns = ["gate", "rho", "mixer", "attention", "loop", "model", "transformer", "BSD", "commit"];
        const evidence = body.split(/(?<=[.!?}\]])\s+/).filter(s => patterns.some(p => s.toLowerCase().includes(p))).slice(0, 60);
        const needles = ["class GatedAttention", "self.gate", "residual_weight", "loop_residual", "residual_gate", "loop_count", "for loop_idx", "parse_layer_pattern", "layer_pattern", "GatedDeltaNet", "sliding_window"];
        const contexts = [];
        for (const needle of needles) {
          let offset = 0;
          for (let count = 0; count < 8; count++) {
            const index = body.indexOf(needle, offset);
            if (index < 0) break;
            contexts.push({ needle, index, text: body.slice(Math.max(0, index - 500), Math.min(body.length, index + needle.length + 3000)) });
            offset = index + needle.length;
          }
        }
        return { bodySample: body.slice(0, 4000), links: links.slice(0, 300), evidence, contexts };
      });
      records.push({ requestedUrl, finalUrl: page.url(), status: response?.status() ?? null, responseBytes: rawBody.length, responseSha256: crypto.createHash("sha256").update(rawBody).digest("hex"), title: await page.title(), ...data, diagnostics, contextClosed: true });
    } catch (error) {
      records.push({ requestedUrl, finalUrl: page.url(), status: response?.status() ?? null, error: String(error), diagnostics, contextClosed: true });
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
}
await fs.writeFile("tmp/playwright/exp672/inspect.json", JSON.stringify({ playwrightVersion: "1.61.1", chromiumVersion: browser.version(), mode: "managed regular Chromium/new headless", records, cleanup: { browserClosed: true } }, null, 2) + "\n");
