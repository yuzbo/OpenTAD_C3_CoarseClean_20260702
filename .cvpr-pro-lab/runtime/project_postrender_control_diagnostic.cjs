#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const PLAYWRIGHT_NODE_MODULES = "C:/Users/skywalker/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
module.paths.unshift(PLAYWRIGHT_NODE_MODULES);
const { chromium } = require("playwright");

function isExactProjectUrl(rawUrl, projectId) {
  try {
    const url = new URL(rawUrl);
    return url.origin === "https://chatgpt.com" && url.pathname === `/g/${projectId}/project`;
  } catch (_) {
    return false;
  }
}

function isoNow() {
  return new Date().toISOString();
}

async function waitForProjectContent(page) {
  for (let elapsed = 0; elapsed < 45000; elapsed += 1000) {
    const mainText = (await page.locator("main").first().innerText().catch(() => "")) || "";
    if (["DUCA", "Chats", "Sources"].every((token) => mainText.includes(token))) return true;
    await page.waitForTimeout(1000);
  }
  return false;
}

async function relevantControls(page) {
  return page.locator("main button, main a, main [role=tab], main [role=combobox], header button, header a, header [role=tab], header [role=combobox]").evaluateAll((elements) => {
    const visible = (element) => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    };
    const summary = (element) => {
      const text = (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim().slice(0, 120);
      const aria = (element.getAttribute("aria-label") || "").trim().slice(0, 120);
      const href = element.getAttribute("href") || "";
      const testid = element.getAttribute("data-testid") || "";
      const parent = element.parentElement;
      const parentText = (parent?.innerText || parent?.textContent || "").replace(/\s+/g, " ").trim().slice(0, 160);
      return {
        role: element.getAttribute("role") || element.tagName.toLowerCase(),
        text,
        aria_label: aria,
        href,
        data_testid: testid,
        parent_text: parentText,
      };
    };
    return elements
      .filter(visible)
      .map(summary)
      .filter((item) => /\bnew\s+chat\b|\bstart(?:\s+a)?\s+chat(?:ting)?\b|\bchat\b|\bpro\b|\bchats\b|\bsources\b/i.test(`${item.text} ${item.aria_label} ${item.href} ${item.data_testid} ${item.parent_text}`))
      .slice(0, 24);
  });
}

async function main() {
  const request = JSON.parse(fs.readFileSync(0, "utf8"));
  const projectId = String(request.projectId || "").trim();
  const projectUrl = String(request.projectUrl || "").trim();
  const cdpUrl = String(request.cdpUrl || "").trim();
  if (!projectId || !isExactProjectUrl(projectUrl, projectId) || !cdpUrl || !request.receiptPath) {
    throw new Error("projectId, exact projectUrl, cdpUrl, and receiptPath are required.");
  }
  const runtimeCdp = /^(https?|wss?):\/\//.test(cdpUrl) ? cdpUrl : `http://${cdpUrl}`;
  const receipt = {
    schema_version: "1.0",
    dispatch_id: request.dispatchId || null,
    project: projectId,
    operation: "PROJECT_UI_POSTRENDER_CONTROL_DIAGNOSTIC",
    runtime_cdp: runtimeCdp,
    completed_at: null,
    final_url: null,
    render_ready: false,
    candidates: [],
    outcome: "QUARANTINED_PRECHECK_FAILED",
    actual_attempt_increased: false,
  };
  let browser;
  let page;
  try {
    browser = await chromium.connectOverCDP(runtimeCdp, { timeout: 30000 });
    const contexts = browser.contexts();
    if (contexts.length !== 1) throw new Error(`Expected one iXBrowser context, found ${contexts.length}.`);
    page = await contexts[0].newPage();
    await page.goto(projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    receipt.final_url = page.url();
    if (!isExactProjectUrl(receipt.final_url, projectId)) {
      receipt.outcome = "QUARANTINED_EXACT_PROJECT_ROUTE_LOST";
      return receipt;
    }
    receipt.render_ready = await waitForProjectContent(page);
    if (!receipt.render_ready) {
      receipt.outcome = "QUARANTINED_POSTRENDER_SIGNALS_MISSING";
      return receipt;
    }
    receipt.candidates = await relevantControls(page);
    receipt.outcome = "PASS_READONLY_POSTRENDER_CONTROL_DIAGNOSTIC";
    return receipt;
  } catch (error) {
    receipt.error = String(error && error.message ? error.message : error);
    return receipt;
  } finally {
    if (page) await page.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    receipt.completed_at = isoNow();
    fs.mkdirSync(path.dirname(request.receiptPath), { recursive: true });
    fs.writeFileSync(request.receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify(receipt)}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${JSON.stringify({ ok: false, error: String(error.message || error) })}\n`);
  process.exitCode = 2;
});
