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

function relevantPath(rawUrl, projectId) {
  try {
    const url = new URL(rawUrl);
    if (url.href.includes(projectId) || url.pathname.includes("/backend-api/") || url.pathname.includes("workspace")) {
      return url.pathname;
    }
  } catch (_) {
    // Ignore malformed request URLs.
  }
  return null;
}

function isoNow() {
  return new Date().toISOString();
}

async function firstVisible(locator) {
  const count = await locator.count();
  for (let index = 0; index < count; index += 1) {
    if (await locator.nth(index).isVisible().catch(() => false)) return true;
  }
  return false;
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
    operation: "PROJECT_LOAD_READONLY_DIAGNOSTIC",
    runtime_cdp: runtimeCdp,
    completed_at: null,
    final_url: null,
    ready_state: null,
    main_text_length: null,
    main_text_prefix: null,
    relevant_responses: [],
    error_types: [],
    http_error_statuses: [],
    spinner_visible: false,
    modal_visible: false,
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
    const responses = new Map();
    const errors = new Set();
    page.on("response", (response) => {
      const route = relevantPath(response.url(), projectId);
      if (!route) return;
      const key = `${response.status()} ${route}`;
      responses.set(key, { path: route, status: response.status(), resource_type: response.request().resourceType() });
    });
    page.on("console", (message) => {
      if (["error", "warning"].includes(message.type())) errors.add(`console:${message.type()}`);
    });
    page.on("pageerror", () => errors.add("pageerror"));
    await page.goto(projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(30000);
    receipt.final_url = page.url();
    receipt.ready_state = await page.evaluate(() => document.readyState);
    if (!isExactProjectUrl(receipt.final_url, projectId)) {
      receipt.outcome = "QUARANTINED_EXACT_PROJECT_ROUTE_LOST";
      return receipt;
    }
    const main = page.locator("main").first();
    const mainText = (await main.innerText().catch(() => "")) || "";
    receipt.main_text_length = mainText.length;
    receipt.main_text_prefix = mainText.slice(0, 200);
    receipt.relevant_responses = [...responses.values()].sort((left, right) => `${left.status}:${left.path}`.localeCompare(`${right.status}:${right.path}`));
    receipt.error_types = [...errors].sort();
    receipt.http_error_statuses = receipt.relevant_responses.filter((item) => item.status === 401 || item.status === 403 || item.status === 429 || item.status >= 500);
    receipt.spinner_visible = await firstVisible(page.locator('[role="progressbar"], [aria-busy="true"], [data-testid*="spinner" i], [data-testid*="loading" i]'));
    receipt.modal_visible = await firstVisible(page.locator('[role="dialog"], [aria-modal="true"]'));
    receipt.outcome = "PASS_READONLY_PROJECT_LOAD_DIAGNOSTIC";
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
