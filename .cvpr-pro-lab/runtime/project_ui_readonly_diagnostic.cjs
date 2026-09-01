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

async function uiSummary(page) {
  return page.locator("main button, main a, main [role=tab], header button, header a, header [role=tab]").evaluateAll((elements) => {
    const visible = (element) => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    };
    return elements
      .filter(visible)
      .map((element) => ({
        role: element.getAttribute("role") || element.tagName.toLowerCase(),
        text: (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim().slice(0, 120),
        aria_label: (element.getAttribute("aria-label") || "").trim().slice(0, 120),
        href: element.getAttribute("href") || "",
        data_testid: element.getAttribute("data-testid") || "",
        aria_selected: element.getAttribute("aria-selected") || "",
      }))
      .filter((item) => item.text || item.aria_label || item.data_testid)
      .slice(0, 32);
  });
}

async function waitForSpa(page) {
  let previous = null;
  let stable = 0;
  for (let elapsed = 0; elapsed < 30000; elapsed += 500) {
    const fingerprint = `${page.url()}|${await page.locator("main button, main a, header button, header a").count()}`;
    stable = fingerprint === previous ? stable + 1 : 0;
    if (stable >= 2) return;
    previous = fingerprint;
    await page.waitForTimeout(500);
  }
}

async function main() {
  const request = JSON.parse(fs.readFileSync(0, "utf8"));
  const projectId = String(request.projectId || "").trim();
  const cdpUrl = String(request.cdpUrl || "").trim();
  const projectUrl = String(request.projectUrl || "").trim();
  if (!projectId || !cdpUrl || !request.screenshotPath || !request.receiptPath || !isExactProjectUrl(projectUrl, projectId)) {
    throw new Error("projectId, exact projectUrl, cdpUrl, screenshotPath, and receiptPath are required.");
  }
  const runtimeCdp = /^(https?|wss?):\/\//.test(cdpUrl) ? cdpUrl : `http://${cdpUrl}`;
  const receipt = {
    schema_version: "1.0",
    dispatch_id: request.dispatchId || null,
    project: projectId,
    operation: "PROJECT_UI_READONLY_DIAGNOSTIC",
    runtime_cdp: runtimeCdp,
    completed_at: null,
    project_url: null,
    page_title: null,
    screenshot_path: request.screenshotPath,
    login_state: "unknown",
    project_load_state: "not_checked",
    controls: [],
    chat_work_mode_controls: [],
    outcome: "QUARANTINED_EXACT_PROJECT_PAGE_NOT_OPEN",
    actual_attempt_increased: false,
  };
  let browser;
  let temporaryPage = false;
  let page;
  let screenshotStyle = null;
  try {
    browser = await chromium.connectOverCDP(runtimeCdp, { timeout: 30000 });
    const contexts = browser.contexts();
    const pages = contexts.flatMap((context) => context.pages());
    const matches = pages.filter((page) => isExactProjectUrl(page.url(), projectId));
    if (matches.length > 1 || (matches.length === 0 && (!request.allowExactProjectNavigation || contexts.length !== 1))) {
      receipt.error = `Expected exactly one already-open exact Project page, found ${matches.length}.`;
      return receipt;
    }
    page = matches[0];
    if (!page) {
      page = await contexts[0].newPage();
      temporaryPage = true;
      await page.goto(projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    }
    receipt.project_url = page.url();
    receipt.page_title = await page.title();
    if (!isExactProjectUrl(receipt.project_url, projectId)) {
      receipt.error = "Exact Project navigation did not retain the requested Project URL.";
      return receipt;
    }
    await waitForSpa(page);
    const loginControls = page.locator('input[type="password"], input[name*="email" i], button:has-text("Log in")');
    receipt.login_state = (await loginControls.count()) > 0 ? "login_or_auth_control_visible" : "no_login_control_visible";
    receipt.project_load_state = (await page.locator("main").count()) > 0 ? "exact_project_main_visible" : "exact_project_main_not_visible";
    receipt.controls = await uiSummary(page);
    receipt.chat_work_mode_controls = receipt.controls.filter((item) => /\bchat\b|\bwork\b/i.test(`${item.text} ${item.aria_label} ${item.data_testid}`));
    screenshotStyle = await page.addStyleTag({
      content: "nav, aside, [role=navigation] { visibility: hidden !important; }",
    });
    fs.mkdirSync(path.dirname(request.screenshotPath), { recursive: true });
    await page.screenshot({ path: request.screenshotPath, fullPage: true });
    receipt.outcome = "PASS_READONLY_PROJECT_UI_DIAGNOSTIC";
    return receipt;
  } catch (error) {
    receipt.error = String(error && error.message ? error.message : error);
    return receipt;
  } finally {
    if (screenshotStyle) await screenshotStyle.evaluate((element) => element.remove()).catch(() => {});
    if (temporaryPage && page) await page.close().catch(() => {});
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
