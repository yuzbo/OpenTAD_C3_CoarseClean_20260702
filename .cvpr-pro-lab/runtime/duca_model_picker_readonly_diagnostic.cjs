#!/usr/bin/env node
"use strict";

// One central-lease, read-only diagnostic for the exact DUCA Project composer.
const fs = require("fs");
const path = require("path");
const PLAYWRIGHT_NODE_MODULES = "C:/Users/skywalker/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
module.paths.unshift(PLAYWRIGHT_NODE_MODULES);
const { chromium } = require("playwright");

function now() { return new Date().toISOString(); }
function compact(value) { return String(value || "").replace(/\s+/g, " ").trim().slice(0, 160); }
function projectUrl(id) { return `https://chatgpt.com/g/${id}/project`; }
function isExact(url, id) { try { const parsed = new URL(url); return parsed.origin === "https://chatgpt.com" && parsed.pathname === `/g/${id}/project`; } catch (_) { return false; } }
function scoped(url, id) { try { const parsed = new URL(url); return parsed.origin === "https://chatgpt.com" && parsed.pathname.startsWith(`/g/${id}/`); } catch (_) { return false; } }
function write(file, value) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

async function waitForProject(page) {
  for (let elapsed = 0; elapsed < 45000; elapsed += 1000) {
    const text = await page.locator("main").first().innerText().catch(() => "");
    if (["DUCA", "Chats", "Sources"].every((token) => text.includes(token))) return;
    await page.waitForTimeout(1000);
  }
  throw new Error("Rendered DUCA/Chats/Sources Project surface was not available.");
}

async function exactContext(browser, id) {
  const matches = browser.contexts().filter((context) => context.pages().some((page) => scoped(page.url(), id)));
  if (matches.length === 1) return matches[0];
  if (matches.length > 1 || browser.contexts().length !== 1) throw new Error("No unambiguous exact DUCA browser context.");
  return browser.contexts()[0];
}

async function openBlankProjectComposer(page, id) {
  const controls = page.locator("header [role=radio]");
  const matches = [];
  for (let index = 0; index < await controls.count(); index += 1) {
    const control = controls.nth(index);
    if ((await control.isVisible().catch(() => false)) && /^chat$/i.test(compact(await control.innerText().catch(() => "")))) matches.push(control);
  }
  if (matches.length !== 1) throw new Error("Unique header Project Chat control is unavailable.");
  await matches[0].click();
  await page.waitForTimeout(750);
  if (!scoped(page.url(), id)) throw new Error("Project binding was not retained after Project Chat.");
  const editor = page.locator("textarea, [contenteditable=true]");
  for (let index = 0; index < await editor.count(); index += 1) if (await editor.nth(index).isVisible().catch(() => false)) return;
  throw new Error("Project Chat produced no blank composer.");
}

async function modelCandidates(page) {
  return page.locator("button, [role=combobox]").evaluateAll((nodes) => {
    const isVisible = (node) => { const box = node.getBoundingClientRect(); return box.width > 0 && box.height > 0; };
    return nodes.filter(isVisible).map((node, index) => {
      const text = (node.innerText || node.textContent || "").replace(/\s+/g, " ").trim().slice(0, 160);
      const aria = (node.getAttribute("aria-label") || "").trim().slice(0, 160);
      const title = (node.getAttribute("title") || "").trim().slice(0, 160);
      const testid = (node.getAttribute("data-testid") || "").trim().slice(0, 120);
      const label = `${text} ${aria} ${title} ${testid}`.trim();
      return { index, tag: node.tagName.toLowerCase(), role: node.getAttribute("role") || "", text, aria, title, testid, modelLike: /\b(?:pro|gpt|model)\b/i.test(label) };
    }).filter((item) => item.modelLike);
  });
}

async function visibleMenuEntries(page) {
  return page.locator("button, [role=option], [role=menuitem], [role=radio], [role=combobox]").evaluateAll((nodes) => {
    const isVisible = (node) => { const box = node.getBoundingClientRect(); return box.width > 0 && box.height > 0; };
    return nodes.filter(isVisible).map((node) => {
      const text = (node.innerText || node.textContent || "").replace(/\s+/g, " ").trim().slice(0, 160);
      const aria = (node.getAttribute("aria-label") || "").trim().slice(0, 160);
      const signal = `${text} ${aria} ${node.getAttribute("title") || ""}`;
      return { tag: node.tagName.toLowerCase(), role: node.getAttribute("role") || "", text, aria, ariaSelected: node.getAttribute("aria-selected"), disabled: node.hasAttribute("disabled") || node.getAttribute("aria-disabled") === "true", relevant: /\b(?:pro|gpt|model|reasoning|thinking|effort|max|high|extended)\b/i.test(signal) };
    }).filter((item) => item.relevant).slice(0, 24);
  });
}

async function main() {
  const request = JSON.parse(fs.readFileSync(0, "utf8"));
  const id = String(request.projectId || "").trim();
  const cdp = /^(https?|wss?):\/\//.test(String(request.cdpUrl || "")) ? request.cdpUrl : `http://${request.cdpUrl}`;
  if (!id || !cdp || request.projectUrl !== projectUrl(id)) throw new Error("Request must identify the exact DUCA Project and runtime CDP endpoint.");
  const receipt = { schema_version: "1.0", dispatch_id: request.dispatchId || null, project: id, operation: "MODEL_PICKER_READONLY_DIAGNOSTIC", runtime_cdp: cdp, completed_at: null, project_bound_composer: false, model_candidates: [], opened_unique_candidate: null, menu_entries: [], outcome: "QUARANTINED_PRECHECK_FAILED", actual_attempt_increased: false };
  let browser; let page;
  try {
    browser = await chromium.connectOverCDP(cdp, { timeout: 30000 });
    const context = await exactContext(browser, id);
    page = await context.newPage();
    await page.goto(request.projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    if (!isExact(page.url(), id)) throw new Error("Navigation left the exact DUCA Project.");
    await waitForProject(page);
    await openBlankProjectComposer(page, id);
    receipt.project_bound_composer = true;
    receipt.model_candidates = await modelCandidates(page);
    if (receipt.model_candidates.length === 1) {
      const candidate = receipt.model_candidates[0];
      const nodes = page.locator("button, [role=combobox]");
      await nodes.nth(candidate.index).click();
      receipt.opened_unique_candidate = candidate;
      await page.waitForTimeout(300);
      receipt.menu_entries = await visibleMenuEntries(page);
      await page.keyboard.press("Escape").catch(() => {});
      receipt.outcome = "MODEL_CONTROL_IDENTIFIED";
    } else {
      receipt.outcome = "MODEL_CONTROL_UNRESOLVED";
    }
  } catch (error) {
    receipt.error = String(error && error.message ? error.message : error);
  } finally {
    receipt.completed_at = now();
    if (page) await page.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    write(request.receiptPath, receipt);
    process.stdout.write(`${JSON.stringify(receipt)}\n`);
  }
}

main().catch((error) => { process.stderr.write(`${JSON.stringify({ ok: false, error: String(error.message || error) })}\n`); process.exitCode = 2; });
