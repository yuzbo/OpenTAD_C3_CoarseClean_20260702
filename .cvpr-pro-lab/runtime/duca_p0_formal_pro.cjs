#!/usr/bin/env node
"use strict";

// Exact DUCA P0 lease runner. It deliberately has one project route, one prompt,
// one submission, and one attached monitor; it is not a general browser backend.
const fs = require("fs");
const os = require("os");
const path = require("path");

function resolvePlaywright() {
  const candidates = [];
  if (process.env.NODE_PATH) candidates.push(...process.env.NODE_PATH.split(path.delimiter));
  candidates.push("C:/Users/skywalker/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules");
  const cache = path.join(os.homedir(), ".cache", "codex-runtimes");
  if (fs.existsSync(cache)) for (const runtime of fs.readdirSync(cache).sort().reverse()) candidates.push(path.join(cache, runtime, "dependencies", "node", "node_modules"));
  for (const name of ["playwright", "playwright-core"]) {
    try { return { modulePath: require.resolve(name), chromium: require(name).chromium }; } catch (_) {}
    for (const modules of candidates) {
      const packagePath = path.join(modules, name);
      if (fs.existsSync(path.join(packagePath, "package.json"))) return { modulePath: require.resolve(packagePath), chromium: require(packagePath).chromium };
    }
  }
  throw new Error("Playwright is unavailable for the approved direct-CDP route.");
}

function isoNow() { return new Date().toISOString(); }
function normalizedCdp(value) { return /^(https?|wss?):\/\//.test(value) ? value : `http://${value}`; }
function projectPath(id) { return `/g/${id}/project`; }
function exactProject(url, id) { try { const parsed = new URL(url); return parsed.origin === "https://chatgpt.com" && parsed.pathname === projectPath(id); } catch (_) { return false; } }
function projectScoped(url, id) { try { const parsed = new URL(url); return parsed.origin === "https://chatgpt.com" && parsed.pathname.startsWith(`/g/${id}/`); } catch (_) { return false; } }
function ui(value) { return String(value || "").replace(/\s+/g, " ").trim(); }
function writeJson(file, value) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

function readRequest() {
  const request = JSON.parse(fs.readFileSync(0, "utf8"));
  for (const key of ["cdpUrl", "projectId", "projectUrl", "promptPath", "turnId", "nonce", "receiptPath", "rawResponsePath", "hardTimeoutAt"]) {
    if (typeof request[key] !== "string" || !request[key].trim()) throw new Error(`Missing ${key}`);
  }
  if (!exactProject(request.projectUrl, request.projectId)) throw new Error("Request project URL is not the exact Project root.");
  if (!fs.existsSync(request.promptPath)) throw new Error("Formal prompt is unavailable.");
  return { ...request, cdpUrl: normalizedCdp(request.cdpUrl), monitorIntervalMs: Number(request.monitorIntervalMs || 5000) };
}

async function chooseContext(browser, projectId) {
  const matches = browser.contexts().filter((context) => context.pages().some((page) => projectScoped(page.url(), projectId)));
  if (matches.length === 1) return matches[0];
  if (matches.length > 1 || browser.contexts().length !== 1) throw new Error("CDP did not expose one unambiguous Project context.");
  return browser.contexts()[0];
}

async function waitForProject(page) {
  for (let elapsed = 0; elapsed < 45000; elapsed += 1000) {
    const text = await page.locator("main").first().innerText().catch(() => "");
    if (["DUCA", "Chats", "Sources"].every((token) => text.includes(token))) return;
    await page.waitForTimeout(1000);
  }
  throw new Error("DUCA Project did not reach its verified rendered state.");
}

async function openFreshProjectComposer(page, projectId) {
  const radios = page.locator("header [role=radio]");
  const matches = [];
  for (let index = 0; index < await radios.count(); index += 1) {
    const radio = radios.nth(index);
    if (!(await radio.isVisible().catch(() => false))) continue;
    if (/^chat$/i.test(ui(await radio.innerText().catch(() => "")))) matches.push(radio);
  }
  if (matches.length !== 1) throw new Error("The verified Project-internal Chat control is not uniquely available.");
  await matches[0].click();
  await page.waitForTimeout(750);
  if (!projectScoped(page.url(), projectId)) throw new Error("Project Chat control lost exact Project binding.");
  const composer = page.locator("textarea, [contenteditable=true]");
  for (let index = 0; index < await composer.count(); index += 1) if (await composer.nth(index).isVisible().catch(() => false)) return composer.nth(index);
  throw new Error("Project Chat control did not produce a visible composer.");
}

function rankModel(label) {
  const version = label.match(/gpt[- ]?(\d+)(?:\.(\d+))?/i);
  return [/\bpro\b/i.test(label) ? 1 : 0, version ? Number(version[1]) : -1, version && version[2] ? Number(version[2]) : -1];
}
function compareModels(left, right) { const a = rankModel(left); const b = rankModel(right); for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return b[i] - a[i]; return 0; }

async function chooseHighestPro(page) {
  const selectors = '[data-testid="model-switcher-dropdown-button"], button.__composer-pill[aria-haspopup="menu"], button[aria-haspopup="menu"], button';
  const candidates = page.locator(selectors);
  const picker = [];
  for (let index = 0; index < await candidates.count(); index += 1) {
    const node = candidates.nth(index);
    if (!(await node.isVisible().catch(() => false))) continue;
    const label = ui(`${await node.innerText().catch(() => "")} ${await node.getAttribute("aria-label").catch(() => "")} ${await node.getAttribute("title").catch(() => "")}`);
    const testid = ui(await node.getAttribute("data-testid").catch(() => ""));
    if (testid === "model-switcher-dropdown-button" || /\b(?:model|gpt|pro)\b/i.test(label)) picker.push({ node, label, testid });
  }
  const exactPro = picker.filter((entry) => /^pro$/i.test(entry.label));
  const selectedPicker = exactPro.length === 1 ? exactPro[0] : picker.length === 1 ? picker[0] : null;
  if (!selectedPicker) throw new Error("No unique composer-adjacent Pro/model picker is browser-verifiable.");
  await selectedPicker.node.click();
  await page.waitForTimeout(300);
  const optionNodes = page.locator('[role=option], [role=menuitem], [role=radio], [data-testid*="model"]');
  const choices = [];
  for (let index = 0; index < await optionNodes.count(); index += 1) {
    const node = optionNodes.nth(index);
    if (!(await node.isVisible().catch(() => false))) continue;
    const label = ui(`${await node.innerText().catch(() => "")} ${await node.getAttribute("aria-label").catch(() => "")}`);
    if (/\bpro\b/i.test(label)) choices.push({ node, label });
  }
  if (!choices.length && /\bpro\b/i.test(selectedPicker.label)) {
    await page.keyboard.press("Escape").catch(() => {});
    return selectedPicker.label;
  }
  if (!choices.length) throw new Error("Opened model picker exposed no verifiable Pro tier.");
  choices.sort((left, right) => compareModels(left.label, right.label));
  await choices[0].node.click();
  return choices[0].label;
}

async function chooseMaxEffort(page) {
  const buttons = page.locator("button");
  let picker = null;
  for (let index = 0; index < await buttons.count(); index += 1) {
    const node = buttons.nth(index);
    if (!(await node.isVisible().catch(() => false))) continue;
    const label = ui(`${await node.innerText().catch(() => "")} ${await node.getAttribute("aria-label").catch(() => "")} ${await node.getAttribute("title").catch(() => "")}`);
    if (/reasoning|thinking|effort/i.test(label)) { picker = node; break; }
  }
  if (!picker) return "MAX_EFFORT_NOT_SEPARATELY_EXPOSED";
  await picker.click();
  for (const needle of [/\bmax\b/i, /extended/i, /\bhigh\b/i]) {
    const nodes = page.locator('[role=option], [role=menuitem], [role=radio]');
    for (let index = 0; index < await nodes.count(); index += 1) {
      const node = nodes.nth(index);
      const label = ui(await node.innerText().catch(() => ""));
      if ((await node.isVisible().catch(() => false)) && needle.test(label)) { await node.click(); return label; }
    }
  }
  throw new Error("An effort control is exposed but its maximum level is not verifiable.");
}

async function sendPrompt(page, composer, prompt) {
  await composer.fill(prompt);
  const direct = page.locator('[data-testid="send-button"], button[aria-label^="Send"]');
  for (let index = 0; index < await direct.count(); index += 1) if (await direct.nth(index).isVisible().catch(() => false) && await direct.nth(index).isEnabled().catch(() => false)) { await direct.nth(index).click(); return; }
  const buttons = page.getByRole("button", { name: /send|submit/i });
  for (let index = 0; index < await buttons.count(); index += 1) if (await buttons.nth(index).isVisible().catch(() => false) && await buttons.nth(index).isEnabled().catch(() => false)) { await buttons.nth(index).click(); return; }
  throw new Error("The Project-bound composer has no enabled send control.");
}

async function assistantText(page) {
  const nodes = page.locator('[data-message-author-role="assistant"]');
  const count = await nodes.count();
  if (!count) return "";
  return ui(await nodes.nth(count - 1).innerText().catch(() => ""));
}

async function streamActive(page) {
  const stop = page.locator('[data-testid*="stop"], button[aria-label*="Stop" i]');
  for (let index = 0; index < await stop.count(); index += 1) if (await stop.nth(index).isVisible().catch(() => false)) return true;
  return false;
}

async function main() {
  const request = readRequest();
  const prompt = fs.readFileSync(request.promptPath, "utf8");
  const receipt = {
    schema_version: "1.0", dispatch_id: request.dispatchId, request_id: request.requestId,
    project: request.projectId, expected_project_url: request.projectUrl, turn_id: request.turnId,
    nonce: request.nonce, expected_commit: request.expectedCommit, expected_sources: request.expectedSources,
    runtime_cdp: request.cdpUrl, submitted_at: null, completed_at: null, conversation_id: null,
    selected_model: null, selected_effort: null, stream_state: "NOT_SUBMITTED", last_output_at: null,
    hard_timeout_at: request.hardTimeoutAt, raw_response_path: request.rawResponsePath, outcome: "PRE_SUBMISSION_FAILED"
  };
  let browser; let page; let submitted = false;
  try {
    const pw = resolvePlaywright(); receipt.playwright_module = pw.modulePath;
    browser = await pw.chromium.connectOverCDP(request.cdpUrl, { timeout: 30000 });
    const context = await chooseContext(browser, request.projectId);
    page = await context.newPage();
    await page.goto(request.projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    if (!exactProject(page.url(), request.projectId)) throw new Error("Navigation did not reach the exact DUCA Project.");
    await waitForProject(page);
    const composer = await openFreshProjectComposer(page, request.projectId);
    receipt.selected_model = await chooseHighestPro(page);
    receipt.selected_effort = await chooseMaxEffort(page);
    await sendPrompt(page, composer, prompt);
    submitted = true;
    receipt.submitted_at = isoNow(); receipt.last_output_at = receipt.submitted_at; receipt.stream_state = "ACTIVE";
    await page.waitForTimeout(750);
    receipt.project_url_after_submit = page.url();
    if (!projectScoped(receipt.project_url_after_submit, request.projectId)) throw new Error("Submission lost Project binding.");
    receipt.conversation_id = new URL(page.url()).pathname.match(/\/c\/([^/?#]+)/)?.[1] || null;
    receipt.outcome = "SUBMITTED";
    writeJson(request.receiptPath, receipt);
    process.stdout.write(`${JSON.stringify({ event: "submitted", ...receipt })}\n`);
    const deadline = Date.parse(request.hardTimeoutAt);
    let observed = "";
    while (Date.now() < deadline) {
      await page.waitForTimeout(request.monitorIntervalMs);
      const text = await assistantText(page);
      if (text && text !== observed) { observed = text; receipt.last_output_at = isoNow(); }
      if (text && !(await streamActive(page))) {
        fs.mkdirSync(path.dirname(request.rawResponsePath), { recursive: true });
        fs.writeFileSync(request.rawResponsePath, `${text}\n`, "utf8");
        receipt.completed_at = isoNow(); receipt.stream_state = "FINAL"; receipt.outcome = "COMPLETED";
        writeJson(request.receiptPath, receipt);
        process.stdout.write(`${JSON.stringify({ event: "final", ...receipt })}\n`);
        return;
      }
    }
    receipt.completed_at = isoNow(); receipt.stream_state = "HARD_TIMEOUT"; receipt.outcome = "HARD_TIMEOUT";
    writeJson(request.receiptPath, receipt);
    process.stdout.write(`${JSON.stringify({ event: "hard_timeout", ...receipt })}\n`);
  } catch (error) {
    receipt.completed_at = isoNow();
    receipt.outcome = submitted ? "UNKNOWN_SUBMISSION_STATE" : "PRE_SUBMISSION_FAILED";
    receipt.stream_state = submitted ? "UNKNOWN" : "NOT_SUBMITTED";
    receipt.error = String(error && error.message ? error.message : error);
    writeJson(request.receiptPath, receipt);
    process.stderr.write(`${JSON.stringify({ event: "error", ...receipt })}\n`);
    process.exitCode = 2;
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
}

main();
