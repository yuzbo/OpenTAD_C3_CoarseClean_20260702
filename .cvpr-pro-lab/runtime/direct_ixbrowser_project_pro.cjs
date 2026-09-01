#!/usr/bin/env node
"use strict";

// One bounded, direct-iXBrowser Project turn.  It deliberately has no retry,
// scheduler, or fallback backend: a central lease invokes it once.
const fs = require("fs");
const os = require("os");
const path = require("path");

function playwrightSearchPaths() {
  const candidates = [];
  if (process.env.NODE_PATH) candidates.push(...process.env.NODE_PATH.split(path.delimiter));
  candidates.push(path.resolve(path.dirname(process.execPath), "..", "node_modules"));
  const cache = path.join(os.homedir(), ".cache", "codex-runtimes");
  if (fs.existsSync(cache)) {
    for (const runtime of fs.readdirSync(cache).sort().reverse()) {
      candidates.push(path.join(cache, runtime, "dependencies", "node", "node_modules"));
    }
  }
  return [...new Set(candidates.filter((candidate) => fs.existsSync(candidate)))];
}

function resolvePlaywright() {
  for (const name of ["playwright", "playwright-core"]) {
    try {
      return { name, modulePath: require.resolve(name), api: require(name) };
    } catch (_) {
      // Match the Sources broker bootstrap when the current Node has no NODE_PATH.
    }
    for (const nodeModules of playwrightSearchPaths()) {
      const packagePath = path.join(nodeModules, name);
      if (fs.existsSync(path.join(packagePath, "package.json"))) {
        return { name, modulePath: require.resolve(packagePath), api: require(packagePath) };
      }
    }
  }
  throw new Error("Playwright/playwright-core is unavailable for direct CDP.");
}

function readRequest() {
  const raw = fs.readFileSync(0, "utf8");
  return JSON.parse(raw);
}

function isoNow() {
  return new Date().toISOString();
}

function projectPath(projectId) {
  return `/g/${projectId}/project`;
}

function pathMatchesProject(rawUrl, projectId) {
  try {
    return new URL(rawUrl).pathname === projectPath(projectId);
  } catch (_) {
    return false;
  }
}

function sourcesUrl(projectUrl) {
  const url = new URL(projectUrl);
  url.searchParams.set("tab", "sources");
  return url.toString();
}

function requireText(request, key) {
  const value = request[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Missing non-empty request field: ${key}`);
  }
  return value.trim();
}

function normalizeCdpUrl(value) {
  if (/^127\.0\.0\.1:\d+$/.test(value)) return `http://${value}`;
  let url;
  try {
    url = new URL(value);
  } catch (_) {
    throw new Error("cdpUrl must be localhost host:port or a http(s)/ws(s) URL.");
  }
  if (!/^(http:|https:|ws:|wss:)$/.test(url.protocol) || !url.hostname || !url.port) {
    throw new Error("cdpUrl must be localhost host:port or a http(s)/ws(s) URL.");
  }
  return value;
}

function validateRequest(request) {
  if (!request || typeof request !== "object") {
    throw new Error("Expected one JSON request object on stdin.");
  }
  const cdpUrl = requireText(request, "cdpUrl");
  const projectId = requireText(request, "projectId");
  const projectUrl = requireText(request, "projectUrl");
  const nonce = requireText(request, "nonce");
  const turnId = requireText(request, "turnId");
  const promptPath = requireText(request, "promptPath");
  const expectedCommit = requireText(request, "expectedCommit");
  const finalTextPath = requireText(request, "finalTextPath");
  const expectedSources = Array.isArray(request.expectedSources) ? request.expectedSources : [];

  const normalizedCdpUrl = normalizeCdpUrl(cdpUrl);
  if (!/^[0-9a-f]{40}$/i.test(expectedCommit)) {
    throw new Error("expectedCommit must be an exact 40-hex Git revision.");
  }
  if (!fs.statSync(promptPath).isFile()) {
    throw new Error(`Prompt file is unavailable: ${promptPath}`);
  }
  if (!expectedSources.length || expectedSources.some((name) => typeof name !== "string" || !name)) {
    throw new Error("expectedSources must name the confirmed Sources required for this turn.");
  }
  const url = new URL(projectUrl);
  if (url.origin !== "https://chatgpt.com" || url.pathname !== projectPath(projectId)) {
    throw new Error("projectUrl must be the exact https://chatgpt.com/g/<project-id>/project URL.");
  }
  return {
    ...request,
    cdpUrl: normalizedCdpUrl,
    projectId,
    projectUrl: url.toString(),
    nonce,
    turnId,
    promptPath,
    expectedCommit: expectedCommit.toLowerCase(),
    finalTextPath,
    expectedSources: [...new Set(expectedSources)],
    timeoutMs: Number(request.timeoutMs || 90000),
  };
}

function writeReceipt(receiptPath, receipt) {
  if (!receiptPath) return;
  fs.mkdirSync(path.dirname(receiptPath), { recursive: true });
  fs.writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
}

async function chooseContext(browser, projectId) {
  const contexts = browser.contexts();
  const matches = contexts.filter((context) =>
    context.pages().some((page) => pathMatchesProject(page.url(), projectId))
  );
  if (matches.length === 1) return matches[0];
  if (matches.length > 1) {
    throw new Error("More than one CDP context contains the requested Project.");
  }
  if (contexts.length === 1) return contexts[0];
  throw new Error("CDP exposes multiple contexts without one exact Project context.");
}

async function sourceNames(page) {
  const actions = page.getByRole("button", { name: "Source actions", exact: true });
  return actions.evaluateAll((buttons) => {
    const names = [];
    for (const button of buttons) {
      let node = button;
      for (let depth = 0; depth < 7 && node; depth += 1, node = node.parentElement) {
        const text = (node.innerText || "").trim();
        if (/\nFile\s*[·•]/i.test(text)) {
          const name = text.split(/\r?\n/, 1)[0].trim();
          if (name) names.push(name);
          break;
        }
      }
    }
    return [...new Set(names)];
  });
}

async function verifySources(page, request) {
  await page.goto(sourcesUrl(request.projectUrl), {
    waitUntil: "domcontentloaded",
    timeout: request.timeoutMs,
  });
  if (!pathMatchesProject(page.url(), request.projectId)) {
    throw new Error(`Sources route left the expected Project: ${page.url()}`);
  }
  await page.getByRole("button", { name: "Add sources", exact: true }).waitFor({
    state: "visible",
    timeout: request.timeoutMs,
  });
  await page.waitForTimeout(1000);
  const names = await sourceNames(page);
  const missing = request.expectedSources.filter((name) => !names.includes(name));
  if (missing.length) {
    throw new Error(`Required Project Sources are not visible: ${missing.join(", ")}`);
  }
  return names;
}

function modelRank(label) {
  const lower = label.toLowerCase();
  const version = lower.match(/gpt[- ]?(\d+)(?:\.(\d+))?/i);
  return [
    /pro\b/i.test(label) ? 1 : 0,
    version ? Number(version[1]) : -1,
    version && version[2] ? Number(version[2]) : -1,
  ];
}

function compareRank(a, b) {
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return b[i] - a[i];
  }
  return 0;
}

function normalizeUiText(value) {
  return String(value || "").trim();
}

function selectHighestProLabel(labels) {
  const options = labels
    .map((label) => normalizeUiText(label))
    .filter((label) => label && /\bpro\b/i.test(label));
  if (!options.length) return null;
  options.sort((left, right) => compareRank(modelRank(left), modelRank(right)));
  return options[0];
}

async function chooseHighestPro(page, timeoutMs) {
  const buttons = page.locator("button");
  const count = await buttons.count();
  let picker = null;
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    const text = normalizeUiText(await button.innerText().catch(() => ""));
    const aria = normalizeUiText(await button.getAttribute("aria-label").catch(() => ""));
    if (/\b(gpt|model)\b/i.test(`${text} ${aria}`)) {
      picker = button;
      break;
    }
  }
  if (!picker) throw new Error("The Project UI exposed no identifiable model picker.");
  await picker.click();

  const candidates = page.locator('[role="option"], [role="menuitem"]');
  await candidates.first().waitFor({ state: "visible", timeout: timeoutMs });
  const options = [];
  const optionCount = await candidates.count();
  for (let index = 0; index < optionCount; index += 1) {
    const candidate = candidates.nth(index);
    const label = normalizeUiText(await candidate.innerText().catch(() => ""));
    if (label && /\bpro\b/i.test(label)) options.push({ label, candidate });
  }
  if (!options.length) throw new Error("The visible model picker exposed no verifiable Pro tier.");
  const selectedLabel = selectHighestProLabel(options.map((option) => option.label));
  const selected = options.find((option) => option.label === selectedLabel);
  if (!selected) throw new Error("The highest visible Pro tier could not be selected safely.");
  await selected.candidate.click();
  return selected.label;
}

async function chooseMaximumEffortIfExposed(page, timeoutMs) {
  const buttons = page.locator("button");
  const count = await buttons.count();
  let picker = null;
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    const text = normalizeUiText(await button.innerText().catch(() => ""));
    const aria = normalizeUiText(await button.getAttribute("aria-label").catch(() => ""));
    if (/reasoning|thinking|effort/i.test(`${text} ${aria}`)) {
      picker = button;
      break;
    }
  }
  if (!picker) return "MAX_EFFORT_NOT_SEPARATELY_EXPOSED";
  await picker.click();
  const options = page.locator('[role="option"], [role="menuitem"]');
  await options.first().waitFor({ state: "visible", timeout: timeoutMs });
  const preference = [/\bmax\b/i, /extended/i, /\bhigh\b/i];
  for (const pattern of preference) {
    const optionCount = await options.count();
    for (let index = 0; index < optionCount; index += 1) {
      const option = options.nth(index);
      const label = normalizeUiText(await option.innerText().catch(() => ""));
      if (pattern.test(label)) {
        await option.click();
        return label;
      }
    }
  }
  throw new Error("Reasoning-effort control is exposed but no maximum level is identifiable.");
}

async function findComposer(page) {
  const textarea = page.locator("textarea");
  if (await textarea.count()) return textarea.last();
  const editable = page.locator('[contenteditable="true"]');
  if (await editable.count()) return editable.last();
  throw new Error("The new Project conversation has no writable composer.");
}

async function findSendButton(page) {
  const buttons = page.getByRole("button", { name: /send|submit/i });
  const count = await buttons.count();
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    if (await button.isEnabled().catch(() => false)) return button;
  }
  throw new Error("The new Project conversation has no enabled send control.");
}

async function submit(request) {
  const playwright = resolvePlaywright();
  const { chromium } = playwright.api;
  const prompt = fs.readFileSync(request.promptPath, "utf8");
  let browser;
  let page;
  let submitted = false;
  const receipt = {
    schema_version: "1.0",
    operation: "PRO_REVIEW",
    status: "PRE_SUBMISSION",
    expectedProjectId: request.projectId,
    expectedProjectUrl: request.projectUrl,
    expectedCommit: request.expectedCommit,
    expectedSources: request.expectedSources,
    turnId: request.turnId,
    nonce: request.nonce,
    requestedPromptPath: request.promptPath,
    finalTextPath: request.finalTextPath,
    runtimeCdp: request.cdpUrl,
    playwrightModule: playwright.modulePath,
    submitted_at: null,
    last_output_at: null,
    stream_state: "NOT_SUBMITTED",
    hard_timeout_at: null,
  };
  try {
    browser = await chromium.connectOverCDP(request.cdpUrl, { timeout: request.timeoutMs });
    const context = await chooseContext(browser, request.projectId);
    page = await context.newPage();

    const visibleSources = await verifySources(page, request);
    await page.goto(request.projectUrl, { waitUntil: "domcontentloaded", timeout: request.timeoutMs });
    if (!pathMatchesProject(page.url(), request.projectId)) {
      throw new Error(`Fresh-conversation route left the expected Project: ${page.url()}`);
    }
    const selectedModel = await chooseHighestPro(page, request.timeoutMs);
    const selectedEffort = await chooseMaximumEffortIfExposed(page, request.timeoutMs);
    const composer = await findComposer(page);
    await composer.fill(prompt);
    const send = await findSendButton(page);
    await send.click();
    submitted = true;
    const submittedAt = isoNow();
    await page.waitForTimeout(750);

    receipt.status = "SUBMITTED";
    receipt.projectUrl = page.url();
    receipt.projectTitle = await page.title();
    receipt.visibleSources = visibleSources;
    receipt.selectedModel = selectedModel;
    receipt.selectedEffort = selectedEffort;
    receipt.submitted_at = submittedAt;
    receipt.last_output_at = submittedAt;
    receipt.stream_state = "ACTIVE";
    receipt.hard_timeout_at = new Date(Date.parse(submittedAt) + 120 * 60 * 1000).toISOString();
    receipt.conversationId = new URL(page.url()).pathname.match(/\/c\/([^/?#]+)/)?.[1] || null;
    writeReceipt(request.receiptPath, receipt);
    return receipt;
  } catch (error) {
    receipt.status = submitted ? "UNKNOWN_SUBMISSION_STATE" : "PRE_SUBMISSION_FAILED";
    receipt.stream_state = submitted ? "UNKNOWN" : "NOT_SUBMITTED";
    receipt.error = String(error && error.message ? error.message : error);
    writeReceipt(request.receiptPath, receipt);
    throw Object.assign(new Error(receipt.error), { receipt });
  } finally {
    if (page && !submitted) await page.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
  }
}

async function main() {
  const request = validateRequest(readRequest());
  if (request.action === "fixture") {
    process.stdout.write(`${JSON.stringify({ ok: true, action: "fixture", turnId: request.turnId, nonce: request.nonce })}\n`);
    return;
  }
  if (request.action === "require-resolution") {
    const playwright = resolvePlaywright();
    if (!playwright.api || !playwright.api.chromium || typeof playwright.api.chromium.connectOverCDP !== "function") {
      throw new Error("Resolved Playwright package does not export chromium.connectOverCDP.");
    }
    process.stdout.write(`${JSON.stringify({
      ok: true,
      action: "require-resolution",
      module: playwright.name,
      modulePath: playwright.modulePath,
      connectOverCDP: true,
    })}\n`);
    return;
  }
  if (request.action && request.action !== "submit") {
    throw new Error(`Unsupported action: ${request.action}`);
  }
  process.stdout.write(`${JSON.stringify(await submit(request))}\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${JSON.stringify({ ok: false, error: String(error.message || error), receipt: error.receipt || null })}\n`);
    process.exitCode = 2;
  });
}

module.exports = {
  validateRequest,
  normalizeCdpUrl,
  normalizeUiText,
  selectHighestProLabel,
  modelRank,
  compareRank,
  projectPath,
};
