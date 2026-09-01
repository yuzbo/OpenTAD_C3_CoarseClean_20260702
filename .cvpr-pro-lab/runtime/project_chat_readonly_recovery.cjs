#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const PLAYWRIGHT_NODE_MODULES = "C:/Users/skywalker/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
module.paths.unshift(PLAYWRIGHT_NODE_MODULES);
const { chromium } = require("playwright");

function readRequest() {
  return JSON.parse(fs.readFileSync(0, "utf8"));
}

function isoNow() {
  return new Date().toISOString();
}

function projectPrefix(projectId) {
  return `/g/${projectId}/`;
}

function isExactProjectUrl(rawUrl, projectId) {
  try {
    const url = new URL(rawUrl);
    return url.origin === "https://chatgpt.com" && url.pathname === `${projectPrefix(projectId)}project`;
  } catch (_) {
    return false;
  }
}

function remainsProjectScoped(rawUrl, projectId) {
  try {
    const url = new URL(rawUrl);
    return url.origin === "https://chatgpt.com" && url.pathname.startsWith(projectPrefix(projectId));
  } catch (_) {
    return false;
  }
}

async function waitForProjectSpa(page) {
  for (let elapsed = 0; elapsed < 45000; elapsed += 1000) {
    const mainText = (await page.locator("main").first().innerText().catch(() => "")) || "";
    if (["DUCA", "Chats", "Sources"].every((token) => mainText.includes(token))) return true;
    await page.waitForTimeout(1000);
  }
  return false;
}

function isProjectScopedTarget(value, projectId) {
  if (!value) return false;
  try {
    const parsed = new URL(value, "https://chatgpt.com");
    return parsed.origin === "https://chatgpt.com" && parsed.pathname.startsWith(projectPrefix(projectId));
  } catch (_) {
    return false;
  }
}

async function visibleChatCandidates(page, projectId) {
  return page.locator("header [role=radio]").evaluateAll((elements, id) => {
    const prefix = `/g/${id}/`;
    const visible = (element) => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    };
    const projectScoped = (raw) => {
      if (!raw) return false;
      try {
        const url = new URL(raw, location.origin);
        return url.origin === location.origin && url.pathname.startsWith(prefix);
      } catch (_) {
        return false;
      }
    };
    return elements
      .filter(visible)
      .map((element, index) => {
        const text = (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim().slice(0, 96);
        const aria = (element.getAttribute("aria-label") || "").trim().slice(0, 96);
        const href = element.getAttribute("href") || element.getAttribute("data-href") || element.getAttribute("data-url") || "";
        const testid = element.getAttribute("data-testid") || "";
        const owner = element.closest("[data-project-id]");
        const ownerProjectId = owner?.getAttribute("data-project-id") || "";
        const label = `${text} ${aria}`.trim();
        const chatLike = /\b(?:new|start)(?:\s+a)?\s+chat(?:ting)?\b|\bchat\b/i.test(label);
        const targetProjectScoped = projectScoped(href);
        const projectMarker = ownerProjectId === id || /project.*chat|chat.*project/i.test(testid);
        const projectChatMode = element.getAttribute("role") === "radio" && /^chat$/i.test(text);
        return {
          index,
          role: element.tagName.toLowerCase(),
          text,
          aria,
          href,
          testid,
          chatLike,
          targetProjectScoped,
          projectMarker,
          safe: chatLike && (targetProjectScoped || projectMarker || projectChatMode),
        };
      })
      .filter((item) => item.chatLike || /project/i.test(`${item.text} ${item.aria} ${item.testid}`))
      .slice(0, 16);
  }, projectId);
}

async function findUniqueProjectChatControl(page, projectId) {
  const candidates = await visibleChatCandidates(page, projectId);
  const safe = candidates.filter((candidate) => candidate.safe);
  if (safe.length !== 1) return { candidates, control: null };
  return { candidates, control: page.locator("header [role=radio]").nth(safe[0].index) };
}

async function blankComposer(page) {
  const candidates = page.locator("textarea, [contenteditable=\"true\"]");
  const count = await candidates.count();
  for (let index = 0; index < count; index += 1) {
    const candidate = candidates.nth(index);
    if (!(await candidate.isVisible().catch(() => false))) continue;
    const tagName = await candidate.evaluate((element) => element.tagName.toLowerCase());
    const value = tagName === "textarea"
      ? await candidate.inputValue().catch(() => null)
      : await candidate.textContent().catch(() => null);
    if (String(value || "").trim() === "") return true;
  }
  return false;
}

async function projectNavigationEvidence(page, projectId) {
  const prefix = projectPrefix(projectId);
  const candidates = page.locator(`a[href^="${prefix}"], [data-href^="${prefix}"], [data-url^="${prefix}"]`);
  const count = await candidates.count();
  for (let index = 0; index < count; index += 1) {
    if (await candidates.nth(index).isVisible().catch(() => false)) return true;
  }
  return false;
}

async function chooseContext(browser, projectId) {
  const contexts = browser.contexts();
  const exactMatches = contexts.filter((context) =>
    context.pages().some((page) => remainsProjectScoped(page.url(), projectId))
  );
  if (exactMatches.length === 1) return exactMatches[0];
  if (exactMatches.length > 1 || contexts.length !== 1) {
    throw new Error("Unable to bind the recovery to one exact iXBrowser Project context.");
  }
  return contexts[0];
}

async function main() {
  const request = readRequest();
  const projectId = String(request.projectId || "").trim();
  const projectUrl = String(request.projectUrl || "").trim();
  const cdpUrl = String(request.cdpUrl || "").trim();
  const expectedTitle = String(request.expectedTitle || "DUCA").trim();
  if (!projectId || !isExactProjectUrl(projectUrl, projectId) || !cdpUrl) {
    throw new Error("Request must supply an exact Project URL, project ID, and runtime CDP endpoint.");
  }
  const runtimeCdp = /^(https?|wss?):\/\//.test(cdpUrl) ? cdpUrl : `http://${cdpUrl}`;
  const receipt = {
    schema_version: "1.0",
    dispatch_id: request.dispatchId || null,
    project: projectId,
    operation: "PROJECT_CHAT_READONLY_RECOVERY",
    runtime_cdp: runtimeCdp,
    completed_at: null,
    projectChatControlFound: false,
    projectBoundFreshComposer: false,
    before_url: null,
    after_url: null,
    before_title: null,
    after_title: null,
    title_warning: null,
    project_context_evidence: "none",
    dom_candidates: [],
    fresh_conversation_id: null,
    outcome: "QUARANTINED_PRECHECK_FAILED",
    actual_attempt_increased: false,
  };
  let browser;
  let page;
  try {
    browser = await chromium.connectOverCDP(runtimeCdp, { timeout: 30000 });
    const context = await chooseContext(browser, projectId);
    page = await context.newPage();
    await page.goto(projectUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    receipt.before_url = page.url();
    receipt.before_title = await page.title();
    if (!isExactProjectUrl(receipt.before_url, projectId)) {
      throw new Error("The opened page did not prove the exact DUCA Project URL.");
    }
    if (!receipt.before_title.includes(expectedTitle)) receipt.title_warning = "document.title did not expose the expected Project title";
    if (!(await waitForProjectSpa(page))) {
      receipt.outcome = "QUARANTINED_POSTRENDER_SIGNALS_MISSING";
      return receipt;
    }
    const chatLookup = await findUniqueProjectChatControl(page, projectId);
    receipt.dom_candidates = chatLookup.candidates;
    const chat = chatLookup.control;
    receipt.projectChatControlFound = Boolean(chat);
    if (!chat) {
      receipt.outcome = "QUARANTINED_PROJECT_CHAT_CONTROL_UNRESOLVED";
      return receipt;
    }
    await chat.click();
    await page.waitForTimeout(1000);
    receipt.after_url = page.url();
    receipt.after_title = await page.title();
    receipt.fresh_conversation_id = new URL(receipt.after_url).pathname.match(/\/c\/([^/?#]+)/)?.[1] || null;
    const urlEvidence = remainsProjectScoped(receipt.after_url, projectId);
    const navigationEvidence = await projectNavigationEvidence(page, projectId);
    receipt.project_context_evidence = urlEvidence ? "project-scoped-url" : navigationEvidence ? "project-scoped-navigation" : "none";
    receipt.projectBoundFreshComposer =
      (urlEvidence || navigationEvidence) &&
      await blankComposer(page);
    receipt.outcome = receipt.projectBoundFreshComposer
      ? "PASS_PROJECT_BOUND_FRESH_COMPOSER"
      : "QUARANTINED_PROJECT_BINDING_NOT_RETAINED";
    return receipt;
  } catch (error) {
    receipt.error = String(error && error.message ? error.message : error);
    return receipt;
  } finally {
    receipt.completed_at = isoNow();
    if (page) await page.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    fs.mkdirSync(path.dirname(request.receiptPath), { recursive: true });
    fs.writeFileSync(request.receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify(receipt)}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${JSON.stringify({ ok: false, error: String(error.message || error) })}\n`);
  process.exitCode = 2;
});
