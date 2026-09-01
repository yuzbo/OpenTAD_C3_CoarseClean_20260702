"use strict";

const assert = require("assert");
const path = require("path");
const {
  validateRequest,
  normalizeCdpUrl,
  normalizeUiText,
  selectHighestProLabel,
  modelRank,
  compareRank,
} = require("./direct_ixbrowser_project_pro.cjs");

const root = path.resolve(__dirname, "..", "..");
const request = validateRequest({
  action: "fixture",
  cdpUrl: "127.0.0.1:47785",
  projectId: "g-p-6a796fef9a00819194024cf1de3bd697",
  projectUrl: "https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project",
  nonce: "fixture-nonce",
  turnId: "fixture-turn",
  promptPath: path.join(root, ".cvpr-pro-lab", "pro-reviews", "runs", "duca-p0-direct-cdp-prepared", "prompt.md"),
  expectedCommit: "63a726a4aaf48ecbf6780bb196de43a890c6b4df",
  expectedSources: ["CURRENT_RESEARCH_STATE-v005(1).md", "MODEL_EXPERIMENT_HISTORY-v005.md"],
  finalTextPath: path.join(root, ".cvpr-pro-lab", "pro-reviews", "runs", "fixture", "raw-response.md"),
});

assert.strictEqual(request.projectId, "g-p-6a796fef9a00819194024cf1de3bd697");
assert.strictEqual(request.cdpUrl, "http://127.0.0.1:47785");
assert.strictEqual(normalizeCdpUrl("https://127.0.0.1:47785"), "https://127.0.0.1:47785");
assert.strictEqual(normalizeCdpUrl("ws://127.0.0.1:47785/devtools/browser/example"), "ws://127.0.0.1:47785/devtools/browser/example");
assert.strictEqual(normalizeUiText(null), "");
assert.strictEqual(normalizeUiText("   "), "");
assert.strictEqual(
  selectHighestProLabel([null, "", "ordinary button", "GPT-4 Pro", "GPT-5 Pro"]),
  "GPT-5 Pro"
);
assert.strictEqual(selectHighestProLabel([null, "", "ordinary button"]), null);
assert.ok(compareRank(modelRank("GPT-5 Pro"), modelRank("GPT-4 Pro")) < 0);
assert.throws(() => validateRequest({ ...request, projectUrl: "https://chatgpt.com/" }), /exact/);
console.log("direct_ixbrowser_project_pro fixture validation passed");
