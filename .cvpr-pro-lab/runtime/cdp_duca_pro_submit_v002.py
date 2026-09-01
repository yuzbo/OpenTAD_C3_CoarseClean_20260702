#!/usr/bin/env python3
"""Inspect or submit one frozen DUCA Pro prompt through an existing CDP tab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request

import websocket


CDP = "127.0.0.1:15359"
PROJECT_ID = "g-p-6a91061f789881918ccd8357ca3d6c92"
PROJECT_URL = f"https://chatgpt.com/g/{PROJECT_ID}/project?tab=chats"
NONCE = "DUCA-GITHUB-WIKI-COMPREHENSIVE-REVIEW-v002-20260831"
PROMPT = Path(
    r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.cvpr-pro-lab\pro-reviews\prompts"
    r"\PRO_DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW-v002.md"
)
VISIBLE_REPORT = Path(
    r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.cvpr-pro-lab\pro-reviews\runs"
    r"\duca-github-wiki-comprehensive-review-v002\visible-report.md"
)


class CDPClient:
    def __init__(self, url: str) -> None:
        self.socket = websocket.create_connection(url, timeout=20, suppress_origin=True)
        self.counter = 0

    def close(self) -> None:
        self.socket.close()

    def call(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        request_id = self.counter
        self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        value = result.get("result", {})
        if value.get("subtype") == "error":
            raise RuntimeError(value.get("description", "browser evaluation failed"))
        return value.get("value")


def targets() -> list[dict]:
    with urllib.request.urlopen(f"http://{CDP}/json/list", timeout=10) as response:
        return [
            target
            for target in json.load(response)
            if target.get("type") == "page" and PROJECT_ID in target.get("url", "")
        ]


def inspect() -> list[dict]:
    observations = []
    for target in targets():
        client = CDPClient(target["webSocketDebuggerUrl"])
        try:
            state = client.evaluate(
                """(() => {
                  const composer = document.querySelector('#prompt-textarea');
                  const text = document.body?.innerText || '';
                  const assistants = [...document.querySelectorAll('[data-message-author-role="assistant"]')];
                  const lastAssistant = assistants.at(-1)?.innerText || '';
                  const buttons = [...document.querySelectorAll('button')]
                    .map(x => (x.innerText || x.getAttribute('aria-label') || '').trim())
                    .filter(Boolean);
                  return {
                    href: location.href,
                    nonceVisible: text.includes(%s),
                    composerPresent: Boolean(composer),
                    composerText: composer ? (composer.innerText || composer.textContent || '') : '',
                    proVisible: buttons.some(x => /^Pro$/i.test(x)),
                    stopVisible: Boolean(document.querySelector('[data-testid="stop-button"]')) ||
                      buttons.some(x => /stop generating|停止生成/i.test(x)),
                    sendVisible: buttons.some(x => /send prompt|发送提示|发送消息/i.test(x)),
                    assistantCount: assistants.length,
                    terminalMarkerVisible: lastAssistant.includes('DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW_READY'),
                    lastAssistantChars: lastAssistant.length,
                    lastAssistantTail: lastAssistant.slice(-1000)
                  };
                })()""" % json.dumps(NONCE)
            )
            observations.append({"target_id": target["id"], "title": target.get("title"), **state})
        finally:
            client.close()
    return observations


def submit(target_id: str) -> dict:
    matching = [target for target in targets() if target.get("id") == target_id]
    if len(matching) != 1:
        raise RuntimeError("requested target is not one unique exact-DUCA Project tab")
    prompt = PROMPT.read_text(encoding="utf-8")
    if NONCE not in prompt:
        raise RuntimeError("frozen prompt does not contain the required nonce")

    client = CDPClient(matching[0]["webSocketDebuggerUrl"])
    try:
        preflight = client.evaluate(
            """(() => {
              const composer = document.querySelector('#prompt-textarea');
              const text = document.body?.innerText || '';
              return {
                href: location.href,
                nonceVisible: text.includes(%s),
                composerPresent: Boolean(composer),
                composerText: composer ? (composer.innerText || composer.textContent || '') : ''
              };
            })()""" % json.dumps(NONCE)
        )
        if PROJECT_ID not in str(preflight.get("href", "")):
            raise RuntimeError("target left the exact DUCA Project")
        if preflight.get("nonceVisible"):
            raise RuntimeError("nonce is already visible; refusing duplicate submission")
        if not preflight.get("composerPresent"):
            raise RuntimeError("fresh composer is not available")
        if str(preflight.get("composerText", "")).strip():
            raise RuntimeError("composer is not blank")

        focused = client.evaluate(
            """(() => {
              const composer = document.querySelector('#prompt-textarea');
              composer.focus();
              return document.activeElement === composer || composer.contains(document.activeElement);
            })()"""
        )
        if not focused:
            raise RuntimeError("could not focus the exact composer")
        client.call("Input.insertText", {"text": prompt})
        inserted = client.evaluate(
            """(() => {
              const composer = document.querySelector('#prompt-textarea');
              const value = composer ? (composer.innerText || composer.textContent || '') : '';
              return {length: value.length, nonce: value.includes(%s)};
            })()""" % json.dumps(NONCE)
        )
        if not inserted.get("nonce") or inserted.get("length", 0) < len(prompt) - 20:
            raise RuntimeError(f"prompt insertion could not be verified: {inserted}")

        sent = client.evaluate(
            """(() => {
              const send = document.querySelector('[data-testid="send-button"]') ||
                [...document.querySelectorAll('button')].find(x =>
                  /send prompt|发送提示|发送消息/i.test(x.getAttribute('aria-label') || x.innerText || ''));
              if (!send || send.disabled) return {clicked: false, reason: 'send unavailable'};
              send.click();
              return {clicked: true};
            })()"""
        )
        if not sent.get("clicked"):
            raise RuntimeError(f"prompt was inserted but not submitted: {sent}")

        deadline = time.monotonic() + 30
        post = None
        while time.monotonic() < deadline:
            time.sleep(1)
            post = client.evaluate(
                """(() => ({
                  href: location.href,
                  nonceVisible: (document.body?.innerText || '').includes(%s),
                  stopVisible: [...document.querySelectorAll('button')].some(x =>
                    /stop generating|停止生成/i.test(x.getAttribute('aria-label') || x.innerText || ''))
                }))()""" % json.dumps(NONCE)
            )
            if "/c/" in str(post.get("href", "")) and post.get("nonceVisible"):
                return {"submitted": True, **post}
        raise RuntimeError(f"submission could not be verified within 30 seconds: {post}")
    finally:
        client.close()


def capture(target_id: str) -> dict:
    matching = [target for target in targets() if target.get("id") == target_id]
    if len(matching) != 1:
        raise RuntimeError("requested target is not one unique exact-DUCA conversation tab")
    client = CDPClient(matching[0]["webSocketDebuggerUrl"])
    try:
        result = client.evaluate(
            """(() => {
              const assistants = [...document.querySelectorAll('[data-message-author-role="assistant"]')];
              const response = assistants.at(-1)?.innerText || '';
              return {
                href: location.href,
                nonceVisible: (document.body?.innerText || '').includes(%s),
                response
              };
            })()""" % json.dumps(NONCE)
        )
        response = str(result.get("response", ""))
        if PROJECT_ID not in str(result.get("href", "")) or "/c/" not in str(result.get("href", "")):
            raise RuntimeError("target is not the exact DUCA conversation")
        if not result.get("nonceVisible"):
            raise RuntimeError("nonce is not visible in the exact conversation")
        if not response.rstrip().endswith("DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW_READY"):
            raise RuntimeError("assistant response is not terminal")
        VISIBLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        VISIBLE_REPORT.write_text(response.rstrip() + "\n", encoding="utf-8")
        return {"captured": True, "href": result["href"], "response_chars": len(response)}
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "submit", "capture"))
    parser.add_argument("--target-id")
    args = parser.parse_args()
    if args.mode == "inspect":
        print(json.dumps(inspect(), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "capture":
        if not args.target_id:
            parser.error("--target-id is required for capture")
        print(json.dumps(capture(args.target_id), ensure_ascii=False, indent=2))
        return 0
    if not args.target_id:
        parser.error("--target-id is required for submit")
    print(json.dumps(submit(args.target_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
