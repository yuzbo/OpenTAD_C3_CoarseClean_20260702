# CPTC TAR32 terminal Pro submission needs attention

- request_id: `PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION-v001`
- nonce: `ZOOMTOKEN-CPTC-TAR32-TERMINAL-PRO-v001-20260829T224000+0800`
- exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- recorded_at: `2026-08-29T22:52:33+08:00`
- actual_submission_count: `0`
- conversation_created: `false`
- attachments_uploaded: `false`
- browser_contacted: `false`

## Objective blocker

The terminal package is complete and pushed at Git commit
`da006dc03f212a9cd2bb3fedc37bc5564d218b12`. Immediately before the required single
fresh Pro turn, dynamic profile discovery failed closed:

- iXBrowser Local API `127.0.0.1:53200` refused the connection;
- the official runtime helper found no verified replacement Local API endpoint;
- no localhost listener exposed a Chrome DevTools `/json/version` endpoint;
- the Windows process list contained no iXBrowser process or `--remote-debugging-port` process;
- Computer Use listed no targetable iXBrowser window.

The only targetable `ChatGPT` window belongs to the Codex desktop app, which the Windows
automation safety contract forbids automating. Therefore no page was refreshed, restarted
or closed; no attachment was uploaded; no Project conversation was created; and no prompt
was sent. The scientific request remains prepared and deduplicated.

## Exact recovery

The user must make iXBrowser profile `61` available again without discarding its login.
On the next continuation, Codex will dynamically re-run the official profile helper,
verify the exact Project and browser-visible Pro route, acquire the profile/project/turn
locks, and submit this same request exactly once. It must not mint another request,
conversation or nonce unless the current submission state becomes uncertain after an
actual remote send.
