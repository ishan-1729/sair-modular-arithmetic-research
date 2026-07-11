---
name: playwright-chromium-testing
description: Use repository-installed Playwright with its managed Chromium for deterministic local browser automation, end-to-end tests, visual QA, screenshots, traces, and browser diagnostics. Invoke only through an explicit $playwright-chromium-testing request; do not auto-activate for general web research, UI work, an existing signed-in browser session, external-account automation, or a generic mention of Playwright or Chromium.
---

# Playwright Chromium Testing

Use Playwright directly from the target repository and launch the Chromium build managed by that Playwright version. Produce repeatable evidence, distinguish application defects from test and environment defects, and leave no browser or server process behind.

## Activation boundary

- Require the user to invoke `$playwright-chromium-testing`. A natural-language mention of Playwright/Chromium, task breadth, a local URL, screenshots, or the phrase "UI review" alone is not activation.
- Do not substitute an in-app browser, MCP browser, branded Chrome/Edge channel, persistent user profile, or installed browser unless the user explicitly requests that surface and it is appropriate.
- Do not use this skill for web research. Use the web/search capability for reading public pages and sources.
- Do not attach to or inspect an existing signed-in browser session. A Playwright `storageState` file is allowed only when the user placed it in scope and repository policy permits it.
- Do not install packages, download a browser, start a long-lived service, change authentication state, or interact with an external account merely because this skill is active. Those actions still require authorization from the task and repository instructions.
- Treat browser automation as task-local work. It never activates a repository goal, scheduler, frontier, background worker, or user-owned task.

## Non-negotiable defaults

- Read the repository instructions and inspect `git status --short` before writing.
- Preserve the repository's package manager, Playwright version, configuration, test layout, launch arguments, and artifact directories.
- Resolve Playwright from the repository. Do not prefer a global install or a hard-coded user path.
- Distinguish managed regular Chromium/new headless (`channel: 'chromium'`), managed Chromium headless shell (headless with no channel), headed regular Chromium, and branded Chrome/Edge. Preserve repository configuration; use managed regular Chromium/new headless for high-fidelity visual/E2E work unless the repository contract requires another mode.
- Use a fresh, non-persistent `BrowserContext` for each independent test. Make viewport, DPR, locale, timezone, color scheme, motion preference, and permissions explicit when they affect the result. Use the representative product baseline (normally `no-preference`) first and test reduced motion separately.
- Prefer semantic locators and auto-retrying web assertions. Never make fixed sleeps the synchronization strategy.
- Always close pages/contexts/browsers and stop only the verified task-owned server process tree. Never kill by broad process name.
- Never weaken an assertion, inflate a timeout, update a screenshot baseline, or add retries solely to make a failure green.

## Workflow

### 1. Inspect before choosing a test shape

Read, in order:

1. Repository and directory-level agent instructions.
2. Package manifests and lockfiles to identify the package manager and Playwright package/version.
3. Existing `playwright.config.*`, test fixtures, helpers, scripts, server recipes, and ignored artifact paths.
4. The implementation and user-visible contract under test.
5. Current worktree status so generated artifacts and user changes are preserved.

Choose the smallest appropriate mode:

- **Existing suite:** extend or run the repository's test runner and conventions.
- **Durable regression:** add a focused `@playwright/test` spec when the user authorized implementation and the behavior should remain covered.
- **Focused diagnostic:** use a small Playwright library script when a one-off probe is clearer than changing the suite.
- **No-repository-write verification:** inspect test source first. Run only a check proven not to mutate application/external state or write within the repository; redirect authorized ephemeral output to an approved OS-temp location. If that cannot be guaranteed, stop and report the limitation.
- **Strict filesystem read-only review:** use static inspection only. Launching Playwright/Chromium creates process and temporary-profile/cache state even when the repository remains untouched.

Do not conflate strict filesystem read-only, no tracked edits, and no product-code edits. Existing E2E tests can mutate databases/accounts and create reports even when source files are untouched.

Use [execution-workflows.md](references/execution-workflows.md) for package resolution, server lifecycle, Test Runner/library examples, and Windows details.

### 2. Define the QA inventory

Before launching Chromium, map each requested or intended final claim to observable evidence:

- requirement or claim;
- initial state and preconditions;
- user input or event sequence;
- functional assertion;
- visual state and viewport/DPR when appearance matters;
- expected diagnostic artifacts on failure.

Include the primary happy path, meaningful boundary/error states, and at least one recovery or reverse transition for stateful controls. Do not sign off on a claim that has no matching check.

### 3. Resolve the repository toolchain

- Prefer a repository script/task runner or a proven local-only command (`pnpm exec playwright`, `npx --no-install playwright`, `yarn exec playwright`, or a pinned Node plus the known local `node_modules/playwright/cli.js`). Inspect the owning manifest and installed module first; never let a detection command fetch a package.
- On Windows, resolve tools with PowerShell such as `Get-Command pnpm`; do not embed a user-specific executable path.
- Check the installed Playwright version before changing anything.
- If the managed Chromium executable is missing, report the exact failure and repository-appropriate install command. Download it only when installation is in scope.
- Use `scripts/probe-playwright-chromium.mjs` for a zero-page-HTTP(S) launch probe when the repository exposes a Node Playwright package. It refuses ancestor/global package resolution and reports the exact installed manifest.

### 4. Own the server lifecycle

- Prefer an existing `webServer` configuration or repository serve/test recipe.
- Prefer `127.0.0.1` for local browser targets and secure-context-sensitive APIs.
- Do not assume that a listening port serves the current build. Check an app-specific readiness/freshness signal when stale output is possible.
- Inspect configured API, telemetry, websocket, and authentication origins before navigation. A localhost page can still call or mutate production services; use a controlled test backend and a narrow expected-origin policy.
- Poll an HTTP/readiness condition with a bounded timeout instead of sleeping blindly.
- When starting a process manually on Windows, start it hidden and record its root PID, creation time, command line, logs, expected port, and verified descendants. In `finally`-equivalent cleanup, stop only that verified task-owned process tree and confirm its PIDs exited and port was released. Never kill by broad process name.
- Do not reuse an unrelated process merely because it owns the expected port.

### 5. Launch deterministic managed Chromium

Use a fresh context and explicit test conditions. For example:

```js
const browser = await chromium.launch({ headless: true, channel: 'chromium' });
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  deviceScaleFactor: 1,
  locale: 'en-US',
  timezoneId: 'UTC',
  colorScheme: 'light',
  reducedMotion: 'no-preference',
});
const page = await context.newPage();
```

This selects Playwright-managed regular Chromium/new headless. Omitting `channel` while headless selects the separate Chromium headless shell; use and report that mode deliberately for faster functional checks, not as an unlabeled substitute for a high-fidelity visual pass. Headed regular Chromium is a separate pass. Branded `chrome`/`msedge` channels are not managed Chromium.

Use a separate context for each viewport/DPR/device/motion state. Do not mutate one emulated context into a native-window pass. Preserve specialized repository launch flags for WebGPU, extensions, sandboxing, proxies, or media only when the test genuinely needs them.

### 6. Synchronize on meaning, not elapsed time

- Navigate with `domcontentloaded` or `load` according to the app, then wait for an observable readiness condition: a role/label, URL, response, application-ready flag, or stable rendered state.
- Do not make `networkidle` a universal prerequisite; long-polling, analytics, service workers, and background requests can make it misleading.
- Do not use `page.waitForTimeout()` in production tests. It is acceptable only as a temporary debugging aid and must not remain as correctness logic.
- Register event or popup/download waits before the action that triggers them.

### 7. Drive and assert like a user

Prefer locator contracts in this order:

1. `getByRole()` with accessible name;
2. `getByLabel()`, `getByPlaceholder()`, `getByText()`, `getByAltText()`, or `getByTitle()`;
3. a stable repository-defined test id;
4. a narrowly scoped CSS locator only when no user-facing contract exists.

- Avoid positional `nth()` and DOM-structure selectors unless uniqueness is first asserted and the structure is itself the contract.
- Use `await expect(locator).toBeVisible()`, `toHaveText()`, `toHaveValue()`, and other web-first assertions.
- Use `expect.poll()` or `expect(...).toPass()` for observable non-DOM state that must converge.
- Exercise real input APIs for signoff. `page.evaluate()` may inspect or stage state but does not prove a user interaction path.
- Assert the visible outcome and relevant side effects, not private implementation details.

### 8. Collect failure evidence early

Before navigation, subscribe to:

- `pageerror` / context `weberror`;
- relevant console errors and warnings;
- failed requests and unexpected HTTP failures when network behavior is in scope;
- dialogs, downloads, popups, and crashes when the flow can produce them.

For Test Runner suites, prefer traces on first retry or retained on failure, and screenshots on failure. For library probes, enable tracing only when the failure needs it and sensitive capture is authorized; minimize captured sources/snapshots, stop tracing before context closure, keep it in an approved ignored/temp path, and remove unneeded success traces. Keep artifacts under the repository's ignored output directory unless a reviewed baseline is intentionally versioned.

Classify every failure as application, test, environment/toolchain, or inconclusive. Confirm the classification with evidence before editing product code or tests. See [qa-and-failure-analysis.md](references/qa-and-failure-analysis.md).

### 9. Run visual QA separately

Functional success does not prove visual quality.

- Inspect the initial viewport and each claim-bearing state, including the densest realistic state and a post-interaction state.
- Check desktop and the smallest supported viewport; add a DPR pass when pixel density is relevant.
- Look for clipping, overflow, obscured controls, weak contrast, broken layering, font/icon failures, spacing/alignment drift, and unstable motion.
- Prefer viewport or element screenshots for signoff; use full-page screenshots as secondary context.
- Treat screenshot baselines as environment-specific. Generate or update them only deliberately, in the same OS/browser/font/headless environment used for comparison, and review the diff.
- For canvas/WebGPU surfaces, DOM presence is insufficient. Capture compositor pixels and use task-specific non-blank/content assertions; separately prove renderer readiness and served-build freshness when those claims matter.

### 10. Clean up and report honestly

In `finally` or framework teardown:

1. stop tracing and save it only when configured;
2. close pages and contexts;
3. close the browser;
4. stop only the verified task-owned server process tree and confirm its PIDs exited and port was released;
5. confirm no unexpected tracked artifacts or secret state were created.

Report:

- exact commands and project/config used;
- Chromium mode, viewport/DPR, and target URL;
- scenarios and claims exercised;
- passes, failures, flakes, and skipped checks;
- artifact paths;
- cleanup outcome;
- any limitation that prevents a stronger conclusion.

Do not call a test complete when the intended command never ran, matched zero tests, used a stale build, or omitted a required visual state.

## Security and repository hygiene

- Treat `storageState`, cookies, local/session storage, downloaded files, request headers, traces, and videos as potentially sensitive.
- Keep auth state in a gitignored repository-approved location; never print secret values or commit them.
- Do not use the user's normal browser profile. Persistent contexts are exceptional and require explicit need plus an isolated profile directory.
- Do not bypass access controls, anti-bot challenges, certificate policy, or repository capture policy.
- Do not perform destructive or externally visible actions against production systems without explicit authorization.
- Keep generated reports, traces, screenshots, downloads, and browser caches out of version control unless they are intentional reviewed artifacts. Do not overwrite an existing artifact or write outside the approved project/temp root without explicit authorization.

## References

- [execution-workflows.md](references/execution-workflows.md): command resolution, Test Runner/library patterns, server ownership, and cross-platform execution.
- [qa-and-failure-analysis.md](references/qa-and-failure-analysis.md): functional/visual coverage, diagnostics, canvas/WebGPU checks, and failure classification.
- [playwright-source-basis.md](references/playwright-source-basis.md): official Playwright guidance and comparable skill patterns used to validate this skill.
