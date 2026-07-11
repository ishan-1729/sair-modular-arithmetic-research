# Playwright source basis

This reference records the primary sources used to validate the skill's durable rules. Comparable skills were reviewed on 2026-07-11 and are pinned to the reviewed repository commits. Check current official Playwright documentation when a version-specific behavior or command matters.

## Official Playwright guidance

- [Browsers](https://playwright.dev/docs/browsers): managed regular Chromium/new headless (`channel: 'chromium'`), managed Chromium headless shell (headless without a channel), headed regular Chromium, and branded Chrome/Edge channels are distinct. This skill labels the selected mode and never silently switches channels.
- [Playwright Library](https://playwright.dev/docs/library): use Playwright Test for most durable E2E suites and the library for explicit custom automation lifecycles.
- [Locators](https://playwright.dev/docs/locators): prioritize user-facing attributes and explicit test-id contracts; locators re-resolve against the current DOM.
- [Actionability and auto-waiting](https://playwright.dev/docs/actionability): rely on Playwright's actionability checks instead of force actions or timing guesses.
- [Assertions](https://playwright.dev/docs/test-assertions): awaited web-first assertions retry until their condition or timeout; ordinary value assertions do not.
- [Best practices](https://playwright.dev/docs/best-practices): test user-visible behavior, isolate tests, use resilient locators, and prefer trace-based CI debugging.
- [Browser contexts](https://playwright.dev/docs/browser-contexts): non-persistent contexts isolate cookies, local storage, session storage, and other state without writing a normal profile to disk.
- [Configuration](https://playwright.dev/docs/test-configuration) and [web server](https://playwright.dev/docs/test-webserver): preserve repository projects, `baseURL`, output, retry, and server readiness conventions.
- [Trace Viewer](https://playwright.dev/docs/trace-viewer): traces can be collected on first retry or retained on failure and contain action, DOM, network, console, and screenshot evidence; this skill makes that sensitive capture conditional rather than automatic.
- [Authentication](https://playwright.dev/docs/auth): persisted state can contain impersonation-capable cookies and headers and must remain gitignored.
- [Network](https://playwright.dev/docs/network) and [`BrowserContext.routeWebSocket`](https://playwright.dev/docs/api/class-browsercontext#browser-context-route-web-socket): context-level HTTP routing, blocked service workers, and separately registered WebSocket routing are required when a test claims HTTP(S)/WebSocket origin containment; broader egress isolation needs a network boundary outside Playwright.
- [Visual comparisons](https://playwright.dev/docs/test-snapshots): screenshot baselines vary across OS, browser, settings, hardware, and headless mode, so baseline changes require a controlled environment and review.
- [`page.waitForTimeout`](https://playwright.dev/docs/api/class-page#page-wait-for-timeout): fixed waits are discouraged for production tests.
- [`page.waitForLoadState`](https://playwright.dev/docs/api/class-page#page-wait-for-load-state): `networkidle` is discouraged as a universal test-readiness signal; assert the application state that matters.
- [CI](https://playwright.dev/docs/ci): install matching browser dependencies, keep execution reproducible, and retain failure artifacts.

## Comparable skills inspected

### OpenAI `playwright-interactive`

Source: [OpenAI skills repository at `ecc1fb8bc0b82f1d3c7a92a82167ae8e8e613a67`](https://github.com/openai/skills/blob/ecc1fb8bc0b82f1d3c7a92a82167ae8e8e613a67/skills/.curated/playwright-interactive/SKILL.md)

Adopted:

- derive a QA inventory from requirements, implemented behavior, and final claims;
- run functional and visual QA as separate obligations;
- use explicit viewport/DPR contexts and treat native-window checks separately;
- require real input APIs for signoff;
- inspect initial, post-interaction, dense, narrow, and high-DPI states;
- clean up browser and owned process handles.

Not copied:

- persistent `js_repl` bindings, Electron support, and its sandbox preconditions are outside this shell/test-runner-oriented skill;
- its REPL-specific screenshot-emission protocol was not copied; the included probe owns only an optional PNG path with explicit root, symlink, overwrite, and atomic-write safeguards.

### Microsoft `playwright-cli`

Source: [Microsoft Playwright CLI skill at `34bf2ada4d4a28723bd65ba4e56ba724fdaaa959`](https://github.com/microsoft/playwright-cli/blob/34bf2ada4d4a28723bd65ba4e56ba724fdaaa959/skills/playwright-cli/SKILL.md)

Adopted:

- progressive disclosure through focused references;
- explicit session cleanup, diagnostics, tracing, and Windows command considerations;
- separate interactive automation from Playwright Test authoring/debugging.

Not copied:

- CLI-specific command catalogs and `allowed-tools` frontmatter do not match this skill's repository-native scope or local skill schema;
- global installation fallback, persistent profile, cookie manipulation, and kill-all commands are unsafe defaults.

### Anthropic `webapp-testing`

Source: [Anthropic skills repository at `ef740771ac901e03fbca3ce4e1c453a96010f30a`](https://github.com/anthropics/skills/blob/ef740771ac901e03fbca3ce4e1c453a96010f30a/skills/webapp-testing/SKILL.md)

Adopted:

- explicitly own local server lifecycle;
- inspect the rendered app before acting when the DOM is dynamic;
- always close Chromium.

Rejected or corrected:

- unconditional `networkidle` is not a reliable universal readiness rule;
- `waitForTimeout` must not be a normal synchronization recommendation;
- the repository's existing JavaScript/TypeScript conventions take precedence over a mandatory Python wrapper.

### Microsoft Playwright test agents

Sources at `d73823e63502a6386a408a25d74ebcbeb4a58f1c`: [planner](https://github.com/microsoft/playwright/blob/d73823e63502a6386a408a25d74ebcbeb4a58f1c/packages/playwright/src/agents/playwright-test-planner.agent.md), [generator](https://github.com/microsoft/playwright/blob/d73823e63502a6386a408a25d74ebcbeb4a58f1c/packages/playwright/src/agents/playwright-test-generator.agent.md), and [healer](https://github.com/microsoft/playwright/blob/d73823e63502a6386a408a25d74ebcbeb4a58f1c/packages/playwright/src/agents/playwright-test-healer.agent.md).

Adopted:

- plan independent scenarios and outcomes before authoring;
- execute intended interactions against the real page;
- inspect snapshots, logs, selectors, timing, network, and application behavior before repairing a test;
- rerun after remediation.

Rejected or constrained:

- a persistent failure must not be hidden with `test.fixme`, a weakened assertion, or an arbitrary repair loop;
- classify application, test, environment, and authorization failures before changing code.

## Validation conclusions

The source comparison supports five design decisions:

1. Require explicit `$playwright-chromium-testing` invocation and keep implicit invocation disabled.
2. Prefer the repository's pinned Playwright plus managed Chromium over global tools or browser-control substitution.
3. Plan evidence before automation and separate functional, visual, responsive, and pixel/render checks.
4. Synchronize on observable application state, not fixed sleeps or universal network idleness.
5. Preserve isolation, secret handling, artifact hygiene, and exact process cleanup as correctness requirements.
