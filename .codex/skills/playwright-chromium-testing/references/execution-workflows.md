# Execution workflows

Load this reference when selecting the Playwright execution mode, resolving a repository installation, starting a local app, or authoring a reusable test.

## Select library or Test Runner

Use `@playwright/test` when:

- the repository already has a Playwright Test suite;
- the behavior should become a durable regression test;
- fixtures, projects, retries, reporters, traces, or parallel isolation matter;
- CI will run the check.

Use the `playwright` library when:

- the repository already uses small `.mjs`/`.js` probes;
- the task is a bounded diagnostic, capture, or one-off local check;
- custom lifecycle or pixel analysis would be awkward inside the Test Runner;
- the requested boundary permits OS-temp/browser-profile writes and the probe's application/network side effects are understood.

Do not add both packages merely for convenience. Preserve whichever package and execution style the repository already owns.

If the user requires strict filesystem read-only operation, do not launch Playwright or Chromium; browser startup creates temporary-profile/cache and process state. Limit that task to static inspection.

## Discover the repository contract

Look for:

- `package.json`, workspace manifests, and lockfiles;
- `playwright.config.{ts,js,mts,mjs,cts,cjs}`;
- `*.spec.*`, `*.test.*`, fixtures, page objects, and test helpers;
- task runners such as `justfile`, Makefile, npm scripts, or PowerShell scripts;
- environment variables for base URL, headed mode, browser flags, and output paths;
- `.gitignore` entries for `test-results`, reports, screenshots, traces, downloads, and auth state;
- repository-specific readiness and build-freshness signals.

Do not infer that Playwright is absent from a missing root dependency. Monorepos commonly install it in `frontend/`, `web/`, `e2e/`, or another workspace package.
If more than one workspace package owns a Playwright dependency, select the package from the requested application/config and pass it explicitly to any generic probe. Never accept the first installation found by directory search order as evidence for the whole repository.

## Resolve Playwright locally

First inspect the owning manifest and verify that the local package/module exists. Then use a non-fetching version check from that package directory:

| Repository | Local-only version check |
|---|---|
| pnpm | `pnpm exec playwright --version` after confirming the dependency is installed |
| npm | `npx --no-install playwright --version` |
| Yarn node_modules | `yarn exec playwright --version` |
| pinned Node | `node node_modules/playwright/cli.js --version` using the known package path |

Do not use `npm exec`, bare `npx`, or `bunx` for detection: they may fetch a missing package. Yarn Plug'n'Play and non-`node_modules` layouts require the repository's Yarn runtime/hook; the generic Node probe may fail unless invoked through that runtime. Report that limitation instead of falling back to an ancestor or global install.

The included probe considers only repository-local package manifests that directly declare `playwright` or `@playwright/test`, including when `--package-dir` is explicit; it never treats an inherited/hoisted module as proof that the selected package owns Playwright. It fails closed when multiple candidate package owners are found; rerun with `--package-dir` for the package that owns the relevant config/test. Its common-directory discovery is a convenience, not a complete workspace parser, so pass `--package-dir` for nonstandard layouts.

For its optional screenshot, choose an approved path whose parent directory already exists. The probe will not recursively create a lexical parent path before canonical containment is established; create an intended ignored/temp artifact directory separately under the repository's normal workflow when needed.

For repositories with a pinned Node executable, invoke the local Playwright CLI directly:

```powershell
& $node frontend/node_modules/playwright/cli.js --version
```

When a matching managed Chromium download is authorized, invoke the already-installed local CLI only:

```text
pnpm exec playwright install chromium
npx --no-install playwright install chromium
yarn exec playwright install chromium
node node_modules/playwright/cli.js install chromium
```

These install commands are documentation, not standing authorization.

## Resolve the library from a nested package

A script outside the package that owns Playwright should resolve from that package instead of importing globally:

```js
import { createRequire } from 'node:module';

const requireFromWeb = createRequire(
  new URL('../../frontend/package.json', import.meta.url),
);
const { chromium } = requireFromWeb('playwright');
```

For a generic script with an absolute package directory:

```js
import { createRequire } from 'node:module';
import { join } from 'node:path';

const requireFromPackage = createRequire(join(packageDir, 'package.json'));
const { chromium } = requireFromPackage('playwright');
```

Use the repository's existing helper if it already centralizes this resolution. Verify `require.resolve()` and the installed package manifest both resolve inside the declared project root. Never walk above the project root or accept a global/ancestor module merely because Node can resolve it.

## Minimal Playwright Test configuration

Create a config only when no suitable config exists and durable tests are in scope. Adapt names, commands, ports, artifact paths, and timeouts to the repository:

```ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  outputDir: 'test-results',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    channel: 'chromium',
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
    locale: 'en-US',
    timezoneId: 'UTC',
    colorScheme: 'light',
    reducedMotion: 'no-preference',
    trace: process.env.PW_CAPTURE_TRACE === '1' ? 'on-first-retry' : 'off',
    screenshot: process.env.PW_CAPTURE_SCREENSHOTS === '1' ? 'only-on-failure' : 'off',
  },
  webServer: {
    command: 'pnpm run preview --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173/health',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
```

Notes:

- `channel: 'chromium'` selects Playwright-managed regular Chromium/new headless. Headless with no channel selects the separate managed Chromium headless shell. A headed launch uses regular managed Chromium. `channel: 'chrome'` and `channel: 'msedge'` select branded installed browsers, not managed Chromium.
- Use the regular/new-headless mode for high-fidelity E2E and visual signoff unless the repository contract says otherwise. Use headless shell deliberately for faster functional checks and report it by name.
- Use the representative motion preference (normally `no-preference`) for the primary pass. Test `reduce` separately; it can execute different product code and suppress defects.
- Enable traces and screenshots only when their diagnostic value is needed and their potentially sensitive content is authorized. Keep them in an approved ignored/temp path and delete unneeded success artifacts.
- `reuseExistingServer` is safe only when the readiness URL proves the expected app/build. Set it to `false` when stale or unrelated listeners are a realistic risk.
- Retrying once can collect a trace and identify a flake. A retry pass does not erase the first failure; report the test as flaky.
- Do not increase global timeouts to conceal one slow or unsynchronized assertion.

## Durable test pattern

```ts
import { test, expect } from '@playwright/test';

test('saves profile changes and exposes the persisted result', async ({ page }) => {
  await page.goto('/settings');

  const name = page.getByLabel('Display name');
  await expect(name).toHaveValue('Original');
  await name.fill('Updated');
  await page.getByRole('button', { name: 'Save' }).click();

  await expect(page.getByRole('status')).toHaveText('Saved');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(name).toHaveValue('Updated');
});
```

The complete round trip matters: initial state, user action, visible acknowledgement, and persisted outcome.

## Focused library probe pattern

```js
import { createRequire } from 'node:module';

const requireFromPackage = createRequire(new URL('../package.json', import.meta.url));
const { chromium } = requireFromPackage('playwright');

let browser;
let context;
try {
  browser = await chromium.launch({ headless: true, channel: 'chromium' });
  context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
    locale: 'en-US',
    timezoneId: 'UTC',
    colorScheme: 'light',
    reducedMotion: 'no-preference',
    serviceWorkers: 'block',
  });

  const pageErrors = [];
  const consoleErrors = [];
  const expectedHttpOrigins = new Set(['http://127.0.0.1:4173']);
  const expectedWebSocketOrigins = new Set(['ws://127.0.0.1:4173']);

  const attachDiagnostics = page => {
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
  };
  context.on('page', attachDiagnostics);

  // Install containment at context scope before creating any page or popup.
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (['http:', 'https:'].includes(url.protocol) && !expectedHttpOrigins.has(url.origin)) {
      await route.abort('blockedbyclient');
      return;
    }
    await route.continue();
  });

  if (typeof context.routeWebSocket !== 'function') {
    throw new Error('This containment probe requires Playwright 1.48+ for routeWebSocket');
  }
  await context.routeWebSocket(/.*/, webSocket => {
    const origin = new URL(webSocket.url()).origin;
    if (!expectedWebSocketOrigins.has(origin)) {
      webSocket.close({ code: 1008, reason: 'Unexpected WebSocket origin' });
      return;
    }
    webSocket.connectToServer();
  });

  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173', { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: 'Dashboard' }).waitFor();

  // Exercise the requested behavior and assert its observable outcome.

  if (pageErrors.length || consoleErrors.length) {
    throw new Error(JSON.stringify({ pageErrors, consoleErrors }));
  }
} finally {
  await context?.close().catch(() => {});
  await browser?.close().catch(() => {});
}
```

Before defining either allowlist, inspect the app's backend, authentication, telemetry, websocket, and asset configuration. Add only controlled test origins that the scoped flow requires; a local page can otherwise mutate production. `context.route()` covers HTTP(S) requests from every page in the context, while `routeWebSocket()` separately constrains WebSockets and must be registered before pages are created. The example blocks service workers because they can hide network requests from routing. If the behavior under test requires service workers, test them in a separately designed environment and do not claim this pattern provides complete outbound containment. These APIs also do not prove containment of WebRTC, WebTransport, DNS, or browser-internal traffic; use an isolated network/proxy/firewall boundary when complete egress control is required. `routeWebSocket()` requires Playwright 1.48 or newer; on an older repository version, preserve that version and report that this HTTP(S)/WebSocket containment pattern is unavailable instead of silently downgrading it to monitoring. For Test Runner code, import `test` and `expect` from `@playwright/test`. For library code, use Node assertions or explicit checks; do not pretend library locators include Test Runner matchers.

## Local server ownership

Prefer Playwright Test's `webServer` because it owns readiness and teardown. When a manual server is required:

1. Choose the repository's documented serve command and working directory.
2. Prefer launching the final server executable directly. A package-manager wrapper may exit while leaving a child Vite/Node process alive.
3. Capture stdout/stderr to task-local logs and record start time, root PID, command line, expected port, and any verified descendant PIDs.
4. Start hidden on Windows unless the user needs an interactive window.
5. Poll a specific URL or app-ready/build-freshness signal with a deadline.
6. On failure, include recent redacted logs.
7. In cleanup, attempt graceful shutdown first, then stop only the verified task-owned process tree.
8. Verify the root and descendants exited and the expected port was released. If a process remains, report it rather than using a broad kill.

Use a repository helper that owns a Windows Job Object/process group when available. If no reliable descendant ownership mechanism exists, prefer `webServer` or a foreground runner that can be interrupted and observed. Do not assume stopping a wrapper PID stops its children.

Before stopping any PID obtained indirectly, re-read its command line, parent chain, and creation time to verify that it belongs to this run. Never use `taskkill /IM`, broad `Stop-Process -Name`, or a cross-shell deletion/kill pipeline. On SIGINT/SIGTERM or tool interruption, make a bounded best-effort cleanup attempt and inspect/report remaining owned processes on the next step.

## Windows command details

- Quote URLs containing `&`; PowerShell otherwise interprets it as an operator.
- Use `Get-Command pnpm`, `Get-Command node`, or the repository resolver instead of guessing install paths.
- Keep path handling inside PowerShell for process/filesystem operations; do not enumerate in one shell and delete or move in another.
- When a repository requires Bash for a build step, run that build through its documented Bash entry point, but keep Windows process cleanup in PowerShell.
- Prefer non-interactive reports (`PLAYWRIGHT_HTML_OPEN=never`) in automated runs so the test runner does not open an unrelated UI.

## Installation and version policy

- Playwright packages and managed browser binaries are version-coupled. Use the locked package version.
- Do not upgrade to `latest` during a test or bug-fix task unless the user requested an upgrade or the installed version is proven to be the defect.
- Install only Chromium when that is the scoped browser.
- Do not change from managed Chromium to branded Chrome/Edge to bypass a launch failure. Diagnose the missing browser, OS dependency, policy, sandbox, or launch-flag issue.
- Never disable TLS verification, the browser sandbox, or network restrictions as a routine fix.
