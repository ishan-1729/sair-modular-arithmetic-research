# Execution workflows

Load this reference for toolchain setup, browser selection, local servers, durable tests, and bounded probes.

## Discover the repository contract

Inspect agent instructions, manifests, lockfiles, workspace structure, Playwright configuration, existing tests and helpers, server scripts, ignored artifact paths, and worktree status. In a monorepo, select the package that owns the relevant application or test. Never accept the first package found by directory order.

Resolve Playwright locally with a non-fetching command after confirming the dependency exists:

| Package manager | Version check |
|---|---|
| pnpm | pnpm exec playwright --version |
| npm | npx --no-install playwright --version |
| Yarn | yarn exec playwright --version |
| pinned Node | node node_modules/playwright/cli.js --version |

Do not use bare npx, npm exec, or bunx for detection because they may fetch packages.

## Bootstrap an empty tooling repository

When explicit invocation requires browser work and no repository toolchain exists:

1. Select a stable Playwright version from the authoritative package registry.
2. Create a minimal private package.json with an exact development dependency on Playwright.
3. Ignore node_modules, Playwright reports, test results, and task-specific artifact directories.
4. Run the repository package manager to create or update its lockfile and install dependencies.
5. Install only matching managed Chromium using the local CLI.
6. Record the pinned Playwright and managed Chromium versions.

This setup is authorized by explicit skill invocation and needs no additional skill-level prompt. Do not upgrade an existing pinned version unless compatibility or the task requires it.

## Select an execution mode

- Use an existing Playwright Test suite for durable regression testing, fixtures, projects, reporters, and CI.
- Use the Playwright library for bounded diagnostics, supplied-URL checks, research, screenshots, extraction, and custom lifecycles.
- Use managed regular Chromium/new headless by default for rendering fidelity.
- Use managed headless shell only as a deliberately labeled faster functional mode.
- Use headed regular Chromium only when visual or native-window behavior requires it.
- Do not substitute branded Chrome, Edge, or a personal browser profile.

## Local server ownership

Prefer an existing Playwright webServer configuration. Otherwise launch the documented server hidden, record its root PID, creation time, command line, logs, expected port, and verified descendants. Poll a meaningful readiness or build signal with a bounded deadline. Stop only that verified process tree and confirm the port is released. Never kill by broad process name or reuse an unrelated listener merely because it owns the expected port.

Starting and stopping a task-owned local server in the requested workflow requires no additional skill-level confirmation.

## Artifacts

Use OS temporary storage or an existing repository-approved or ignored location by default. Do not overwrite an existing artifact without task need. Confirm before writing outside those locations unless the user already authorized the exact destination. Treat screenshots, traces, videos, downloads, HTML snapshots, storage state, and network logs as potentially sensitive.

## Cleanup

Use finally-equivalent teardown to stop tracing, close pages, contexts, and browsers, stop verified task-owned servers, and check for unexpected artifacts or leaked processes. On interruption, make a bounded cleanup attempt and inspect remaining owned processes before continuing.
