#!/usr/bin/env node

import {
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  realpathSync,
} from 'node:fs';
import { link, open, rename, unlink } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from 'node:path';

const HELP = `
Probe a repository-local Playwright package and a selected managed Chromium mode.

The probe does not navigate to the repository application or an external URL. It
opens an isolated context, loads in-memory HTML, asserts that the page made zero
HTTP(S) requests, performs one semantic locator interaction, optionally saves a
screenshot, and closes its handles. Normal completion guarantees cleanup; signal
handling makes a bounded best-effort cleanup attempt.

Usage:
  node probe-playwright-chromium.mjs [options]

Options:
  --project-root <path>          Repository/workspace root (default: current directory)
  --package-dir <path>           Package directory that owns Playwright; must stay
                                 within --project-root
  --mode <regular|headless-shell>
                                 regular (default) uses Playwright-managed regular
                                 Chromium/new headless via channel "chromium";
                                 headless-shell selects the separate managed shell
  --headed                       Launch managed regular Chromium visibly; incompatible
                                 with --mode headless-shell
  --screenshot <path>            Optional PNG output path; parent must already exist
  --allow-external-screenshot    Permit screenshot output outside --project-root
  --overwrite                    Permit replacing an existing screenshot
  --help                         Show this help

Examples:
  node probe-playwright-chromium.mjs --project-root .
  node probe-playwright-chromium.mjs --project-root . --package-dir frontend
  node probe-playwright-chromium.mjs --mode headless-shell
  node probe-playwright-chromium.mjs --screenshot pw-probe.png
`;

const REQUIRED_CHECKS = [
  'repository-local installed Playwright package resolves',
  'selected managed Chromium mode launches',
  'semantic heading locator resolves',
  'isolated in-memory page starts in expected state',
  'semantic locator interaction updates visible state',
  'explicit context options are observable',
  'page made zero HTTP(S) requests',
  'no page, context, console, or crash errors',
];

function parseArgs(argv) {
  const options = {
    projectRoot: process.cwd(),
    packageDir: null,
    mode: 'regular',
    screenshot: null,
    allowExternalScreenshot: false,
    overwrite: false,
    headed: false,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
    } else if (argument === '--headed') {
      options.headed = true;
    } else if (argument === '--allow-external-screenshot') {
      options.allowExternalScreenshot = true;
    } else if (argument === '--overwrite') {
      options.overwrite = true;
    } else if (['--project-root', '--package-dir', '--mode', '--screenshot'].includes(argument)) {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) {
        throw new Error(`${argument} requires a value`);
      }
      index += 1;
      if (argument === '--project-root') options.projectRoot = value;
      if (argument === '--package-dir') options.packageDir = value;
      if (argument === '--mode') options.mode = value;
      if (argument === '--screenshot') options.screenshot = value;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }

  if (!['regular', 'headless-shell'].includes(options.mode)) {
    throw new Error('--mode must be regular or headless-shell');
  }
  if (options.headed && options.mode === 'headless-shell') {
    throw new Error('--headed is incompatible with --mode headless-shell');
  }
  if (options.allowExternalScreenshot && !options.screenshot) {
    throw new Error('--allow-external-screenshot requires --screenshot');
  }
  if (options.overwrite && !options.screenshot) {
    throw new Error('--overwrite requires --screenshot');
  }

  return options;
}

function isPathWithin(root, candidate) {
  const rel = relative(root, candidate);
  return rel === '' || (!isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${sep}`));
}

function realpathExisting(path, label) {
  if (!existsSync(path)) throw new Error(`${label} does not exist: ${path}`);
  return realpathSync(path);
}

function addCandidate(candidates, seen, projectRootReal, candidate) {
  const absolute = resolve(candidate);
  if (!isPathWithin(projectRootReal, absolute)) return;
  const key = process.platform === 'win32' ? absolute.toLowerCase() : absolute;
  if (!seen.has(key)) {
    seen.add(key);
    candidates.push(absolute);
  }
}

function addImmediateChildren(candidates, seen, projectRootReal, parent) {
  if (!existsSync(parent)) return;
  let parentReal;
  try {
    parentReal = realpathSync(parent);
  } catch {
    return;
  }
  if (!isPathWithin(projectRootReal, parentReal)) return;
  for (const entry of readdirSync(parentReal, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      addCandidate(candidates, seen, projectRootReal, join(parentReal, entry.name));
    }
  }
}

function candidatePackageDirs(projectRootReal, explicitPackageDir) {
  const candidates = [];
  const seen = new Set();

  if (explicitPackageDir) {
    const requested = isAbsolute(explicitPackageDir)
      ? resolve(explicitPackageDir)
      : resolve(projectRootReal, explicitPackageDir);
    if (!isPathWithin(projectRootReal, requested)) {
      throw new Error(`--package-dir must stay within --project-root: ${requested}`);
    }
    addCandidate(candidates, seen, projectRootReal, requested);
    return candidates;
  }

  addCandidate(candidates, seen, projectRootReal, projectRootReal);
  for (const child of ['frontend', 'web', 'client', 'app', 'e2e', 'tests/e2e']) {
    addCandidate(candidates, seen, projectRootReal, join(projectRootReal, child));
  }
  addImmediateChildren(candidates, seen, projectRootReal, join(projectRootReal, 'packages'));
  addImmediateChildren(candidates, seen, projectRootReal, join(projectRootReal, 'apps'));
  return candidates;
}

function findInstalledManifest(entryPath, expectedName, projectRootReal) {
  let current = dirname(entryPath);
  while (isPathWithin(projectRootReal, current)) {
    const manifestPath = join(current, 'package.json');
    if (existsSync(manifestPath)) {
      try {
        const manifestPathReal = realpathSync(manifestPath);
        if (!isPathWithin(projectRootReal, manifestPathReal)) {
          throw new Error('installed manifest resolves outside project root');
        }
        const manifest = JSON.parse(readFileSync(manifestPathReal, 'utf8'));
        if (manifest.name === expectedName && typeof manifest.version === 'string') {
          return { manifestPath: manifestPathReal, manifest };
        }
      } catch {
        // Keep walking; an unreadable unrelated manifest is not the package manifest.
      }
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

function declaredPlaywrightPackages(manifest) {
  const dependencyFields = [
    manifest.dependencies,
    manifest.devDependencies,
    manifest.optionalDependencies,
    manifest.peerDependencies,
  ].filter(field => field && typeof field === 'object');
  return ['playwright', '@playwright/test'].filter(packageName =>
    dependencyFields.some(field => Object.prototype.hasOwnProperty.call(field, packageName)));
}

function loadPlaywright(projectRootReal, explicitPackageDir) {
  const attempts = [];
  const matches = [];
  const canonicalPackageDirs = new Set();

  for (const packageDir of candidatePackageDirs(projectRootReal, explicitPackageDir)) {
    let packageDirReal;
    try {
      packageDirReal = realpathExisting(packageDir, 'candidate package directory');
    } catch (error) {
      attempts.push({ packageDir, reason: error.message });
      continue;
    }
    if (!isPathWithin(projectRootReal, packageDirReal)) {
      attempts.push({
        packageDir,
        reason: `candidate package directory resolves outside project root: ${packageDirReal}`,
      });
      continue;
    }
    const packageDirKey = process.platform === 'win32'
      ? packageDirReal.toLowerCase()
      : packageDirReal;
    if (canonicalPackageDirs.has(packageDirKey)) continue;
    canonicalPackageDirs.add(packageDirKey);

    const unresolvedManifestPath = join(packageDirReal, 'package.json');
    if (!existsSync(unresolvedManifestPath)) {
      attempts.push({ packageDir: packageDirReal, reason: 'package.json not found' });
      continue;
    }

    let manifest;
    let manifestPath;
    try {
      manifestPath = realpathSync(unresolvedManifestPath);
      if (!isPathWithin(projectRootReal, manifestPath)) {
        attempts.push({
          packageDir: packageDirReal,
          reason: `package.json resolves outside project root: ${manifestPath}`,
        });
        continue;
      }
      manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    } catch (error) {
      attempts.push({ packageDir: packageDirReal, reason: `invalid package.json: ${error.message}` });
      continue;
    }

    const declaredPackages = declaredPlaywrightPackages(manifest);
    if (declaredPackages.length === 0) {
      attempts.push({
        packageDir: packageDirReal,
        reason: 'manifest does not directly declare playwright or @playwright/test',
      });
      continue;
    }

    const requireFromPackage = createRequire(manifestPath);
    for (const packageName of declaredPackages) {
      try {
        const resolvedEntry = realpathSync(requireFromPackage.resolve(packageName));
        if (!isPathWithin(projectRootReal, resolvedEntry)) {
          attempts.push({
            packageDir,
            packageName,
            reason: `module resolves outside project root: ${resolvedEntry}`,
          });
          continue;
        }

        let installed;
        try {
          const installedManifestPath = realpathSync(
            requireFromPackage.resolve(`${packageName}/package.json`),
          );
          if (!isPathWithin(projectRootReal, installedManifestPath)) {
            throw new Error('installed package manifest resolves outside project root');
          }
          const installedManifest = JSON.parse(readFileSync(installedManifestPath, 'utf8'));
          installed = { manifestPath: installedManifestPath, manifest: installedManifest };
        } catch {
          installed = findInstalledManifest(resolvedEntry, packageName, projectRootReal);
        }

        if (!installed || !isPathWithin(projectRootReal, installed.manifestPath)) {
          attempts.push({
            packageDir,
            packageName,
            reason: 'could not prove an installed package manifest inside project root',
          });
          continue;
        }
        if (installed.manifest.name !== packageName || typeof installed.manifest.version !== 'string') {
          attempts.push({
            packageDir,
            packageName,
            reason: 'installed package manifest name/version is invalid',
          });
          continue;
        }

        const module = requireFromPackage(packageName);
        if (!module?.chromium) {
          attempts.push({ packageDir, packageName, reason: 'module has no chromium export' });
          continue;
        }

        matches.push({
          chromium: module.chromium,
          packageDir: packageDirReal,
          packageName,
          packageVersion: installed.manifest.version,
          versionSource: 'installed-package-manifest',
          installedManifestPath: installed.manifestPath,
          resolvedModulePath: resolvedEntry,
          manifestName: manifest.name ?? null,
          attempts,
        });
        break;
      } catch (error) {
        attempts.push({
          packageDir,
          packageName,
          reason: String(error.message).split('\n')[0],
        });
      }
    }
  }

  if (matches.length === 1) return matches[0];

  if (matches.length > 1) {
    const error = new Error(
      'multiple repository-local Playwright package owners were found; pass --package-dir to select the intended package',
    );
    error.attempts = attempts;
    error.candidates = matches.map(match => ({
      packageDir: match.packageDir,
      packageName: match.packageName,
      packageVersion: match.packageVersion,
      installedManifestPath: match.installedManifestPath,
    }));
    throw error;
  }

  const error = new Error(
    explicitPackageDir
      ? `could not resolve a repository-local playwright or @playwright/test from ${explicitPackageDir}`
      : 'could not resolve a repository-local playwright or @playwright/test from the project or supported workspace directories',
  );
  error.attempts = attempts;
  throw error;
}

function lstatIfPresent(path) {
  try {
    return lstatSync(path);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function assertScreenshotLeaf(path, overwrite) {
  const stats = lstatIfPresent(path);
  if (!stats) return;
  if (stats.isSymbolicLink()) {
    throw new Error(`screenshot target must not be a symbolic link: ${path}`);
  }
  if (!stats.isFile()) {
    throw new Error(`screenshot target must be a regular file path: ${path}`);
  }
  if (!overwrite) {
    throw new Error(`screenshot already exists; pass --overwrite to replace it: ${path}`);
  }
}

function samePath(left, right) {
  const leftResolved = resolve(left);
  const rightResolved = resolve(right);
  return process.platform === 'win32'
    ? leftResolved.toLowerCase() === rightResolved.toLowerCase()
    : leftResolved === rightResolved;
}

async function writeScreenshotBuffer({
  buffer,
  outputPath,
  projectRootReal,
  allowExternal,
  overwrite,
}) {
  const parentReal = realpathExisting(dirname(outputPath), 'screenshot parent');
  if (!allowExternal && !isPathWithin(projectRootReal, parentReal)) {
    throw new Error(`screenshot parent resolves outside --project-root: ${parentReal}`);
  }

  const targetPath = join(parentReal, basename(outputPath));
  if (!allowExternal && !isPathWithin(projectRootReal, targetPath)) {
    throw new Error(`screenshot target resolves outside --project-root: ${targetPath}`);
  }
  assertScreenshotLeaf(targetPath, overwrite);

  const tempPath = join(
    parentReal,
    `.${basename(outputPath)}.${process.pid}.${randomUUID()}.tmp`,
  );
  let tempExists = false;
  try {
    const handle = await open(tempPath, 'wx', 0o600);
    tempExists = true;
    try {
      await handle.writeFile(buffer);
      await handle.sync();
    } finally {
      await handle.close();
    }

    const parentNow = realpathExisting(parentReal, 'canonical screenshot parent');
    if (!samePath(parentNow, parentReal)) {
      throw new Error(`screenshot parent changed during capture: ${parentReal}`);
    }
    if (!allowExternal && !isPathWithin(projectRootReal, parentNow)) {
      throw new Error(`screenshot parent escaped --project-root during capture: ${parentNow}`);
    }
    assertScreenshotLeaf(targetPath, overwrite);

    if (overwrite) {
      await rename(tempPath, targetPath);
    } else {
      try {
        await link(tempPath, targetPath);
      } catch (error) {
        if (error.code === 'EEXIST') {
          throw new Error(`screenshot target appeared during capture; refusing to overwrite: ${targetPath}`);
        }
        throw error;
      }
      await unlink(tempPath);
    }
    tempExists = false;
  } finally {
    if (tempExists) await unlink(tempPath).catch(() => {});
  }

  const written = lstatIfPresent(targetPath);
  if (!written?.isFile() || written.isSymbolicLink() || written.size !== buffer.length) {
    throw new Error(`screenshot output verification failed: ${targetPath}`);
  }
  return targetPath;
}

function resolveScreenshotPath(projectRootReal, value, allowExternal) {
  if (!value) return null;
  const outputPath = isAbsolute(value) ? resolve(value) : resolve(projectRootReal, value);
  if (!allowExternal && !isPathWithin(projectRootReal, outputPath)) {
    throw new Error('screenshot path must stay within --project-root unless --allow-external-screenshot is supplied');
  }

  const parentReal = realpathExisting(dirname(outputPath), 'screenshot parent');
  if (!allowExternal && !isPathWithin(projectRootReal, parentReal)) {
    throw new Error(`screenshot parent resolves outside --project-root: ${parentReal}`);
  }
  return outputPath;
}

let options;
try {
  options = parseArgs(process.argv.slice(2));
} catch (error) {
  console.error(error.message);
  console.error(HELP.trim());
  process.exit(2);
}

if (options.help) {
  console.log(HELP.trim());
  process.exit(0);
}

let projectRoot;
let screenshotPath;
try {
  projectRoot = realpathExisting(resolve(options.projectRoot), '--project-root');
  screenshotPath = resolveScreenshotPath(
    projectRoot,
    options.screenshot,
    options.allowExternalScreenshot,
  );
  if (screenshotPath) {
    assertScreenshotLeaf(screenshotPath, options.overwrite);
    const existingTarget = lstatIfPresent(screenshotPath);
    if (existingTarget && !options.allowExternalScreenshot) {
      const existingTargetReal = realpathSync(screenshotPath);
      if (!isPathWithin(projectRoot, existingTargetReal)) {
        throw new Error(`existing screenshot target resolves outside --project-root: ${existingTargetReal}`);
      }
    }
  }
} catch (error) {
  console.error(error.message);
  process.exit(2);
}

const launch = options.headed
  ? {
      mode: 'managed-regular-chromium-headed',
      options: { headless: false, channel: 'chromium' },
    }
  : options.mode === 'regular'
    ? {
        mode: 'managed-regular-chromium-new-headless',
        options: { headless: true, channel: 'chromium' },
      }
    : {
        mode: 'managed-chromium-headless-shell',
        options: { headless: true },
      };

const result = {
  kind: 'playwright-managed-chromium-probe',
  scope: 'environment-only; no repository application or external URL was tested',
  projectRoot,
  packageDir: null,
  packageName: null,
  packageVersion: null,
  versionSource: null,
  installedManifestPath: null,
  resolvedModulePath: null,
  manifestName: null,
  browserTypeExecutablePath: null,
  browserVersion: null,
  launchMode: launch.mode,
  launchOptions: launch.options,
  context: {
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
    locale: 'en-US',
    timezoneId: 'UTC',
    colorScheme: 'light',
    reducedMotion: 'no-preference',
  },
  requiredChecks: REQUIRED_CHECKS,
  missingRequiredChecks: [...REQUIRED_CHECKS],
  checks: [],
  diagnostics: {
    pageErrors: [],
    webErrors: [],
    consoleErrors: [],
    crashes: [],
    httpRequests: [],
  },
  screenshot: screenshotPath,
  cleanup: {
    contextClosed: false,
    browserClosed: false,
    errors: [],
  },
  resolutionAttempts: [],
  resolutionCandidates: [],
  error: null,
  hint: null,
  passed: false,
};

let browser;
let context;
let cleanupPromise = null;
let signalCleanupStarted = false;

async function cleanupHandles() {
  if (cleanupPromise) return cleanupPromise;
  cleanupPromise = (async () => {
    if (context) {
      try {
        await context.close();
        result.cleanup.contextClosed = true;
      } catch (error) {
        result.cleanup.errors.push(`context close: ${error.message}`);
      }
    } else {
      result.cleanup.contextClosed = true;
    }

    if (browser) {
      try {
        await browser.close();
        result.cleanup.browserClosed = true;
      } catch (error) {
        result.cleanup.errors.push(`browser close: ${error.message}`);
      }
    } else {
      result.cleanup.browserClosed = true;
    }
  })();
  return cleanupPromise;
}

function installSignalHandler(signal, exitCode) {
  try {
    process.once(signal, () => {
      if (signalCleanupStarted) return;
      signalCleanupStarted = true;
      console.error(`Received ${signal}; attempting bounded Playwright cleanup.`);
      const deadline = setTimeout(() => process.exit(exitCode), 5_000);
      void cleanupHandles().finally(() => {
        clearTimeout(deadline);
        process.exit(exitCode);
      });
    });
  } catch {
    // Some signals are not available on every platform.
  }
}

installSignalHandler('SIGINT', 130);
installSignalHandler('SIGTERM', 143);
if (process.platform === 'win32') installSignalHandler('SIGBREAK', 131);

try {
  const loaded = loadPlaywright(projectRoot, options.packageDir);
  result.packageDir = loaded.packageDir;
  result.packageName = loaded.packageName;
  result.packageVersion = loaded.packageVersion;
  result.versionSource = loaded.versionSource;
  result.installedManifestPath = loaded.installedManifestPath;
  result.resolvedModulePath = loaded.resolvedModulePath;
  result.manifestName = loaded.manifestName;
  result.resolutionAttempts = loaded.attempts;
  result.browserTypeExecutablePath = loaded.chromium.executablePath();
  result.checks.push({
    name: 'repository-local installed Playwright package resolves',
    pass: existsSync(result.installedManifestPath)
      && existsSync(result.resolvedModulePath)
      && isPathWithin(projectRoot, result.installedManifestPath)
      && isPathWithin(projectRoot, result.resolvedModulePath),
    detail: {
      packageName: result.packageName,
      packageVersion: result.packageVersion,
      versionSource: result.versionSource,
      installedManifestPath: result.installedManifestPath,
      resolvedModulePath: result.resolvedModulePath,
    },
  });

  browser = await loaded.chromium.launch(launch.options);
  result.browserVersion = browser.version();
  result.checks.push({
    name: 'selected managed Chromium mode launches',
    pass: true,
    detail: {
      launchMode: result.launchMode,
      browserVersion: result.browserVersion,
      browserTypeExecutablePath: result.browserTypeExecutablePath,
      note: result.launchMode === 'managed-chromium-headless-shell'
        ? 'browserTypeExecutablePath describes regular managed Chromium, not the separate headless-shell binary selected by this mode'
        : 'browserTypeExecutablePath corresponds to the selected managed regular Chromium family',
    },
  });

  context = await browser.newContext(result.context);
  context.on('weberror', event => {
    result.diagnostics.webErrors.push(String(event.error()));
  });

  const page = await context.newPage();
  page.setDefaultTimeout(10_000);
  page.on('pageerror', error => {
    result.diagnostics.pageErrors.push(String(error));
  });
  page.on('console', message => {
    if (message.type() === 'error') result.diagnostics.consoleErrors.push(message.text());
  });
  page.on('crash', () => {
    result.diagnostics.crashes.push('page crashed');
  });
  page.on('request', request => {
    if (/^https?:/i.test(request.url())) {
      result.diagnostics.httpRequests.push({ method: request.method(), url: request.url() });
    }
  });

  await page.setContent(`<!doctype html>
    <html lang="en">
      <head><meta charset="utf-8"><title>Managed Chromium probe</title></head>
      <body>
        <main>
          <h1>Managed Chromium probe</h1>
          <button type="button">Run semantic interaction</button>
          <output data-testid="status" aria-live="polite">idle</output>
        </main>
        <script>
          document.querySelector('button').addEventListener('click', () => {
            document.querySelector('[data-testid="status"]').textContent = 'chromium-ok';
          });
        </script>
      </body>
    </html>`, { waitUntil: 'domcontentloaded' });

  const heading = page.getByRole('heading', { name: 'Managed Chromium probe' });
  const button = page.getByRole('button', { name: 'Run semantic interaction' });
  const status = page.getByTestId('status');

  await heading.waitFor({ state: 'visible' });
  result.checks.push({ name: 'semantic heading locator resolves', pass: true });

  const initialStatus = await status.textContent();
  result.checks.push({
    name: 'isolated in-memory page starts in expected state',
    pass: initialStatus === 'idle',
    detail: initialStatus,
  });

  await button.click();
  await status.waitFor({ state: 'visible' });
  const finalStatus = await status.textContent();
  result.checks.push({
    name: 'semantic locator interaction updates visible state',
    pass: finalStatus === 'chromium-ok',
    detail: finalStatus,
  });

  const contextState = await page.evaluate(() => ({
    viewport: { width: window.innerWidth, height: window.innerHeight },
    deviceScaleFactor: window.devicePixelRatio,
    locale: navigator.language,
    timezoneId: Intl.DateTimeFormat().resolvedOptions().timeZone,
    colorScheme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
    reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'reduce' : 'no-preference',
  }));
  const contextPass = contextState.viewport.width === result.context.viewport.width
    && contextState.viewport.height === result.context.viewport.height
    && contextState.deviceScaleFactor === result.context.deviceScaleFactor
    && contextState.locale === result.context.locale
    && contextState.timezoneId === result.context.timezoneId
    && contextState.colorScheme === result.context.colorScheme
    && contextState.reducedMotion === result.context.reducedMotion;
  result.checks.push({
    name: 'explicit context options are observable',
    pass: contextPass,
    detail: contextState,
  });

  result.checks.push({
    name: 'page made zero HTTP(S) requests',
    pass: result.diagnostics.httpRequests.length === 0,
    detail: result.diagnostics.httpRequests,
  });

  if (screenshotPath) {
    const screenshotBuffer = await page.screenshot({ type: 'png' });
    const writtenScreenshotPath = await writeScreenshotBuffer({
      buffer: screenshotBuffer,
      outputPath: screenshotPath,
      projectRootReal: projectRoot,
      allowExternal: options.allowExternalScreenshot,
      overwrite: options.overwrite,
    });
    result.screenshot = writtenScreenshotPath;
    result.checks.push({
      name: 'optional screenshot written',
      pass: true,
      detail: writtenScreenshotPath,
    });
  }

  const noRuntimeErrors = [
    result.diagnostics.pageErrors,
    result.diagnostics.webErrors,
    result.diagnostics.consoleErrors,
    result.diagnostics.crashes,
  ].every(items => items.length === 0);
  result.checks.push({
    name: 'no page, context, console, or crash errors',
    pass: noRuntimeErrors,
    detail: {
      pageErrors: result.diagnostics.pageErrors,
      webErrors: result.diagnostics.webErrors,
      consoleErrors: result.diagnostics.consoleErrors,
      crashes: result.diagnostics.crashes,
    },
  });
} catch (error) {
  result.error = String(error.stack ?? error);
  if (error.attempts) result.resolutionAttempts = error.attempts;
  if (error.candidates) result.resolutionCandidates = error.candidates;
  const lowerError = result.error.toLowerCase();
  if (lowerError.includes('executable doesn\'t exist') || lowerError.includes('browser was not found')) {
    result.hint = 'The selected matching managed Chromium binary is missing. Run the repository-local Playwright install chromium command only if installation is in scope.';
  } else if (!result.packageName) {
    result.hint = 'Pass --package-dir for the workspace package inside --project-root that owns playwright or @playwright/test; no ancestor or global package will be used.';
  }
} finally {
  await cleanupHandles();
}

const checksByName = new Map(result.checks.map(check => [check.name, check]));
result.missingRequiredChecks = REQUIRED_CHECKS.filter(name => !checksByName.has(name));
const requiredChecksPass = REQUIRED_CHECKS.every(name => checksByName.get(name)?.pass === true);
const allEmittedChecksPass = result.checks.every(check => check.pass === true);

result.passed = result.error === null
  && result.missingRequiredChecks.length === 0
  && requiredChecksPass
  && allEmittedChecksPass
  && result.cleanup.contextClosed
  && result.cleanup.browserClosed
  && result.cleanup.errors.length === 0;

if (result.packageDir) {
  result.packageDirRelativeToProject = relative(projectRoot, result.packageDir) || '.';
}

console.log(JSON.stringify(result, null, 2));
process.exitCode = result.passed ? 0 : 1;
