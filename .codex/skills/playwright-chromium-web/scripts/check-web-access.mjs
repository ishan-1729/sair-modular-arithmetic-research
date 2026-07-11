#!/usr/bin/env node

import { existsSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

function usage() {
  return 'Usage: node check-web-access.mjs [--project-root <path>] [--timeout <ms>] [--headed] [--out <json>] <url>...';
}

function parseArgs(argv) {
  const options = {
    projectRoot: process.cwd(),
    timeout: 30000,
    headed: false,
    out: null,
    urls: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (['--project-root', '--timeout', '--out'].includes(argument)) {
      const value = argv[index + 1];
      if (!value) throw new Error(argument + ' requires a value');
      index += 1;
      if (argument === '--project-root') options.projectRoot = value;
      if (argument === '--timeout') options.timeout = Number(value);
      if (argument === '--out') options.out = value;
    } else if (argument === '--headed') {
      options.headed = true;
    } else if (argument === '--help' || argument === '-h') {
      console.log(usage());
      process.exit(0);
    } else {
      options.urls.push(argument);
    }
  }
  if (!Number.isFinite(options.timeout) || options.timeout < 1000 || options.timeout > 120000) {
    throw new Error('--timeout must be between 1000 and 120000 milliseconds');
  }
  if (options.urls.length === 0) throw new Error('at least one URL is required');
  options.urls = options.urls.map(value => {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) {
      throw new Error('unsupported URL protocol: ' + value);
    }
    return url.href;
  });
  return options;
}

function isWithin(root, candidate) {
  const pathRelative = relative(root, candidate);
  return pathRelative === ''
    || (!isAbsolute(pathRelative)
      && pathRelative !== '..'
      && !pathRelative.startsWith('..' + sep));
}

function loadLocalPlaywright(projectRoot) {
  const root = realpathSync(resolve(projectRoot));
  const manifestPath = join(root, 'package.json');
  if (!existsSync(manifestPath)) {
    throw new Error('package.json not found in project root: ' + root);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const fields = [
    manifest.dependencies,
    manifest.devDependencies,
    manifest.optionalDependencies,
  ];
  const declared = fields.some(field => field && Object.hasOwn(field, 'playwright'));
  if (!declared) {
    throw new Error('project package.json does not directly declare playwright');
  }
  const requireFromProject = createRequire(manifestPath);
  const entryPath = realpathSync(requireFromProject.resolve('playwright'));
  const installedManifestPath = realpathSync(
    requireFromProject.resolve('playwright/package.json'),
  );
  if (!isWithin(root, entryPath) || !isWithin(root, installedManifestPath)) {
    throw new Error('playwright resolved outside the project root');
  }
  const installed = JSON.parse(readFileSync(installedManifestPath, 'utf8'));
  const playwright = requireFromProject('playwright');
  return { root, chromium: playwright.chromium, version: installed.version };
}

function findBlockedMarker(text) {
  const normalized = text.toLowerCase();
  const markers = [
    'verify you are human',
    'captcha',
    'access denied',
    'checking your browser',
    'just a moment',
    'unusual traffic',
    'temporarily blocked',
    'enable cookies to continue',
  ];
  return markers.find(marker => normalized.includes(marker)) ?? null;
}

async function inspectUrl(browser, url, timeout) {
  const diagnostics = {
    pageErrors: [],
    webErrors: [],
    consoleErrors: [],
    requestFailures: [],
    httpFailures: [],
  };
  const result = {
    requestedUrl: url,
    finalUrl: null,
    status: null,
    title: null,
    heading: null,
    visibleTextSample: null,
    blockedMarker: null,
    accessible: false,
    diagnostics,
    contextClosed: false,
    error: null,
  };
  let context;
  try {
    context = await browser.newContext({
      viewport: { width: 1280, height: 800 },
      deviceScaleFactor: 1,
      locale: 'en-US',
      timezoneId: 'UTC',
      colorScheme: 'light',
      reducedMotion: 'no-preference',
      serviceWorkers: 'block',
      acceptDownloads: false,
    });
    context.on('weberror', event => {
      diagnostics.webErrors.push(String(event.error()));
    });
    await context.route('**/*', async route => {
      const method = route.request().method();
      if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
        await route.abort('blockedbyclient');
        return;
      }
      await route.continue();
    });
    const page = await context.newPage();
    page.setDefaultTimeout(Math.min(timeout, 10000));
    page.on('pageerror', error => {
      diagnostics.pageErrors.push(String(error));
    });
    page.on('console', message => {
      if (message.type() === 'error') diagnostics.consoleErrors.push(message.text());
    });
    page.on('requestfailed', request => {
      diagnostics.requestFailures.push({
        method: request.method(),
        url: request.url(),
        error: request.failure()?.errorText ?? null,
      });
    });
    page.on('response', response => {
      if (response.status() >= 400) {
        diagnostics.httpFailures.push({
          status: response.status(),
          url: response.url(),
        });
      }
    });

    const response = await page.goto(url, {
      waitUntil: 'domcontentloaded',
      timeout,
    });
    await page.locator('body').waitFor({
      state: 'visible',
      timeout: Math.min(timeout, 10000),
    });
    const headingLocator = page.locator('h1');
    const headingCount = await headingLocator.count();
    const bodyText = (await page.locator('body').innerText())
      .replace(/\s+/g, ' ')
      .trim();
    result.finalUrl = page.url();
    result.status = response?.status() ?? null;
    result.title = await page.title();
    result.heading = headingCount > 0
      ? (await headingLocator.first().innerText()).trim()
      : null;
    result.visibleTextSample = bodyText.slice(0, 500);
    result.blockedMarker = findBlockedMarker(
      result.title + '\n' + bodyText.slice(0, 2000),
    );
    result.accessible = Boolean(
      response
      && response.status() >= 200
      && response.status() < 400
      && bodyText.length > 0
      && !result.blockedMarker,
    );
  } catch (error) {
    result.error = String(error.stack ?? error);
  } finally {
    if (context) {
      try {
        await context.close();
        result.contextClosed = true;
      } catch (error) {
        result.error ??= 'context close failed: ' + error.message;
      }
    } else {
      result.contextClosed = true;
    }
  }
  return result;
}

let options;
try {
  options = parseArgs(process.argv.slice(2));
} catch (error) {
  console.error(error.message);
  console.error(usage());
  process.exit(2);
}

let browser;
const report = {
  kind: 'playwright-chromium-web-access',
  generatedAt: new Date().toISOString(),
  playwrightVersion: null,
  chromiumMode: options.headed
    ? 'managed-regular-chromium-headed'
    : 'managed-regular-chromium-new-headless',
  chromiumVersion: null,
  results: [],
  cleanup: { browserClosed: false, error: null },
  error: null,
};

try {
  const local = loadLocalPlaywright(options.projectRoot);
  report.playwrightVersion = local.version;
  browser = await local.chromium.launch({
    headless: !options.headed,
    channel: 'chromium',
  });
  report.chromiumVersion = browser.version();
  for (const url of options.urls) {
    report.results.push(await inspectUrl(browser, url, options.timeout));
  }
} catch (error) {
  report.error = String(error.stack ?? error);
} finally {
  if (browser) {
    try {
      await browser.close();
      report.cleanup.browserClosed = true;
    } catch (error) {
      report.cleanup.error = String(error);
    }
  } else {
    report.cleanup.browserClosed = true;
  }
}

const output = JSON.stringify(report, null, 2) + '\n';
if (options.out) {
  const outputPath = resolve(options.out);
  if (!existsSync(dirname(outputPath))) {
    throw new Error('output parent does not exist: ' + dirname(outputPath));
  }
  writeFileSync(outputPath, output, { encoding: 'utf8', flag: 'wx' });
} else {
  process.stdout.write(output);
}

const passed = !report.error
  && report.cleanup.browserClosed
  && report.results.every(item => item.accessible && item.contextClosed);
process.exitCode = passed ? 0 : 1;
