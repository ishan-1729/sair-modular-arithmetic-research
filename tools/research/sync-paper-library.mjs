#!/usr/bin/env node

/**
 * Reproducibly acquire the paper catalog declared in research/sources.json.
 *
 * Browser-visible source checks and every acquisition happen through the
 * repository's pinned Playwright and its managed Chromium.  The script never
 * attaches to an installed/personal browser profile.
 */

import {
  closeSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
  fsyncSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from 'node:path';

const EXPECTED_PLAYWRIGHT_VERSION = '1.61.1';
const NAVIGATION_TIMEOUT_MS = 60_000;
const REQUEST_TIMEOUT_MS = 90_000;
const MAX_ATTEMPTS = 3;
const RETRY_BASE_DELAY_MS = 750;
const SCRIPT_PATH = 'tools/research/sync-paper-library.mjs';

function usage() {
  return [
    'Usage: node tools/research/sync-paper-library.mjs [options]',
    '',
    'Options:',
    '  --project-root <path>  Repository root (default: current directory)',
    '  --force                Reacquire papers even when the cached SHA-256 is valid',
    '  --help, -h             Show this help text',
  ].join('\n');
}

function parseArgs(argv) {
  const options = { projectRoot: process.cwd(), force: false, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--project-root') {
      const value = argv[index + 1];
      if (!value) throw new Error('--project-root requires a value');
      options.projectRoot = value;
      index += 1;
    } else if (argument === '--force') {
      options.force = true;
    } else if (argument === '--help' || argument === '-h') {
      options.help = true;
    } else {
      throw new Error(`unknown argument: ${argument}`);
    }
  }
  return options;
}

function isWithin(root, candidate) {
  const pathRelative = relative(root, candidate);
  return pathRelative === ''
    || (!isAbsolute(pathRelative)
      && pathRelative !== '..'
      && !pathRelative.startsWith(`..${sep}`));
}

function portablePathKey(path) {
  // The repository is commonly hosted on case-insensitive Windows storage.
  // Treat case aliases as the same managed path on every platform so a catalog
  // remains portable and selected-file protection cannot be bypassed by case.
  return resolve(path).toLowerCase();
}

function assertNoSymlinkedComponents(root, candidate, label) {
  const pathRelative = relative(root, candidate);
  let current = root;
  for (const component of ['', ...pathRelative.split(sep).filter(Boolean)]) {
    if (component) current = join(current, component);
    if (existsSync(current) && lstatSync(current).isSymbolicLink()) {
      throw new Error(`refusing ${label} through symbolic link: ${current}`);
    }
  }
}

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function atomicWrite(path, bytes) {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = join(
    dirname(path),
    `.${basename(path)}.${process.pid}.${Date.now()}.tmp`,
  );
  let descriptor;
  try {
    descriptor = openSync(temporary, 'wx', 0o644);
    writeFileSync(descriptor, bytes);
    fsyncSync(descriptor);
    closeSync(descriptor);
    descriptor = undefined;
    renameSync(temporary, path);
  } catch (error) {
    if (descriptor !== undefined) closeSync(descriptor);
    if (existsSync(temporary)) unlinkSync(temporary);
    throw error;
  }
}

function compactText(value, maximum = 500) {
  if (typeof value !== 'string') return null;
  const normalized = value.replace(/\s+/g, ' ').trim();
  return normalized ? normalized.slice(0, maximum) : null;
}

function uniqueStrings(values) {
  return [...new Set(values.filter(value => typeof value === 'string' && value.trim())
    .map(value => value.trim()))].sort((left, right) => left.localeCompare(right));
}

function uniqueStringsInOrder(values) {
  return [...new Set(values.filter(value => typeof value === 'string' && value.trim())
    .map(value => value.trim()))];
}

function findBlockedMarker(text) {
  const normalized = String(text ?? '').toLowerCase();
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

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertArray(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
}

function validateArxivId(value, label) {
  if (typeof value !== 'string') throw new Error(`${label} must be a string`);
  const match = value.trim().match(
    /^((?:[a-z-]+(?:\.[A-Z]{2})?\/\d{7}|\d{4}\.\d{4,5}))(v\d+)?$/i,
  );
  if (!match) throw new Error(`${label} is not a supported arXiv identifier: ${value}`);
  return { base: match[1], requestedVersion: match[2] ?? null };
}

function validateHttpUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${label} must be a valid URL`);
  }
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`${label} must use HTTP or HTTPS`);
  }
  return parsed.href;
}

function sourceMention(sourceFile, citationMatch) {
  if (typeof sourceFile !== 'string' || !sourceFile.trim()) return null;
  return {
    source_file: sourceFile.trim().replaceAll('\\', '/'),
    citation_match: typeof citationMatch === 'string' && citationMatch.trim()
      ? citationMatch.trim()
      : null,
  };
}

function normalizeMentions(mentions) {
  const deduplicated = new Map();
  for (const mention of mentions.filter(Boolean)) {
    const key = `${mention.source_file}\u0000${mention.citation_match ?? ''}`;
    deduplicated.set(key, mention);
  }
  return [...deduplicated.values()].sort((left, right) => (
    left.source_file.localeCompare(right.source_file)
      || String(left.citation_match ?? '').localeCompare(String(right.citation_match ?? ''))
  ));
}

function arxivFilename(arxivId) {
  return `arxiv-${arxivId.replaceAll('/', '-')}.pdf`;
}

function arxivCitationsInText(text) {
  const matches = [];
  const patterns = [
    /https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf|html)\/((?:[a-z-]+(?:\.[A-Z]{2})?\/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)(?:\.pdf)?/gi,
    /\barXiv\s*:\s*((?:[a-z-]+(?:\.[A-Z]{2})?\/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)/gi,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      const before = text.slice(0, match.index);
      const lineNumber = before.split(/\r?\n/).length;
      const lineStart = before.lastIndexOf('\n') + 1;
      const nextNewline = text.indexOf('\n', match.index);
      const lineEnd = nextNewline === -1 ? text.length : nextNewline;
      matches.push({
        rawId: match[1],
        lineNumber,
        lineText: compactText(text.slice(lineStart, lineEnd), 500),
      });
    }
  }
  return matches;
}

function loadCatalog(projectRoot) {
  const path = join(projectRoot, 'research', 'sources.json');
  if (!existsSync(path)) throw new Error(`paper source catalog not found: ${path}`);
  const raw = readFileSync(path);
  let catalog;
  try {
    catalog = JSON.parse(raw.toString('utf8'));
  } catch (error) {
    throw new Error(`could not parse research/sources.json: ${error.message}`);
  }
  assertObject(catalog, 'research/sources.json');
  if (catalog.schema_version !== 1) {
    throw new Error('research/sources.json schema_version must be 1');
  }
  assertArray(catalog.collections, 'collections');
  assertArray(catalog.resolved_arxiv_papers, 'resolved_arxiv_papers');
  assertArray(catalog.web_papers, 'web_papers');

  const collections = new Map();
  for (const [index, collection] of catalog.collections.entries()) {
    assertObject(collection, `collections[${index}]`);
    if (typeof collection.id !== 'string' || !collection.id.trim()) {
      throw new Error(`collections[${index}].id must be a non-empty string`);
    }
    if (collections.has(collection.id)) {
      throw new Error(`duplicate collection id: ${collection.id}`);
    }
    assertArray(collection.source_files, `collections[${index}].source_files`);
    if (typeof collection.include_arxiv_citations !== 'boolean') {
      throw new Error(`collections[${index}].include_arxiv_citations must be boolean`);
    }
    const explicit = collection.explicit_arxiv_ids ?? [];
    assertArray(explicit, `collections[${index}].explicit_arxiv_ids`);
    const explicitIds = explicit.map((value, explicitIndex) => (
      validateArxivId(value, `collections[${index}].explicit_arxiv_ids[${explicitIndex}]`).base
    ));
    collections.set(collection.id, {
      id: collection.id,
      sourceFiles: uniqueStrings(collection.source_files.map(value => String(value).replaceAll('\\', '/'))),
      explicitIds,
    });
  }

  const arxivById = new Map();
  for (const [index, paper] of catalog.resolved_arxiv_papers.entries()) {
    assertObject(paper, `resolved_arxiv_papers[${index}]`);
    const parsed = validateArxivId(
      paper.arxiv_id,
      `resolved_arxiv_papers[${index}].arxiv_id`,
    );
    assertArray(paper.collections, `resolved_arxiv_papers[${index}].collections`);
    for (const collectionId of paper.collections) {
      if (!collections.has(collectionId)) {
        throw new Error(`arXiv ${parsed.base} names unknown collection: ${collectionId}`);
      }
    }
    const mention = sourceMention(paper.source_file, paper.citation_match);
    const existing = arxivById.get(parsed.base) ?? {
      key: `arxiv:${parsed.base}`,
      kind: 'arxiv',
      arxivId: parsed.base,
      requestedVersion: parsed.requestedVersion,
      collections: [],
      sourceMentions: [],
      localPath: `research/papers/${arxivFilename(parsed.base)}`,
    };
    if (existing.requestedVersion && parsed.requestedVersion
        && existing.requestedVersion !== parsed.requestedVersion) {
      throw new Error(`conflicting requested versions for arXiv ${parsed.base}`);
    }
    existing.requestedVersion ??= parsed.requestedVersion;
    existing.collections.push(...paper.collections);
    if (mention) existing.sourceMentions.push(mention);
    arxivById.set(parsed.base, existing);
  }

  // Explicit IDs are part of the acquisition contract even if a catalog
  // generator omitted a corresponding resolved citation row.
  for (const collection of collections.values()) {
    for (const arxivId of collection.explicitIds) {
      const existing = arxivById.get(arxivId) ?? {
        key: `arxiv:${arxivId}`,
        kind: 'arxiv',
        arxivId,
        requestedVersion: null,
        collections: [],
        sourceMentions: [],
        localPath: `research/papers/${arxivFilename(arxivId)}`,
      };
      existing.collections.push(collection.id);
      for (const sourceFile of collection.sourceFiles) {
        existing.sourceMentions.push(sourceMention(sourceFile, `explicit arXiv id ${arxivId}`));
      }
      arxivById.set(arxivId, existing);
    }
  }

  for (const paper of arxivById.values()) {
    paper.collections = uniqueStrings(paper.collections);
    paper.sourceMentions = normalizeMentions(paper.sourceMentions);
  }

  for (const collection of collections.values()) {
    const original = catalog.collections.find(candidate => candidate.id === collection.id);
    if (!original.include_arxiv_citations) continue;
    for (const sourceFile of collection.sourceFiles) {
      const sourcePath = resolve(projectRoot, sourceFile);
      if (!isWithin(projectRoot, sourcePath)) {
        throw new Error(`collection ${collection.id} source_file escapes the project root`);
      }
      if (!existsSync(sourcePath) || !statSync(sourcePath).isFile()) {
        throw new Error(`collection ${collection.id} source_file not found: ${sourceFile}`);
      }
      const sourceText = readFileSync(sourcePath, 'utf8');
      for (const citation of arxivCitationsInText(sourceText)) {
        const parsed = validateArxivId(
          citation.rawId,
          `${sourceFile}:${citation.lineNumber} arXiv citation`,
        );
        const existing = arxivById.get(parsed.base) ?? {
          key: `arxiv:${parsed.base}`,
          kind: 'arxiv',
          arxivId: parsed.base,
          requestedVersion: parsed.requestedVersion,
          collections: [],
          sourceMentions: [],
          localPath: `research/papers/${arxivFilename(parsed.base)}`,
        };
        if (existing.requestedVersion && parsed.requestedVersion
            && existing.requestedVersion !== parsed.requestedVersion) {
          throw new Error(`conflicting requested versions for arXiv ${parsed.base}`);
        }
        existing.requestedVersion ??= parsed.requestedVersion;
        existing.collections.push(collection.id);
        existing.sourceMentions.push(sourceMention(
          sourceFile,
          `${sourceFile}:${citation.lineNumber}: ${citation.lineText ?? citation.rawId}`,
        ));
        arxivById.set(parsed.base, existing);
      }
    }
  }

  for (const paper of arxivById.values()) {
    paper.collections = uniqueStrings(paper.collections);
    paper.sourceMentions = normalizeMentions(paper.sourceMentions);
  }

  const webItems = [];
  const webIds = new Set();
  for (const [index, paper] of catalog.web_papers.entries()) {
    assertObject(paper, `web_papers[${index}]`);
    if (typeof paper.id !== 'string' || !/^[a-z0-9][a-z0-9._-]*$/i.test(paper.id)) {
      throw new Error(`web_papers[${index}].id must be a filesystem-safe identifier`);
    }
    if (webIds.has(paper.id)) throw new Error(`duplicate web paper id: ${paper.id}`);
    webIds.add(paper.id);
    if (typeof paper.title !== 'string' || !paper.title.trim()) {
      throw new Error(`web_papers[${index}].title must be a non-empty string`);
    }
    assertArray(paper.authors ?? [], `web_papers[${index}].authors`);
    const declaredAuthors = uniqueStrings((paper.authors ?? []).map(value => String(value)));
    if (declaredAuthors.length === 0) {
      throw new Error(`web_papers[${index}].authors must contain at least one author`);
    }
    if (!['browser_print', 'direct_pdf'].includes(paper.acquisition_mode)) {
      throw new Error(`web_papers[${index}] has unsupported acquisition_mode`);
    }
    assertArray(paper.collections, `web_papers[${index}].collections`);
    for (const collectionId of paper.collections) {
      if (!collections.has(collectionId)) {
        throw new Error(`web paper ${paper.id} names unknown collection: ${collectionId}`);
      }
    }
    if (typeof paper.local_path !== 'string' || !paper.local_path.endsWith('.pdf')) {
      throw new Error(`web_papers[${index}].local_path must name a PDF`);
    }
    const localPath = paper.local_path.replaceAll('\\', '/');
    const target = resolve(projectRoot, localPath);
    const papersRoot = resolve(projectRoot, 'research', 'papers');
    if (!isWithin(papersRoot, target) || target === papersRoot) {
      throw new Error(`web paper ${paper.id} local_path must stay under research/papers`);
    }
    const pdfUrl = paper.acquisition_mode === 'direct_pdf'
      ? validateHttpUrl(paper.pdf_url, `web_papers[${index}].pdf_url`)
      : null;
    webItems.push({
      key: `web:${paper.id}`,
      kind: 'web',
      id: paper.id,
      declaredTitle: paper.title.trim(),
      declaredAuthors,
      publicationYear: Number.isInteger(paper.publication_year)
        ? paper.publication_year
        : null,
      doi: typeof paper.doi === 'string' && paper.doi.trim() ? paper.doi.trim() : null,
      url: validateHttpUrl(paper.url, `web_papers[${index}].url`),
      pdfUrl,
      acquisitionMode: paper.acquisition_mode,
      collections: uniqueStrings(paper.collections),
      sourceMentions: normalizeMentions([
        sourceMention(paper.source_file, paper.citation_match),
      ]),
      localPath,
    });
  }

  const items = [...arxivById.values(), ...webItems]
    .sort((left, right) => left.key.localeCompare(right.key));
  if (items.length === 0) throw new Error('research/sources.json declares no papers');
  const itemByPortablePath = new Map();
  for (const item of items) {
    const pathKey = portablePathKey(resolve(projectRoot, item.localPath));
    const prior = itemByPortablePath.get(pathKey);
    if (prior) {
      throw new Error(
        `paper paths must be unique ignoring case: ${prior.localPath} and ${item.localPath}`,
      );
    }
    itemByPortablePath.set(pathKey, item);
  }
  return { path, raw, catalog, items };
}

function loadLocalPlaywright(projectRoot) {
  const manifestPath = join(projectRoot, 'package.json');
  if (!existsSync(manifestPath)) throw new Error(`package.json not found: ${manifestPath}`);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const declarations = [
    manifest.dependencies,
    manifest.devDependencies,
    manifest.optionalDependencies,
  ];
  const declaredVersion = declarations
    .map(field => field?.playwright)
    .find(value => value !== undefined);
  if (declaredVersion !== EXPECTED_PLAYWRIGHT_VERSION) {
    throw new Error(
      `package.json must directly pin playwright ${EXPECTED_PLAYWRIGHT_VERSION}; found ${declaredVersion ?? 'no declaration'}`,
    );
  }
  const requireFromProject = createRequire(manifestPath);
  const entryPath = realpathSync(requireFromProject.resolve('playwright'));
  const installedManifestPath = realpathSync(requireFromProject.resolve('playwright/package.json'));
  if (!isWithin(projectRoot, entryPath) || !isWithin(projectRoot, installedManifestPath)) {
    throw new Error('playwright resolved outside the repository root');
  }
  const installed = JSON.parse(readFileSync(installedManifestPath, 'utf8'));
  if (installed.version !== EXPECTED_PLAYWRIGHT_VERSION) {
    throw new Error(
      `installed Playwright is ${installed.version}; expected ${EXPECTED_PLAYWRIGHT_VERSION}`,
    );
  }
  const playwright = requireFromProject('playwright');
  return { chromium: playwright.chromium, version: installed.version };
}

function loadManifest(path) {
  if (!existsSync(path)) return null;
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(path, 'utf8'));
  } catch (error) {
    throw new Error(`could not parse existing research/manifest.json: ${error.message}`);
  }
  assertObject(manifest, 'research/manifest.json');
  if (!Array.isArray(manifest.papers)) {
    throw new Error('research/manifest.json papers must be an array');
  }
  return manifest;
}

function isPdfBytes(bytes) {
  return bytes.length >= 5 && bytes.subarray(0, 5).toString('ascii') === '%PDF-';
}

function validateCachedPaper(projectRoot, item, entry) {
  if (!entry || !['downloaded', 'cached'].includes(entry.status)
      || typeof entry.sha256 !== 'string') return null;
  if (entry.local_path !== item.localPath) return null;
  if (item.kind === 'arxiv' && item.requestedVersion
      && entry.version !== item.requestedVersion) return null;
  const target = resolve(projectRoot, item.localPath);
  const papersRoot = resolve(projectRoot, 'research', 'papers');
  if (!isWithin(papersRoot, target) || target === papersRoot) return null;
  assertNoSymlinkedComponents(papersRoot, target, 'cached paper access');
  if (!existsSync(target) || !statSync(target).isFile()) return null;
  const bytes = readFileSync(target);
  if (!isPdfBytes(bytes)) return null;
  const actualHash = sha256Bytes(bytes);
  if (actualHash !== entry.sha256 || bytes.length !== entry.bytes) return null;
  return { bytes: bytes.length, sha256: actualHash };
}

function assertManagedPaperPathsStable(projectRoot, existingEntries, selectedItems) {
  const papersRoot = resolve(projectRoot, 'research', 'papers');
  for (const item of selectedItems) {
    const prior = existingEntries.get(item.key);
    if (!prior) continue;
    if (typeof prior.local_path !== 'string' || !prior.local_path.trim()) {
      throw new Error(`existing paper ${item.key} is missing local_path`);
    }
    if (prior.local_path === item.localPath) continue;

    const priorTarget = resolve(projectRoot, prior.local_path);
    if (!isWithin(papersRoot, priorTarget) || priorTarget === papersRoot) {
      throw new Error(`existing paper ${item.key} has a path outside research/papers`);
    }
    if (existsSync(priorTarget)) {
      throw new Error(
        `paper ${item.key} changes local_path while the prior file exists; remove the old catalog entry and sync before assigning the new path`,
      );
    }
  }
}

function pruneStalePaperFiles(projectRoot, staleEntries, selectedItems) {
  const papersRoot = resolve(projectRoot, 'research', 'papers');
  const selectedPaths = new Set(
    selectedItems.map(item => portablePathKey(resolve(projectRoot, item.localPath))),
  );
  const pruned = [];

  for (const entry of staleEntries) {
    if (!entry || typeof entry.local_path !== 'string' || !entry.local_path.trim()) {
      throw new Error('refusing to discard a stale manifest entry missing local_path');
    }
    const target = resolve(projectRoot, entry.local_path);
    if (!isWithin(papersRoot, target) || target === papersRoot) {
      throw new Error(`refusing to prune stale path outside research/papers: ${entry.local_path}`);
    }
    if (selectedPaths.has(portablePathKey(target))) continue;
    assertNoSymlinkedComponents(papersRoot, target, 'stale paper cleanup');
    if (!existsSync(target)) continue;

    const fileStatus = lstatSync(target);
    if (!fileStatus.isFile() || fileStatus.isSymbolicLink()) {
      throw new Error(`refusing to prune non-regular stale paper: ${entry.local_path}`);
    }
    if (typeof entry.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(entry.sha256)) {
      throw new Error(`refusing to prune stale paper without a recorded SHA-256: ${entry.local_path}`);
    }
    if (sha256File(target) !== entry.sha256.toLowerCase()) {
      throw new Error(`refusing to prune locally modified stale paper: ${entry.local_path}`);
    }

    unlinkSync(target);
    pruned.push(entry.local_path);
  }

  return pruned;
}

function diagnosticsFor(page) {
  const diagnostics = {
    page_errors: [],
    console_errors: [],
    request_failures: [],
    http_failures: [],
  };
  page.on('pageerror', error => {
    if (diagnostics.page_errors.length < 10) diagnostics.page_errors.push(String(error));
  });
  page.on('console', message => {
    if (message.type() === 'error' && diagnostics.console_errors.length < 10) {
      diagnostics.console_errors.push(message.text().slice(0, 500));
    }
  });
  page.on('requestfailed', request => {
    if (diagnostics.request_failures.length < 10) {
      diagnostics.request_failures.push({
        method: request.method(),
        url: request.url().slice(0, 500),
        error: request.failure()?.errorText ?? null,
      });
    }
  });
  page.on('response', response => {
    if (response.status() >= 400 && diagnostics.http_failures.length < 10) {
      diagnostics.http_failures.push({ status: response.status(), url: response.url().slice(0, 500) });
    }
  });
  return diagnostics;
}

async function newRestrictedContext(browser) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
    locale: 'en-US',
    timezoneId: 'UTC',
    colorScheme: 'light',
    reducedMotion: 'no-preference',
    serviceWorkers: 'block',
    acceptDownloads: false,
  });
  await context.route('**/*', async route => {
    if (!['GET', 'HEAD', 'OPTIONS'].includes(route.request().method())) {
      await route.abort('blockedbyclient');
      return;
    }
    await route.continue();
  });
  return context;
}

async function inspectArxivAndDownload(browser, item) {
  let context;
  let contextClosed = false;
  try {
    context = await newRestrictedContext(browser);
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const diagnostics = diagnosticsFor(page);
    const landingUrl = `https://arxiv.org/abs/${item.arxivId}`;
    const response = await page.goto(landingUrl, {
      waitUntil: 'domcontentloaded',
      timeout: NAVIGATION_TIMEOUT_MS,
    });
    await page.locator('body').waitFor({ state: 'visible', timeout: 15_000 });
    const bodyText = compactText(await page.locator('body').innerText(), 2_500) ?? '';
    const pageTitle = await page.title();
    const blockedMarker = findBlockedMarker(`${pageTitle}\n${bodyText}`);
    const finalUrl = page.url();
    const status = response?.status() ?? null;
    if (!response || status < 200 || status >= 400) {
      throw new Error(`arXiv landing page returned HTTP ${status ?? 'unknown'}`);
    }
    if (blockedMarker) throw new Error(`arXiv landing page was blocked: ${blockedMarker}`);
    const hostname = new URL(finalUrl).hostname.toLowerCase();
    if (hostname !== 'arxiv.org' && !hostname.endsWith('.arxiv.org')) {
      throw new Error(`arXiv landing page redirected to unexpected host: ${hostname}`);
    }

    const metadata = await page.evaluate(() => {
      const metaValues = name => [...document.querySelectorAll(`meta[name="${name}"]`)]
        .map(node => node.getAttribute('content')?.trim())
        .filter(Boolean);
      const firstText = selector => document.querySelector(selector)?.textContent?.trim() ?? null;
      const licenseLink = document.querySelector(
        'a[href*="creativecommons.org/licenses"], a[href*="arxiv.org/licenses"]',
      );
      return {
        title: metaValues('citation_title')[0]
          ?? firstText('h1.title')?.replace(/^Title:\s*/i, '')
          ?? null,
        authors: metaValues('citation_author').length
          ? metaValues('citation_author')
          : [...document.querySelectorAll('div.authors a')]
            .map(node => node.textContent?.trim()).filter(Boolean),
        citationDate: metaValues('citation_date')[0] ?? null,
        citationPdfUrl: metaValues('citation_pdf_url')[0]
          ?? document.querySelector('a.download-pdf')?.href
          ?? null,
        submissionHistory: firstText('.submission-history'),
        license: licenseLink
          ? { label: licenseLink.textContent?.trim() || null, url: licenseLink.href }
          : null,
        heading: firstText('h1'),
      };
    });
    if (!metadata.title || metadata.authors.length === 0) {
      throw new Error('arXiv landing page did not expose title and author metadata');
    }

    const versions = [];
    const versionPattern = /\[v(\d+)\]\s*([^\n(]+(?:\([^)]*\)[^\n(]*)?)\s*\([^\n)]*(?:KB|MB|bytes?)[^\n)]*\)/gi;
    for (const match of String(metadata.submissionHistory ?? '').matchAll(versionPattern)) {
      versions.push({
        version: `v${Number(match[1])}`,
        submitted_at_raw: compactText(match[2], 200),
      });
    }
    if (versions.length === 0) {
      const fallbackPattern = /\[v(\d+)\]\s*([^\n]+)/gi;
      for (const match of String(metadata.submissionHistory ?? '').matchAll(fallbackPattern)) {
        versions.push({ version: `v${Number(match[1])}`, submitted_at_raw: compactText(match[2], 200) });
      }
    }
    versions.sort((left, right) => Number(left.version.slice(1)) - Number(right.version.slice(1)));
    const discoveredVersion = versions.at(-1)?.version ?? null;
    const version = item.requestedVersion ?? discoveredVersion;
    if (!version) throw new Error('could not determine the latest arXiv version');
    if (item.requestedVersion && discoveredVersion
        && Number(item.requestedVersion.slice(1)) > Number(discoveredVersion.slice(1))) {
      throw new Error(`requested ${item.requestedVersion}, but arXiv exposes only ${discoveredVersion}`);
    }

    // Construct a version-pinned URL even when citation_pdf_url points at the
    // unversioned latest document.  context.request shares this browser
    // context's cookies and storage with the inspected landing page.
    const pinnedPdfUrl = `https://arxiv.org/pdf/${item.arxivId}${version}`;
    const pdfResponse = await context.request.get(pinnedPdfUrl, {
      timeout: REQUEST_TIMEOUT_MS,
      failOnStatusCode: false,
      maxRedirects: 10,
    });
    const pdfStatus = pdfResponse.status();
    const pdfBytes = Buffer.from(await pdfResponse.body());
    const contentType = pdfResponse.headers()['content-type'] ?? null;
    if (pdfStatus < 200 || pdfStatus >= 300) {
      throw new Error(`version-pinned arXiv PDF returned HTTP ${pdfStatus}`);
    }
    if (!isPdfBytes(pdfBytes)) {
      throw new Error(`arXiv response is not a PDF (content-type ${contentType ?? 'unknown'})`);
    }

    return {
      bytes: pdfBytes,
      metadata: {
        title: compactText(metadata.title, 1_000),
        authors: uniqueStringsInOrder(metadata.authors),
        arxiv_id: item.arxivId,
        version,
        submitted_date: versions[0]?.submitted_at_raw ?? metadata.citationDate,
        latest_version_date: versions.at(-1)?.submitted_at_raw ?? metadata.citationDate,
        versions,
        license: metadata.license,
        landing_url: landingUrl,
        pdf_url: pinnedPdfUrl,
        http: {
          status: pdfStatus,
          content_type: contentType,
          final_url: pdfResponse.url(),
        },
        browser_evidence: {
          requested_url: landingUrl,
          final_url: finalUrl,
          status,
          title: compactText(pageTitle, 500),
          heading: compactText(metadata.heading, 500),
          visible_text_sample: bodyText.slice(0, 500),
          blocked_marker: blockedMarker,
          diagnostics,
          context_closed: false,
        },
      },
    };
  } finally {
    if (context) {
      await context.close();
      contextClosed = true;
    } else {
      contextClosed = true;
    }
    // The returned object is mutable and its evidence is updated by the caller
    // after this function resolves; context closure failures reject the attempt.
    void contextClosed;
  }
}

async function inspectWebPaperAndAcquire(browser, item) {
  let context;
  let result;
  try {
    context = await newRestrictedContext(browser);
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const diagnostics = diagnosticsFor(page);
    const response = await page.goto(item.url, {
      waitUntil: 'domcontentloaded',
      timeout: NAVIGATION_TIMEOUT_MS,
    });
    await page.locator('body').waitFor({ state: 'visible', timeout: 15_000 });
    const bodyText = compactText(await page.locator('body').innerText(), 2_500) ?? '';
    const pageTitle = await page.title();
    const blockedMarker = findBlockedMarker(`${pageTitle}\n${bodyText}`);
    const status = response?.status() ?? null;
    if (!response || status < 200 || status >= 400) {
      throw new Error(`web paper returned HTTP ${status ?? 'unknown'}`);
    }
    if (blockedMarker) throw new Error(`web paper was blocked: ${blockedMarker}`);
    const heading = await page.locator('h1').count()
      ? compactText(await page.locator('h1').first().innerText(), 500)
      : null;
    let pdfBytes;
    let pdfHttp = null;
    if (item.acquisitionMode === 'browser_print') {
      const article = page.locator('article, d-article').first();
      if (await article.count()) {
        await article.waitFor({ state: 'visible', timeout: 15_000 });
      }
      await page.evaluate(async () => {
        if (document.fonts?.ready) await document.fonts.ready;
      });
      await page.waitForFunction(
        () => [...document.images].every(image => image.complete),
        undefined,
        { timeout: 30_000 },
      );
      await page.emulateMedia({ media: 'print' });
      pdfBytes = Buffer.from(await page.pdf({
        format: 'A4',
        printBackground: true,
        preferCSSPageSize: true,
        displayHeaderFooter: false,
        tagged: true,
      }));
    } else {
      const pdfResponse = await context.request.get(item.pdfUrl, {
        timeout: REQUEST_TIMEOUT_MS,
        failOnStatusCode: false,
        maxRedirects: 10,
      });
      pdfBytes = Buffer.from(await pdfResponse.body());
      pdfHttp = {
        status: pdfResponse.status(),
        content_type: pdfResponse.headers()['content-type'] ?? null,
        final_url: pdfResponse.url(),
      };
      if (pdfHttp.status < 200 || pdfHttp.status >= 300) {
        throw new Error(`direct PDF returned HTTP ${pdfHttp.status}`);
      }
    }
    if (!isPdfBytes(pdfBytes)) throw new Error('web acquisition output is not a PDF');
    result = {
      bytes: pdfBytes,
      metadata: {
        title: item.declaredTitle,
        source_title: compactText(pageTitle, 1_000),
        landing_url: item.url,
        pdf_url: item.pdfUrl,
        acquisition_mode: item.acquisitionMode,
        http: pdfHttp,
        browser_evidence: {
          requested_url: item.url,
          final_url: page.url(),
          status,
          title: compactText(pageTitle, 500),
          heading,
          visible_text_sample: bodyText.slice(0, 500),
          blocked_marker: blockedMarker,
          diagnostics,
          context_closed: false,
        },
      },
    };
    return result;
  } finally {
    if (context) await context.close();
    if (result) result.metadata.browser_evidence.context_closed = true;
  }
}

async function acquireWithRetries(browser, item) {
  let lastError;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      const acquired = item.kind === 'arxiv'
        ? await inspectArxivAndDownload(browser, item)
        : await inspectWebPaperAndAcquire(browser, item);
      acquired.attempts = attempt;
      // inspectArxivAndDownload returns after its finally closed the context.
      acquired.metadata.browser_evidence.context_closed = true;
      return acquired;
    } catch (error) {
      lastError = error;
      if (attempt < MAX_ATTEMPTS) {
        await new Promise(resolveDelay => setTimeout(
          resolveDelay,
          RETRY_BASE_DELAY_MS * (2 ** (attempt - 1)),
        ));
      }
    }
  }
  const wrapped = new Error(lastError?.message ?? 'paper acquisition failed');
  wrapped.cause = lastError;
  wrapped.attempts = MAX_ATTEMPTS;
  throw wrapped;
}

function baseEntry(item) {
  const entry = {
    id: item.key,
    kind: item.kind,
    collections: item.collections,
    source_mentions: item.sourceMentions,
    local_path: item.localPath,
  };
  if (item.kind === 'arxiv') entry.arxiv_id = item.arxivId;
  else {
    entry.web_id = item.id;
    entry.title = item.declaredTitle;
    entry.authors = item.declaredAuthors;
    entry.publication_year = item.publicationYear;
    entry.doi = item.doi;
    entry.acquisition_mode = item.acquisitionMode;
  }
  return entry;
}

function availableEntry(item, acquired, retrievedAt) {
  return {
    ...baseEntry(item),
    ...acquired.metadata,
    status: 'downloaded',
    retrieved_at: retrievedAt,
    bytes: acquired.bytes.length,
    sha256: sha256Bytes(acquired.bytes),
    acquisition_attempts: acquired.attempts,
  };
}

function failedEntry(item, error, attemptedAt) {
  return {
    ...baseEntry(item),
    status: 'failed',
    attempted_at: attemptedAt,
    acquisition_attempts: error.attempts ?? MAX_ATTEMPTS,
    error: compactText(error.stack ?? error.message ?? String(error), 2_000),
  };
}

function configFieldsUpdated(entry, item) {
  const updated = {
    ...entry,
    status: 'cached',
    collections: item.collections,
    source_mentions: item.sourceMentions,
    local_path: item.localPath,
  };
  if (item.kind === 'arxiv') updated.arxiv_id = item.arxivId;
  else {
    updated.web_id = item.id;
    updated.title = item.declaredTitle;
    updated.authors = item.declaredAuthors;
    updated.publication_year = item.publicationYear;
    updated.doi = item.doi;
    updated.acquisition_mode = item.acquisitionMode;
  }
  return updated;
}

function stablePapers(entries) {
  return [...entries.values()].sort((left, right) => left.id.localeCompare(right.id));
}

function writeState(manifestPath, checksumsPath, manifest, entries) {
  manifest.papers = stablePapers(entries);
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  const checksumLines = manifest.papers
    .filter(entry => ['downloaded', 'cached'].includes(entry.status)
      && entry.sha256 && entry.local_path)
    .map(entry => `${entry.sha256}  ${entry.local_path}`)
    .sort((left, right) => left.localeCompare(right));
  atomicWrite(manifestPath, manifestText);
  atomicWrite(checksumsPath, checksumLines.length ? `${checksumLines.join('\n')}\n` : '');
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    console.error(usage());
    return 2;
  }
  if (options.help) {
    console.log(usage());
    return 0;
  }

  let projectRoot;
  let catalog;
  let localPlaywright;
  let existingManifest;
  try {
    projectRoot = realpathSync(resolve(options.projectRoot));
    catalog = loadCatalog(projectRoot);
    localPlaywright = loadLocalPlaywright(projectRoot);
    existingManifest = loadManifest(join(projectRoot, 'research', 'manifest.json'));
  } catch (error) {
    console.error(`configuration error: ${error.message}`);
    return 2;
  }

  const manifestPath = join(projectRoot, 'research', 'manifest.json');
  const checksumsPath = join(projectRoot, 'research', 'checksums.sha256');
  const existingEntries = new Map(
    (existingManifest?.papers ?? [])
      .filter(entry => entry && typeof entry.id === 'string')
      .map(entry => [entry.id, entry]),
  );
  try {
    assertManagedPaperPathsStable(projectRoot, existingEntries, catalog.items);
  } catch (error) {
    console.error(`configuration error: ${error.message}`);
    return 2;
  }
  const selectedKeys = new Set(catalog.items.map(item => item.key));
  const staleEntries = [...existingEntries]
    .filter(([identifier]) => !selectedKeys.has(identifier));
  // Preserve stale entries in every incomplete checkpoint. They are removed
  // from the manifest only after all safe filesystem pruning succeeds, so a
  // failed or interrupted run retains the hashes needed for a retry.
  const entries = new Map(existingEntries);
  const catalogHash = sha256Bytes(catalog.raw);
  const manifest = {
    schema_version: 1,
    kind: 'sair-research-paper-library',
    generated_at: existingManifest?.generated_at ?? new Date().toISOString(),
    updated_at: existingManifest?.updated_at ?? new Date().toISOString(),
    source_catalog: 'research/sources.json',
    source_catalog_sha256: catalogHash,
    selected_paper_ids: catalog.items.map(item => item.key),
    sync_complete: false,
    tool: {
      script: SCRIPT_PATH,
      playwright_version: localPlaywright.version,
      chromium_mode: 'managed-regular-chromium-new-headless',
      chromium_version: existingManifest?.tool?.chromium_version ?? null,
    },
    papers: [],
  };

  let browser;
  let failed = 0;
  try {
    browser = await localPlaywright.chromium.launch({
      headless: true,
      channel: 'chromium',
    });
    manifest.tool.chromium_version = browser.version();

    for (const [index, item] of catalog.items.entries()) {
      const label = `[${index + 1}/${catalog.items.length}] ${item.key}`;
      const prior = entries.get(item.key);
      const cache = validateCachedPaper(projectRoot, item, prior);
      if (cache && !options.force) {
        const updated = configFieldsUpdated(prior, item);
        updated.status = 'cached';
        entries.set(item.key, updated);
        if (JSON.stringify(updated) !== JSON.stringify(prior)) {
          manifest.updated_at = new Date().toISOString();
        }
        console.log(`${label} cached (${cache.bytes} bytes)`);
        writeState(manifestPath, checksumsPath, manifest, entries);
        continue;
      }

      console.log(`${label} acquiring${options.force && cache ? ' (--force)' : ''}`);
      try {
        const acquired = await acquireWithRetries(browser, item);
        const target = resolve(projectRoot, item.localPath);
        const papersRoot = resolve(projectRoot, 'research', 'papers');
        if (!isWithin(papersRoot, target) || target === papersRoot) {
          throw new Error('local paper path must stay under research/papers');
        }
        assertNoSymlinkedComponents(papersRoot, target, 'paper acquisition');
        atomicWrite(target, acquired.bytes);
        const entry = availableEntry(item, acquired, new Date().toISOString());
        entries.set(item.key, entry);
        manifest.updated_at = new Date().toISOString();
        console.log(`${label} ok (${entry.bytes} bytes, ${entry.sha256.slice(0, 12)})`);
      } catch (error) {
        failed += 1;
        if (cache && prior) {
          entries.set(item.key, {
            ...configFieldsUpdated(prior, item),
            last_refresh_attempt_at: new Date().toISOString(),
            last_refresh_error: compactText(error.stack ?? error.message ?? String(error), 2_000),
          });
        } else {
          entries.set(item.key, failedEntry(item, error, new Date().toISOString()));
        }
        manifest.updated_at = new Date().toISOString();
        console.error(`${label} failed: ${error.message}`);
      }
      writeState(manifestPath, checksumsPath, manifest, entries);
    }
  } catch (error) {
    console.error(`browser error: ${error.stack ?? error}`);
    failed += 1;
  } finally {
    if (browser) {
      try {
        await browser.close();
      } catch (error) {
        console.error(`browser cleanup failed: ${error.message}`);
        failed += 1;
      }
    }
  }

  if (failed > 0) {
    manifest.sync_complete = false;
    writeState(manifestPath, checksumsPath, manifest, entries);
    console.error(`paper sync completed with ${failed} failure(s)`);
    return 1;
  }
  try {
    const pruned = pruneStalePaperFiles(
      projectRoot,
      staleEntries.map(([, entry]) => entry),
      catalog.items,
    );
    for (const localPath of pruned) console.log(`pruned stale PDF ${localPath}`);
    for (const [identifier] of staleEntries) entries.delete(identifier);
    if (pruned.length > 0) manifest.updated_at = new Date().toISOString();
  } catch (error) {
    manifest.sync_complete = false;
    manifest.updated_at = new Date().toISOString();
    manifest.stale_cleanup_error = compactText(error.stack ?? error.message ?? String(error), 2_000);
    writeState(manifestPath, checksumsPath, manifest, entries);
    console.error(`stale paper cleanup failed: ${error.message}`);
    return 1;
  }
  delete manifest.stale_cleanup_error;
  manifest.sync_complete = true;
  manifest.completed_at = new Date().toISOString();
  manifest.updated_at = manifest.completed_at;
  writeState(manifestPath, checksumsPath, manifest, entries);
  console.log(`paper sync complete: ${catalog.items.length} paper(s) available`);
  return 0;
}

export { assertManagedPaperPathsStable, pruneStalePaperFiles };

const invokedAsScript = process.argv[1]
  ? pathToFileURL(realpathSync(resolve(process.argv[1]))).href === import.meta.url
  : false;
if (invokedAsScript) process.exitCode = await main();
