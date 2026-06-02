#!/usr/bin/env node
import { spawn } from "node:child_process";
import { accessSync } from "node:fs";
import { copyFile, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DEFAULT_TIMEOUT_MS = 300000;
const SAME_IDENTITY_PERCENT_THRESHOLD = 50;
// Minimum gap (percentage points) between the weakest George match and the
// strongest non-George score. The 50% boundary alone passes even if both
// sides crowd it; this guards the *separation*, so an embedding-quality
// regression (e.g. a swapped SFace channel order) that compresses scores
// without crossing 50% still fails. Observed margin is ~40pts; 8 is a safe
// floor across browsers/ORT builds.
const MIN_SEPARATION_MARGIN = 8;

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    socket.addEventListener("message", (message) => this.handleMessage(message));
  }

  static connect(wsUrl) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl);
      socket.addEventListener("open", () => resolve(new CdpClient(socket)), { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
  }

  send(method, params = {}, timeout = 30000) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out waiting for CDP response to ${method}`));
      }, timeout);
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        }
      });
    });
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) || [];
    listeners.push(listener);
    this.listeners.set(method, listeners);
  }

  handleMessage(message) {
    const payload = JSON.parse(message.data);
    if (payload.id && this.pending.has(payload.id)) {
      const pending = this.pending.get(payload.id);
      this.pending.delete(payload.id);
      if (payload.error) pending.reject(new Error(payload.error.message));
      else pending.resolve(payload.result || {});
      return;
    }
    if (payload.method) {
      for (const listener of this.listeners.get(payload.method) || []) {
        listener(payload.params || {});
      }
    }
  }

  close() {
    this.socket.close();
  }
}

const options = parseArgs(process.argv.slice(2));
if (!options.url) {
  throw new Error("Usage: node tools/chrome_smoke.mjs --url file:///absolute/path/index.html [--timeout-ms 300000]");
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const timeoutMs = Number(options.timeoutMs || DEFAULT_TIMEOUT_MS);
const chromePath = options.chromePath || findChromePath();
const keepProfile = options.keepProfile === "true";
const runCopy = options.skipCopy !== "true";

const anonymizedDir = path.join(repoRoot, "test-data", "facetrace-standalone-model-migration", "docx-cropped-images");
const georgeDir = path.join(repoRoot, "test-data", "facetrace-standalone-model-migration", "george");
const anonymizedFiles = (await listImageFiles(anonymizedDir)).sort();
const georgeFiles = (await listImageFiles(georgeDir)).sort();
const georgeReference = path.join(georgeDir, "george1.webp");
const georgeGroup = path.join(georgeDir, "george4.jpg");
const anonymizedSmoke = anonymizedFiles[0];

if (anonymizedFiles.length !== 332) {
  throw new Error(`Expected 332 anonymized fixture images, found ${anonymizedFiles.length}`);
}
if (georgeFiles.length !== 8) {
  throw new Error(`Expected 8 George fixture images, found ${georgeFiles.length}`);
}
for (const file of [anonymizedSmoke, georgeReference, georgeGroup]) {
  if (!file) throw new Error("Missing required smoke fixture image.");
}

const output = {
  status: "started",
  chromePath,
  startedAt: new Date().toISOString(),
  inputUrl: options.url,
  runs: []
};

let copyDir = null;
try {
  output.runs.push(await runChromeSmoke({
    label: "source-index",
    url: options.url,
    chromePath,
    timeoutMs,
    fullFixtures: true
  }));

  if (runCopy) {
    copyDir = await mkdtemp(path.join(tmpdir(), "facetrace-single-file-"));
    const copyPath = path.join(copyDir, "index.html");
    await copyFile(fileURLToPath(new URL(options.url)), copyPath);
    output.copiedIndexPath = copyPath;
    output.runs.push(await runChromeSmoke({
      label: "copied-index-empty-dir",
      url: pathToFileURL(copyPath).href,
      chromePath,
      timeoutMs,
      fullFixtures: false
    }));
  }

  const failures = output.runs.filter((run) => run.status !== "ok");
  output.status = failures.length ? "failed" : "ok";
  console.log(JSON.stringify(output, null, 2));
  process.exitCode = failures.length ? 1 : 0;
} catch (error) {
  output.status = "failed";
  output.error = error?.stack || error?.message || String(error);
  console.log(JSON.stringify(output, null, 2));
  process.exitCode = 1;
} finally {
  if (copyDir && options.keepCopy !== "true") {
    await rm(copyDir, { recursive: true, force: true });
  }
}

async function runChromeSmoke({ label, url, chromePath, timeoutMs, fullFixtures }) {
  const profileDir = options.profileDir
    ? path.resolve(options.profileDir)
    : await mkdtemp(path.join(tmpdir(), `facetrace-${label}-chrome-`));
  if (options.profileDir) {
    await mkdir(profileDir, { recursive: true });
    await rm(path.join(profileDir, "DevToolsActivePort"), { force: true });
  }

  const result = {
    label,
    url,
    profileDir,
    status: "started",
    browser: "",
    modelStatus: "",
    modelReadyMs: null,
    networkRequests: [],
    externalRequests: [],
    failedRequests: [],
    consoleErrors: [],
    consoleWarnings: [],
    pageErrors: [],
    smoke: null,
    fixtures: null
  };

  const chrome = spawn(chromePath, [
    "--headless=new",
    "--remote-debugging-port=0",
    `--user-data-dir=${profileDir}`,
    "--disable-background-networking",
    "--disable-sync",
    "--disable-extensions",
    "--disable-component-update",
    "--no-first-run",
    "--no-default-browser-check",
    "--host-resolver-rules=MAP * 0.0.0.0,EXCLUDE 127.0.0.1",
    "about:blank"
  ], { stdio: ["ignore", "ignore", "pipe"] });

  let stderr = "";
  chrome.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  let cdp = null;
  try {
    const target = await waitForPageDevtoolsTarget(profileDir, timeoutMs);
    cdp = await CdpClient.connect(target.wsUrl);
    await cdp.send("Runtime.enable");
    await cdp.send("Log.enable").catch(() => {});
    await cdp.send("Page.enable");
    await cdp.send("DOM.enable");
    await cdp.send("Network.enable");
    await cdp.send("Network.setBlockedURLs", { urls: ["http://*/*", "https://*/*"] }).catch(() => {});

    cdp.on("Network.requestWillBeSent", (event) => {
      const requestUrl = event.request?.url || "";
      result.networkRequests.push(requestUrl);
      if (/^https?:\/\//i.test(requestUrl)) {
        result.externalRequests.push(requestUrl);
      }
    });
    cdp.on("Network.loadingFailed", (event) => {
      const requestUrl = event.requestId || "";
      result.failedRequests.push({
        requestId: requestUrl,
        errorText: event.errorText || "",
        blockedReason: event.blockedReason || ""
      });
    });
    cdp.on("Runtime.consoleAPICalled", (event) => {
      const text = (event.args || []).map((arg) => arg.value ?? arg.description ?? "").join(" ");
      if (event.type === "error") result.consoleErrors.push(text);
      if (event.type === "warning" || event.type === "warn") result.consoleWarnings.push(text);
    });
    cdp.on("Runtime.exceptionThrown", (event) => {
      result.pageErrors.push(event.exceptionDetails?.text || event.exceptionDetails?.exception?.description || "Runtime exception");
    });
    cdp.on("Log.entryAdded", (event) => {
      const entry = event.entry || {};
      if (entry.level === "error") result.consoleErrors.push(entry.text || "");
      if (entry.level === "warning") result.consoleWarnings.push(entry.text || "");
    });

    const readyStarted = Date.now();
    await cdp.send("Page.navigate", { url }, timeoutMs);
    await waitForCondition(cdp, "document.readyState === 'complete' || document.readyState === 'interactive'", timeoutMs);
    result.browser = await evaluate(cdp, "navigator.userAgent");
    await waitForCondition(cdp, `
      document.querySelector('#modelDot')?.classList.contains('ready')
      || document.querySelector('#modelDot')?.classList.contains('error')
    `, timeoutMs);
    result.modelStatus = await evaluate(cdp, "document.querySelector('#modelStatusText')?.textContent.trim() || ''");
    if (!(await evaluate(cdp, "document.querySelector('#modelDot')?.classList.contains('ready')"))) {
      throw new Error(`Model did not become ready: ${result.modelStatus}`);
    }
    result.modelReadyMs = Date.now() - readyStarted;

    result.smoke = await runBasicImageSmoke(cdp, timeoutMs);
    if (fullFixtures) {
      result.fixtures = await runFullFixtureSmoke(cdp, timeoutMs);
    }

    const meaningfulFailed = result.failedRequests.filter((item) => item.blockedReason || item.errorText);
    if (result.externalRequests.length || result.consoleErrors.length || result.pageErrors.length || meaningfulFailed.length) {
      throw new Error(`Browser smoke observed failures: external=${result.externalRequests.length}, consoleErrors=${result.consoleErrors.length}, pageErrors=${result.pageErrors.length}, failedRequests=${meaningfulFailed.length}`);
    }
    result.status = "ok";
  } catch (error) {
    result.status = "failed";
    result.error = error?.stack || error?.message || String(error);
    result.modelStatus ||= cdp
      ? await evaluate(cdp, "document.querySelector('#modelStatusText')?.textContent.trim() || ''", 2000).catch(() => "")
      : "";
    result.referenceMessage = cdp
      ? await evaluate(cdp, "document.querySelector('#referenceMessage')?.textContent.trim() || ''", 2000).catch(() => "")
      : "";
    result.candidateMessage = cdp
      ? await evaluate(cdp, "document.querySelector('#candidateMessage')?.textContent.trim() || ''", 2000).catch(() => "")
      : "";
    result.chromeStderrTail = stderr.split("\n").slice(-20).join("\n");
  } finally {
    cdp?.close();
    chrome.kill("SIGTERM");
    await Promise.race([
      new Promise((resolve) => chrome.once("exit", resolve)),
      sleep(3000).then(() => chrome.kill("SIGKILL"))
    ]);
    if (!options.profileDir && !keepProfile) {
      await rm(profileDir, { recursive: true, force: true });
    }
  }

  return result;
}

async function runBasicImageSmoke(cdp, timeoutMs) {
  await resetSession(cdp);
  await setFileInput(cdp, "#referenceInput", georgeReference);
  await waitForReferenceReady(cdp, timeoutMs);
  await setFileInput(cdp, "#candidateInput", [anonymizedSmoke, georgeReference, georgeGroup]);
  await waitForCandidateCount(cdp, 3, timeoutMs);
  const cards = await readResultCards(cdp);
  const byName = new Map(cards.map((card) => [card.fileName, card]));
  const anonymized = byName.get(path.basename(anonymizedSmoke));
  const georgeSelf = byName.get(path.basename(georgeReference));
  const georgeGroupCard = byName.get(path.basename(georgeGroup));
  if (!anonymized || anonymized.faceCount !== 1) {
    throw new Error(`Anonymized smoke fixture did not detect exactly one face: ${JSON.stringify(anonymized)}`);
  }
  if (!georgeSelf || georgeSelf.faceCount < 1 || georgeSelf.percent < SAME_IDENTITY_PERCENT_THRESHOLD) {
    throw new Error(`George reference smoke did not produce a same-identity score: ${JSON.stringify(georgeSelf)}`);
  }
  if (!georgeGroupCard || georgeGroupCard.faceCount < 1 || georgeGroupCard.percent < SAME_IDENTITY_PERCENT_THRESHOLD) {
    throw new Error(`George group smoke did not produce a same-identity score: ${JSON.stringify(georgeGroupCard)}`);
  }
  return { cards };
}

async function runFullFixtureSmoke(cdp, timeoutMs) {
  await resetSession(cdp);
  await setFileInput(cdp, "#referenceInput", georgeReference);
  await waitForReferenceReady(cdp, timeoutMs);
  await setFileInput(cdp, "#candidateInput", [...anonymizedFiles, ...georgeFiles]);
  await waitForCandidateCount(cdp, anonymizedFiles.length + georgeFiles.length, timeoutMs);
  const cards = await readResultCards(cdp);
  const byName = new Map(cards.map((card) => [card.fileName, card]));

  const anonymized = anonymizedFiles.map((file) => byName.get(path.basename(file))).filter(Boolean);
  const missingAnonymized = anonymizedFiles
    .map((file) => path.basename(file))
    .filter((name) => !byName.has(name));
  const wrongFaceCount = anonymized.filter((card) => card.faceCount !== 1);
  const anonymizedMatches = anonymized.filter((card) => Number.isFinite(card.percent) && card.percent >= SAME_IDENTITY_PERCENT_THRESHOLD);

  const george = georgeFiles.map((file) => byName.get(path.basename(file))).filter(Boolean);
  const missingGeorge = georgeFiles
    .map((file) => path.basename(file))
    .filter((name) => !byName.has(name));
  const georgeNoFace = george.filter((card) => card.faceCount < 1);
  const georgeNonMatches = george.filter((card) => !Number.isFinite(card.percent) || card.percent < SAME_IDENTITY_PERCENT_THRESHOLD);

  // Separation margin: weakest same-identity score vs. strongest cross-identity
  // score. Only meaningful once both buckets are fully scored.
  const georgePercents = george.map((card) => card.percent).filter(Number.isFinite);
  const anonPercents = anonymized.map((card) => card.percent).filter(Number.isFinite);
  const minGeorgePercent = georgePercents.length ? Math.min(...georgePercents) : null;
  const maxAnonPercent = anonPercents.length ? Math.max(...anonPercents) : null;
  const separationMargin = (minGeorgePercent !== null && maxAnonPercent !== null)
    ? minGeorgePercent - maxAnonPercent
    : null;
  const marginTooSmall = separationMargin === null || separationMargin < MIN_SEPARATION_MARGIN;

  if (cards.length !== anonymizedFiles.length + georgeFiles.length) {
    throw new Error(`Expected ${anonymizedFiles.length + georgeFiles.length} result cards, got ${cards.length}`);
  }
  if (missingAnonymized.length || wrongFaceCount.length || missingGeorge.length || georgeNoFace.length || georgeNonMatches.length || anonymizedMatches.length || marginTooSmall) {
    throw new Error(`Fixture assertion failed: ${JSON.stringify({
      missingAnonymized,
      wrongFaceCount: wrongFaceCount.slice(0, 10),
      missingGeorge,
      georgeNoFace,
      georgeNonMatches,
      anonymizedMatches: anonymizedMatches.slice(0, 10),
      separationMargin,
      minGeorgePercent,
      maxAnonPercent,
      requiredMargin: MIN_SEPARATION_MARGIN
    })}`);
  }

  return {
    totalCards: cards.length,
    anonymizedTotal: anonymized.length,
    anonymizedExactlyOne: anonymized.length - wrongFaceCount.length,
    anonymizedAboveSameIdentityThreshold: anonymizedMatches.length,
    georgeTotal: george.length,
    georgeWithFace: george.length - georgeNoFace.length,
    georgeAtOrAboveSameIdentityThreshold: george.length - georgeNonMatches.length,
    minGeorgePercent,
    maxAnonPercent,
    separationMargin,
    requiredMargin: MIN_SEPARATION_MARGIN,
    george: george.map((card) => ({
      fileName: card.fileName,
      faceCount: card.faceCount,
      percent: card.percent
    }))
  };
}

async function resetSession(cdp) {
  await evaluate(cdp, "document.querySelector('#clearButton')?.click()");
  await waitForCondition(cdp, "document.querySelectorAll('.result-card').length === 0", 10000).catch(() => {});
}

async function waitForReferenceReady(cdp, timeoutMs) {
  await waitForCondition(cdp, `
    document.querySelector('#referenceMessage')?.textContent.includes('Reference face ready')
    && document.querySelectorAll('#referenceFaces .face-choice').length >= 1
  `, timeoutMs);
}

async function waitForCandidateCount(cdp, expected, timeoutMs) {
  await waitForCondition(cdp, `
    document.querySelectorAll('.result-card').length === ${expected}
    && Array.from(document.querySelectorAll('.result-card')).every((card) => {
      const score = card.querySelector('.score')?.textContent.trim() || '';
      return /\\d+%|No score/.test(score);
    })
    && !document.querySelector('#progressWrap')?.classList.contains('active')
  `, timeoutMs);
}

async function readResultCards(cdp) {
  return evaluate(cdp, `
    Array.from(document.querySelectorAll('.result-card')).map((card) => {
      const fileName = card.querySelector('.filename')?.getAttribute('title')
        || card.querySelector('.filename')?.textContent.trim()
        || '';
      const scoreText = card.querySelector('.score')?.textContent.trim() || '';
      const percentMatch = scoreText.match(/(\\d+)%/);
      const badges = Array.from(card.querySelectorAll('.badge')).map((badge) => badge.textContent.trim());
      const faceBadge = badges.find((text) => /\\d+\\s+face/.test(text));
      const faceMatch = faceBadge ? faceBadge.match(/(\\d+)/) : null;
      return {
        fileName,
        scoreText,
        percent: percentMatch ? Number(percentMatch[1]) : null,
        badges,
        faceCount: faceMatch ? Number(faceMatch[1]) : null,
        statusText: card.querySelector('.status-row')?.textContent.trim() || ''
      };
    })
  `);
}

async function evaluate(cdp, expression, timeout = 30000) {
  const response = await cdp.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true
  }, timeout);
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.text || response.exceptionDetails.exception?.description || "Runtime.evaluate failed");
  }
  return response.result?.value;
}

async function setFileInput(cdp, selector, files) {
  const fileList = (Array.isArray(files) ? files : [files]).map((file) => path.resolve(file));
  const documentResult = await cdp.send("DOM.getDocument", { depth: -1, pierce: true });
  const nodeResult = await cdp.send("DOM.querySelector", {
    nodeId: documentResult.root.nodeId,
    selector
  });
  if (!nodeResult.nodeId) {
    throw new Error(`Could not find file input: ${selector}`);
  }
  await cdp.send("DOM.setFileInputFiles", {
    nodeId: nodeResult.nodeId,
    files: fileList
  });
  await evaluate(cdp, `
    (() => {
      const input = document.querySelector(${JSON.stringify(selector)});
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    })()
  `);
}

async function waitForCondition(cdp, expression, timeout) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeout) {
    try {
      if (await evaluate(cdp, `Boolean(${expression})`, 5000)) {
        return;
      }
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for condition: ${expression}${lastError ? ` (${lastError.message})` : ""}`);
}

async function waitForPageDevtoolsTarget(profileDir, timeout) {
  const portFile = path.join(profileDir, "DevToolsActivePort");
  const started = Date.now();
  let port = null;
  while (Date.now() - started < timeout) {
    try {
      const text = await readFile(portFile, "utf8");
      port = text.trim().split("\n")[0];
      break;
    } catch {
      await sleep(100);
    }
  }
  if (!port) {
    throw new Error("Timed out waiting for Chrome DevToolsActivePort.");
  }
  while (Date.now() - started < timeout) {
    const tabs = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json()).catch(() => []);
    const page = tabs.find((tab) => tab.type === "page" && tab.url === "about:blank")
      || tabs.find((tab) => tab.type === "page");
    if (page?.webSocketDebuggerUrl) {
      return { wsUrl: page.webSocketDebuggerUrl, port };
    }
    await sleep(100);
  }
  throw new Error("Timed out waiting for Chrome page target.");
}

async function listImageFiles(dir) {
  const { readdir } = await import("node:fs/promises");
  const names = await readdir(dir);
  return names
    .filter((name) => /\.(avif|bmp|gif|jpe?g|png|webp)$/i.test(name))
    .map((name) => path.join(dir, name));
}

function parseArgs(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2).replace(/-([a-z])/g, (_match, char) => char.toUpperCase());
    const next = args[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = "true";
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

function findChromePath() {
  if (process.env.CHROME_PATH) {
    return process.env.CHROME_PATH;
  }
  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium"
  ];
  for (const candidate of candidates) {
    try {
      accessSync(candidate);
      return candidate;
    } catch {
      // Continue looking.
    }
  }
  throw new Error("Could not find Chrome or Edge. Set CHROME_PATH or pass --chrome-path.");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
