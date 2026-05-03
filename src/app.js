(() => {
  "use strict";

  // The directory base passed to face-api and TF.js loaders. The fetch shim
  // strips this prefix and serves bytes from the embedded bundle instead.
  const FACE_API_BASE = "facetrace-models";
  const ARCFACE_BASE = "facetrace-models/arcface";

  const MAX_ANALYSIS_SIDE = 1600;
  const THUMBNAIL_SIDE = 260;
  const FACE_CROP_SIDE = 144;
  const ARCFACE_INPUT_SIDE = 112;
  const ARCFACE_EMBEDDING_DIM = 256;
  const LOCALE_STORAGE_KEY = "facetrace.locale";
  const DEFAULT_LOCALE = "en";
  const SUPPORTED_LOCALES = ["en", "de", "fr"];
  const SEARCH_SET_FORMAT = "facetrace-search-set";
  const SEARCH_SET_ENVELOPE_FORMAT = "facetrace-search-set-envelope";
  const SEARCH_SET_SCHEMA_VERSION = 1;
  const SEARCH_SET_ENVELOPE_VERSION = 1;
  const SEARCH_SET_KEY_PREFIX = "ftsk1_";
  const AES_GCM_KEY_BYTES = 32;
  const AES_GCM_IV_BYTES = 12;
  const AES_GCM_TAG_BITS = 128;
  const MAX_SEARCH_SET_FILE_BYTES = 512 * 1024 * 1024;
  const DOWNLOAD_URL_TTL_MS = 60_000;

  // Calibration of cosine similarity to a user-facing percentage. The model
  // (SE-MobileFaceNet trained with ArcFace loss on MS1M) typically separates
  // same/different identities around cosine 0.32. The sigmoid is centered
  // there so that 50% maps to the empirical decision boundary.
  const COSINE_PERCENT_CENTER = 0.32;
  const COSINE_PERCENT_SLOPE = 12;

  // Quality thresholds used to flag (not reject) marginal faces.
  const MIN_DETECTOR_SCORE_FOR_GOOD = 0.55;
  const MAX_YAW_RATIO_FOR_GOOD = 1.6;       // 1.0 is frontal
  const MAX_PITCH_RATIO_FOR_GOOD = 1.6;
  const MIN_BLUR_VARIANCE_FOR_GOOD = 25;    // Laplacian variance, ~empirical

  const SUPPORTED_IMAGE_EXTENSION = /\.(avif|bmp|gif|jpe?g|png|webp)$/i;
  const READBACK_CONTEXT_OPTIONS = { alpha: false, willReadFrequently: true };
  const DRAW_CONTEXT_OPTIONS = { alpha: false, willReadFrequently: false };

  // ArcFace canonical 5-point landmarks at 112x112 (InsightFace standard).
  const ARCFACE_REFERENCE_LANDMARKS = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
  ];

  let analysisQueue = Promise.resolve();
  let arcfaceModel = null;
  let renderResultsScheduled = false;
  const objectUrls = new Set();
  const objectUrlBlobs = new Map();

  // The bundled face-api UMD ships TensorFlow.js v4 internally and exposes it
  // as faceapi.tf. We use that single engine for face-api detection, ArcFace
  // GraphModel inference, and tensor scope management — no second TF.js
  // bundle, no global kernel-registry conflicts.
  let tfRef = null;

  const state = {
    modelsReady: false,
    modelError: null,
    backend: "",
    modelStatusKey: "model.loading.initial",
    modelStatusVars: {},
    locales: {},
    locale: DEFAULT_LOCALE,
    progressKey: "progress.idle",
    progressVars: {},
    runToken: 0,
    processing: false,
    reference: null,
    referenceFaceIndex: 0,
    candidates: [],
    nextCandidateId: 1,
    nextImportedSetId: 1,
    sortMode: "similarity-desc"
  };

  const elements = {
    modelStatus: document.getElementById("modelStatus"),
    modelDot: document.getElementById("modelDot"),
    modelStatusText: document.getElementById("modelStatusText"),
    progressWrap: document.getElementById("progressWrap"),
    progressBar: document.getElementById("progressBar"),
    progressText: document.getElementById("progressText"),
    clearButton: document.getElementById("clearButton"),
    exportButton: document.getElementById("exportButton"),
    referencePickButton: document.getElementById("referencePickButton"),
    referenceInput: document.getElementById("referenceInput"),
    referenceDropzone: document.getElementById("referenceDropzone"),
    referencePreview: document.getElementById("referencePreview"),
    referenceMessage: document.getElementById("referenceMessage"),
    referenceFaces: document.getElementById("referenceFaces"),
    candidatePickButton: document.getElementById("candidatePickButton"),
    candidateInput: document.getElementById("candidateInput"),
    candidateDropzone: document.getElementById("candidateDropzone"),
    candidateMessage: document.getElementById("candidateMessage"),
    importSetButton: document.getElementById("importSetButton"),
    exportSetButton: document.getElementById("exportSetButton"),
    searchSetInput: document.getElementById("searchSetInput"),
    languageSelect: document.getElementById("languageSelect"),
    sortSelect: document.getElementById("sortSelect"),
    summaryCounts: document.getElementById("summaryCounts"),
    resultsList: document.getElementById("resultsList"),
    attributionDialog: document.getElementById("attributionDialog"),
    attributionDialogBody: document.getElementById("attributionDialogBody"),
    attributionDialogClose: document.getElementById("attributionDialogClose"),
    confirmDialog: document.getElementById("confirmDialog"),
    confirmDialogTitle: document.getElementById("confirmDialogTitle"),
    confirmDialogBody: document.getElementById("confirmDialogBody"),
    confirmDialogActions: document.getElementById("confirmDialogActions"),
    shareKeyDialog: document.getElementById("shareKeyDialog"),
    shareKeyValue: document.getElementById("shareKeyValue"),
    shareKeyCopyStatus: document.getElementById("shareKeyCopyStatus"),
    shareKeyCopyButton: document.getElementById("shareKeyCopyButton"),
    shareKeySaveButton: document.getElementById("shareKeySaveButton"),
    shareKeyContinueButton: document.getElementById("shareKeyContinueButton"),
    shareKeyCancelButton: document.getElementById("shareKeyCancelButton"),
    importKeyDialog: document.getElementById("importKeyDialog"),
    importKeyInput: document.getElementById("importKeyInput"),
    importKeyError: document.getElementById("importKeyError"),
    importKeyConfirmButton: document.getElementById("importKeyConfirmButton"),
    importKeyCancelButton: document.getElementById("importKeyCancelButton")
  };

  // ---------- generic helpers ----------

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;"
    })[char]);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function resolveLocale(locale) {
    const normalized = String(locale || "").toLowerCase().split("-")[0];
    return SUPPORTED_LOCALES.includes(normalized) ? normalized : DEFAULT_LOCALE;
  }

  function getInitialLocale() {
    try {
      const params = new URLSearchParams(window.location.search || "");
      const fromUrl = params.get("lang") || params.get("locale");
      if (fromUrl) return resolveLocale(fromUrl);
    } catch (_error) {
      // ignore URL parse errors
    }
    try {
      const saved = window.localStorage.getItem(LOCALE_STORAGE_KEY);
      if (saved) return resolveLocale(saved);
    } catch (_error) {
      // ignore storage access errors
    }
    return resolveLocale(window.navigator.language || DEFAULT_LOCALE);
  }

  const reportedMissingKeys = new Set();
  function getI18nText(key) {
    const current = state.locales[state.locale] || {};
    if (Object.prototype.hasOwnProperty.call(current, key)) {
      return current[key];
    }
    const fallback = state.locales[DEFAULT_LOCALE] || {};
    if (Object.prototype.hasOwnProperty.call(fallback, key)) {
      return fallback[key];
    }
    if (!reportedMissingKeys.has(key) && window.console && window.console.warn) {
      reportedMissingKeys.add(key);
      window.console.warn("[facetrace i18n] missing key:", key);
    }
    return `[${key}]`;
  }

  function interpolate(text, vars) {
    return String(text).replace(/\{([a-zA-Z0-9_]+)\}/g, (_match, name) => {
      if (!vars || !Object.prototype.hasOwnProperty.call(vars, name)) {
        return `{${name}}`;
      }
      return String(vars[name]);
    });
  }

  function t(key, vars) {
    return interpolate(getI18nText(key), vars);
  }

  function issueLabel(issueKey) {
    return t(issueKey);
  }

  function qualityListLabel(issueKeys) {
    const labels = (issueKeys || []).map(issueLabel);
    if (!labels.length) return "";
    try {
      return new Intl.ListFormat(state.locale, { style: "short", type: "conjunction" }).format(labels);
    } catch (_error) {
      return labels.join(", ");
    }
  }

  function sanitizeI18nHtml(html) {
    // Allow only bare <code> and </code> tags. Everything else is escaped, so
    // a malicious or accidental translation cannot inject markup or scripts.
    const allowedTag = /^<\/?code>$/i;
    return String(html).replace(/<[^>]*>/g, (match) => allowedTag.test(match) ? match : escapeHtml(match));
  }

  function applyI18nToDom() {
    document.documentElement.lang = state.locale;
    document.title = t("app.title");

    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.getAttribute("data-i18n"));
    });
    document.querySelectorAll("[data-i18n-html]").forEach((node) => {
      node.innerHTML = sanitizeI18nHtml(t(node.getAttribute("data-i18n-html")));
    });
    document.querySelectorAll("[data-i18n-aria-label]").forEach((node) => {
      node.setAttribute("aria-label", t(node.getAttribute("data-i18n-aria-label")));
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.getAttribute("data-i18n-placeholder")));
    });
  }

  function setLocale(locale, persist) {
    state.locale = resolveLocale(locale);
    if (persist) {
      try {
        window.localStorage.setItem(LOCALE_STORAGE_KEY, state.locale);
      } catch (_error) {
        // ignore storage access errors
      }
    }
    if (elements.languageSelect) {
      elements.languageSelect.value = state.locale;
    }
    applyI18nToDom();
    refreshModelStatusText();
    refreshProgressText();
    retranslatePlaceholderFileNames();
    renderReference();
    renderResults();
    updateCandidateMessage();
  }

  function retranslatePlaceholderFileNames() {
    const fallback = t("searchset.unnamed_file");
    for (const candidate of state.candidates) {
      if (candidate && candidate.fileNameIsPlaceholder) {
        candidate.fileName = fallback;
        if (candidate.result) candidate.result.fileName = fallback;
      }
    }
  }

  function refreshModelStatusText() {
    elements.modelStatusText.textContent = t(state.modelStatusKey, state.modelStatusVars);
  }

  function refreshProgressText() {
    elements.progressText.textContent = t(state.progressKey, state.progressVars);
  }

  function formatNumber(value, digits = 4) {
    return Number.isFinite(value) ? value.toFixed(digits) : "";
  }

  function safeFileName(value, fallback) {
    const cleaned = String(value || fallback || "facetrace")
      .trim()
      .replace(/[^a-z0-9._-]+/gi, "-")
      .replace(/^-+|-+$/g, "");
    return cleaned || fallback || "facetrace";
  }

  function exportTimestampSlug() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  }

  function safeExternalHref(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    try {
      const url = new URL(raw);
      return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
    } catch (_error) {
      return "";
    }
  }

  function textToBytes(text) {
    return new TextEncoder().encode(String(text));
  }

  function bytesToText(bytes) {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  }

  function bytesToBase64(bytes) {
    const chunkSize = 0x8000;
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      const chunk = bytes.subarray(offset, offset + chunkSize);
      binary += String.fromCharCode.apply(null, chunk);
    }
    return window.btoa(binary);
  }

  function base64ToBytes(base64) {
    const binary = window.atob(String(base64 || ""));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  function bytesToBase64Url(bytes) {
    return bytesToBase64(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  }

  function base64UrlToBytes(value) {
    const normalized = String(value || "").trim().replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4 || 4)) % 4);
    return base64ToBytes(padded);
  }

  function stableStringify(value) {
    if (Array.isArray(value)) {
      return `[${value.map(stableStringify).join(",")}]`;
    }
    if (value && typeof value === "object") {
      return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
    }
    return JSON.stringify(value);
  }

  async function sha256Base64Url(bytes) {
    const digest = await window.crypto.subtle.digest("SHA-256", bytes);
    return bytesToBase64Url(new Uint8Array(digest));
  }

  function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  function yieldToBrowser() {
    if ("requestIdleCallback" in window) {
      return new Promise((resolve) => window.requestIdleCallback(resolve, { timeout: 60 }));
    }
    return sleep(0);
  }

  function runAnalysisExclusive(task) {
    const previous = analysisQueue.catch(() => undefined);
    const current = previous.then(task);
    analysisQueue = current.catch(() => undefined);
    return current;
  }

  function get2dContext(canvas, options) {
    const context = canvas.getContext("2d", options);
    if (!context) {
      const error = new Error("error.canvas_unavailable");
      error.i18nKey = "error.canvas_unavailable";
      throw error;
    }
    return context;
  }

  function makeI18nError(key, vars) {
    const error = new Error(key);
    error.i18nKey = key;
    error.i18nVars = vars || {};
    return error;
  }

  function registerObjectUrl(url, blob) {
    objectUrls.add(url);
    if (blob) {
      objectUrlBlobs.set(url, blob);
    }
    return url;
  }

  function releaseObjectUrl(url) {
    if (typeof url === "string" && url.startsWith("blob:") && objectUrls.delete(url)) {
      objectUrlBlobs.delete(url);
      URL.revokeObjectURL(url);
    }
  }

  function releaseAnalysisUrls(analysis) {
    if (!analysis) return;
    releaseObjectUrl(analysis.thumbnail);
    if (Array.isArray(analysis.faces)) {
      for (const face of analysis.faces) {
        releaseObjectUrl(face && face.crop);
      }
    }
  }

  function releaseAllObjectUrls() {
    for (const url of objectUrls) {
      URL.revokeObjectURL(url);
    }
    objectUrls.clear();
    objectUrlBlobs.clear();
  }

  function releaseCandidateObjectUrls() {
    for (const candidate of state.candidates) {
      releaseAnalysisUrls(candidate.result);
    }
  }

  function normalizeAttribution(value) {
    if (!value || typeof value !== "object") return null;
    const attribution = {
      author: String(value.author || "").trim(),
      license: String(value.license || "").trim(),
      sourceLink: String(value.sourceLink || value.sourceUrl || "").trim(),
      notes: String(value.notes || "").trim()
    };
    return attribution.author || attribution.license || attribution.sourceLink || attribution.notes ? attribution : null;
  }

  function isLikelyImage(file) {
    return Boolean(file && (file.type.startsWith("image/") || SUPPORTED_IMAGE_EXTENSION.test(file.name || "")));
  }

  function base64ToUint8Array(base64) {
    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  // ---------- bundle decompression ----------

  async function decodeEmbeddedBundle() {
    const packed = window.FACETRACE_EMBEDDED_MODELS_GZIP_B64;
    if (typeof packed !== "string" || !packed) {
      throw makeI18nError("error.bundle_missing");
    }

    if (typeof DecompressionStream !== "function") {
      throw makeI18nError("error.decompression_missing");
    }

    const compressed = base64ToUint8Array(packed);
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
    const text = await new Response(stream).text();
    let map;
    try {
      map = JSON.parse(text);
    } catch (error) {
      throw makeI18nError("error.bundle_corrupt");
    }
    if (!map || typeof map !== "object") {
      throw makeI18nError("error.bundle_shape");
    }
    return map;
  }

  function freeEmbeddedBundle(bundle) {
    // Drop references so the GC can reclaim ~3 MB of model bytes plus the
    // ~2 MB compressed base64 once both face-api and ArcFace finish loading.
    try { window.FACETRACE_EMBEDDED_MODELS_GZIP_B64 = ""; } catch (_) {}
    if (bundle) {
      for (const key of Object.keys(bundle)) {
        delete bundle[key];
      }
    }
  }

  // ---------- offline fetch shim ----------

  function installOfflineModelFetch(bundle) {
    function requestName(input) {
      const raw = typeof input === "string" ? input : input && input.url ? input.url : "";
      try {
        const url = new URL(raw, window.location.href);
        return decodeURIComponent(url.pathname.split("/").pop() || "");
      } catch (_error) {
        return decodeURIComponent(String(raw).split("?")[0].split("#")[0].split("/").pop() || "");
      }
    }

    function offlineFetch(input, init) {
      const name = requestName(input);
      const entry = bundle[name];

      if (entry) {
        if (entry.kind === "json") {
          return Promise.resolve(new Response(entry.text, {
            status: 200,
            headers: { "content-type": "application/json" }
          }));
        }
        if (entry.kind === "binary") {
          const bytes = base64ToUint8Array(entry.base64);
          return Promise.resolve(new Response(bytes.buffer, {
            status: 200,
            headers: { "content-type": "application/octet-stream" }
          }));
        }
      }

      const raw = typeof input === "string" ? input : input && input.url ? input.url : "";
      if (/^https?:/i.test(raw)) {
        return Promise.reject(makeI18nError("error.network_disabled"));
      }
      return Promise.reject(makeI18nError("error.blocked_local_request", { name: name || "?" }));
    }

    window.fetch = offlineFetch;

    // face-api ships its own bundled TF.js (faceapi.tf). The model loaders
    // route their requests through faceapi.env.fetch, so patch that too.
    if (window.faceapi && faceapi.env && typeof faceapi.env.getEnv === "function") {
      try {
        faceapi.env.getEnv().fetch = offlineFetch;
      } catch (_error) {
        if (typeof faceapi.env.createBrowserEnv === "function" && typeof faceapi.env.setEnv === "function") {
          const browserEnv = faceapi.env.createBrowserEnv();
          browserEnv.fetch = offlineFetch;
          faceapi.env.setEnv(browserEnv);
        }
      }
    }
  }

  // ---------- TF.js backend selection ----------

  async function selectTfjsBackend() {
    if (!tfRef || typeof tfRef.setBackend !== "function") {
      throw makeI18nError("error.tf_unavailable");
    }
    // Prefer WebGL for ~10-50x speedup on most machines. Locked-down browsers
    // without WebGL fall back to the pure-JS CPU backend.
    for (const name of ["webgl", "cpu"]) {
      try {
        const ok = await tfRef.setBackend(name);
        if (ok) {
          await tfRef.ready();
          if (tfRef.getBackend() === name) return name;
        }
      } catch (_error) {
        // try next
      }
    }
    await tfRef.setBackend("cpu");
    await tfRef.ready();
    return tfRef.getBackend() || "cpu";
  }

  // ---------- model loading ----------

  async function loadModels() {
    setModelStatus("loading", "status.model.decoding");
    let bundle = null;
    try {
      if (!window.faceapi) {
        throw makeI18nError("error.faceapi_unavailable");
      }
      tfRef = window.faceapi.tf || null;
      if (!tfRef || typeof tfRef.loadGraphModel !== "function") {
        throw makeI18nError("error.faceapi_tf_unavailable");
      }

      bundle = await decodeEmbeddedBundle();
      installOfflineModelFetch(bundle);

      setModelStatus("loading", "status.model.select_backend");
      state.backend = await selectTfjsBackend();

      setModelStatus("loading", "status.model.loading_detector", { backend: state.backend });
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_BASE),
        faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_BASE)
      ]);

      setModelStatus("loading", "status.model.loading_arcface", { backend: state.backend });
      arcfaceModel = await tfRef.loadGraphModel(`${ARCFACE_BASE}/model.json`);

      // Warm-up pass: run one zero-input inference so WebGL programs and
      // CPU kernels are compiled now rather than on the first real face.
      tfRef.tidy(() => {
        const dummy = tfRef.zeros([1, ARCFACE_INPUT_SIDE, ARCFACE_INPUT_SIDE, 3], "float32");
        const out = arcfaceModel.predict(dummy);
        if (Array.isArray(out)) out.forEach((t) => t.dispose());
        else out.dispose();
      });

      state.modelsReady = true;
      state.modelError = null;
      setModelStatus("ready", "status.model.ready", { backend: state.backend });
      updateCandidateMessage();
      processCandidateQueue();
    } catch (error) {
      state.modelsReady = false;
      state.modelError = error;
      setModelStatus("error", "status.model.error", { message: normalizeError(error) });
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = t("status.model.unavailable_hint");
      updateCandidateMessage();
    } finally {
      // Free the embedded source bytes regardless of success/failure. On
      // failure there's nothing useful left to do with them anyway.
      freeEmbeddedBundle(bundle);
    }
  }

  function setModelStatus(kind, key, vars) {
    state.modelStatusKey = key;
    state.modelStatusVars = vars || {};
    elements.modelDot.className = `status-dot ${kind === "ready" ? "ready" : kind === "error" ? "error" : ""}`;
    refreshModelStatusText();
  }

  // ---------- 5-point alignment ----------

  function average68LandmarkRange(positions, fromIndex, toExclusive) {
    let sumX = 0;
    let sumY = 0;
    const count = toExclusive - fromIndex;
    for (let i = fromIndex; i < toExclusive; i += 1) {
      sumX += positions[i].x;
      sumY += positions[i].y;
    }
    return [sumX / count, sumY / count];
  }

  function fivePointFrom68(landmarks) {
    const positions = landmarks.positions;
    if (!positions || positions.length < 68) {
      return null;
    }
    return [
      average68LandmarkRange(positions, 36, 42),     // left eye centroid
      average68LandmarkRange(positions, 42, 48),     // right eye centroid
      [positions[30].x, positions[30].y],            // nose tip
      [positions[48].x, positions[48].y],            // left mouth corner
      [positions[54].x, positions[54].y]             // right mouth corner
    ];
  }

  // Closed-form 2D similarity transform (rotation + uniform scale + translation,
  // 4 DOF) from N source points to N reference points by least squares. The
  // returned [a, b, tx, ty] encodes the affine matrix [[a,-b,tx],[b,a,ty]].
  function similarityTransform(sourcePoints, referencePoints) {
    const n = sourcePoints.length;
    let meanSx = 0;
    let meanSy = 0;
    let meanRx = 0;
    let meanRy = 0;
    for (let i = 0; i < n; i += 1) {
      meanSx += sourcePoints[i][0];
      meanSy += sourcePoints[i][1];
      meanRx += referencePoints[i][0];
      meanRy += referencePoints[i][1];
    }
    meanSx /= n;
    meanSy /= n;
    meanRx /= n;
    meanRy /= n;

    let numA = 0;
    let numB = 0;
    let denom = 0;
    for (let i = 0; i < n; i += 1) {
      const sx = sourcePoints[i][0] - meanSx;
      const sy = sourcePoints[i][1] - meanSy;
      const rx = referencePoints[i][0] - meanRx;
      const ry = referencePoints[i][1] - meanRy;
      numA += sx * rx + sy * ry;
      numB += sx * ry - sy * rx;
      denom += sx * sx + sy * sy;
    }

    if (denom < 1e-9) {
      // Degenerate (all source points collapsed) — fall back to identity.
      return [1, 0, 0, 0];
    }

    const a = numA / denom;
    const b = numB / denom;
    const tx = meanRx - (a * meanSx - b * meanSy);
    const ty = meanRy - (b * meanSx + a * meanSy);
    return [a, b, tx, ty];
  }

  function alignedFaceCanvas(sourceCanvas, fivePoints) {
    const [a, b, tx, ty] = similarityTransform(fivePoints, ARCFACE_REFERENCE_LANDMARKS);
    const canvas = document.createElement("canvas");
    canvas.width = ARCFACE_INPUT_SIDE;
    canvas.height = ARCFACE_INPUT_SIDE;
    const context = get2dContext(canvas, READBACK_CONTEXT_OPTIONS);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.fillStyle = "#000000";
    context.fillRect(0, 0, ARCFACE_INPUT_SIDE, ARCFACE_INPUT_SIDE);
    // Canvas matrix is [a c e; b d f]. Our similarity matrix is [a -b tx; b a ty],
    // so set transform = (a, b, -b, a, tx, ty).
    context.setTransform(a, b, -b, a, tx, ty);
    context.drawImage(sourceCanvas, 0, 0);
    context.setTransform(1, 0, 0, 1, 0, 0);
    return canvas;
  }

  function mirroredCanvas(sourceCanvas) {
    const canvas = document.createElement("canvas");
    canvas.width = sourceCanvas.width;
    canvas.height = sourceCanvas.height;
    const context = get2dContext(canvas, READBACK_CONTEXT_OPTIONS);
    context.translate(sourceCanvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(sourceCanvas, 0, 0);
    return canvas;
  }

  // ---------- ArcFace embedding ----------

  function beginTensorScope() {
    // Wrap any tensor-allocating block so transient tensors are reclaimed
    // promptly even when long batches are processed back-to-back.
    const engine = tfRef && typeof tfRef.engine === "function" ? tfRef.engine() : null;
    if (!engine || typeof engine.startScope !== "function" || typeof engine.endScope !== "function") {
      return () => {};
    }
    engine.startScope();
    return () => {
      try { engine.endScope(); } catch (_error) { /* ignore */ }
    };
  }

  function l2NormalizeInPlace(vector) {
    let sumOfSquares = 0;
    for (let i = 0; i < vector.length; i += 1) {
      sumOfSquares += vector[i] * vector[i];
    }
    const norm = Math.sqrt(sumOfSquares);
    if (norm < 1e-9) return vector;
    for (let i = 0; i < vector.length; i += 1) {
      vector[i] /= norm;
    }
    return vector;
  }

  async function arcfaceEmbeddingForCanvas(canvas) {
    // Single-image inference. Tensors are managed manually because predict
    // returns a tensor that escapes the local tidy scope; the caller awaits
    // a typed array and we dispose the tensor before returning.
    const input = tfRef.tidy(() => {
      const pixels = tfRef.browser.fromPixels(canvas, 3);
      const normalized = pixels.toFloat().sub(127.5).div(128.0);
      return normalized.expandDims(0);
    });
    const output = arcfaceModel.predict(input);
    try {
      const data = await output.data();
      return new Float32Array(data);
    } finally {
      input.dispose();
      if (Array.isArray(output)) output.forEach((t) => t.dispose());
      else output.dispose();
    }
  }

  function averageDescriptors(left, right) {
    const out = new Float32Array(left.length);
    for (let i = 0; i < left.length; i += 1) {
      out[i] = (left[i] + right[i]) * 0.5;
    }
    return out;
  }

  async function descriptorWithFlipTta(alignedCanvas) {
    // Test-Time Augmentation: ArcFace is rotation-sensitive but learned to
    // be roughly mirror-invariant. Averaging the original and horizontally
    // flipped embeddings consistently improves matching on out-of-distribution
    // faces (1-3% on benchmarks). We L2-normalize at the end so cosine
    // similarity reduces to a simple dot product.
    const flipped = mirroredCanvas(alignedCanvas);
    try {
      const [a, b] = await Promise.all([
        arcfaceEmbeddingForCanvas(alignedCanvas),
        arcfaceEmbeddingForCanvas(flipped)
      ]);
      const merged = averageDescriptors(a, b);
      l2NormalizeInPlace(merged);
      return merged;
    } finally {
      flipped.width = 0;
      flipped.height = 0;
    }
  }

  // ---------- quality signals ----------

  function poseRatiosFromFivePoints(fivePoints) {
    const [leftEye, rightEye, nose, leftMouth, rightMouth] = fivePoints;

    // Yaw proxy: left half / right half of the line eye-to-eye relative to nose.
    // Frontal faces have ratio ~1; rotation skews one half.
    const leftHalf = Math.hypot(nose[0] - leftEye[0], nose[1] - leftEye[1]);
    const rightHalf = Math.hypot(rightEye[0] - nose[0], rightEye[1] - nose[1]);
    const yawRatio = leftHalf > 1e-6 && rightHalf > 1e-6
      ? Math.max(leftHalf / rightHalf, rightHalf / leftHalf)
      : Infinity;

    // Pitch proxy: distance eye-line-to-nose vs nose-to-mouth-line.
    const eyeMidY = (leftEye[1] + rightEye[1]) * 0.5;
    const mouthMidY = (leftMouth[1] + rightMouth[1]) * 0.5;
    const eyesToNose = Math.abs(nose[1] - eyeMidY);
    const noseToMouth = Math.abs(mouthMidY - nose[1]);
    const pitchRatio = eyesToNose > 1e-6 && noseToMouth > 1e-6
      ? Math.max(eyesToNose / noseToMouth, noseToMouth / eyesToNose)
      : Infinity;

    // Roll: in-plane rotation in degrees from the eye line.
    const rollRadians = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]);
    const rollDegrees = (rollRadians * 180) / Math.PI;

    return { yawRatio, pitchRatio, rollDegrees };
  }

  function laplacianVarianceForCanvas(canvas) {
    // Cheap blur estimator. Lower variance = blurrier image. Operates on a
    // small grayscale grid sampled from the aligned face, so runtime is
    // bounded by ARCFACE_INPUT_SIDE^2.
    const context = get2dContext(canvas, READBACK_CONTEXT_OPTIONS);
    const { data, width, height } = context.getImageData(0, 0, canvas.width, canvas.height);
    const grayscale = new Float32Array(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
      grayscale[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    let sum = 0;
    let sumOfSquares = 0;
    let count = 0;
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const center = grayscale[y * width + x];
        const top = grayscale[(y - 1) * width + x];
        const bottom = grayscale[(y + 1) * width + x];
        const left = grayscale[y * width + (x - 1)];
        const right = grayscale[y * width + (x + 1)];
        const lap = 4 * center - top - bottom - left - right;
        sum += lap;
        sumOfSquares += lap * lap;
        count += 1;
      }
    }
    if (count === 0) return 0;
    const mean = sum / count;
    return sumOfSquares / count - mean * mean;
  }

  function classifyQuality(quality) {
    const issues = [];
    if (Number.isFinite(quality.detectorScore) && quality.detectorScore < MIN_DETECTOR_SCORE_FOR_GOOD) {
      issues.push("quality.low_detector");
    }
    if (Number.isFinite(quality.yawRatio) && quality.yawRatio > MAX_YAW_RATIO_FOR_GOOD) {
      issues.push("quality.off_axis_yaw");
    }
    if (Number.isFinite(quality.pitchRatio) && quality.pitchRatio > MAX_PITCH_RATIO_FOR_GOOD) {
      issues.push("quality.off_axis_pitch");
    }
    if (Number.isFinite(quality.blurVariance) && quality.blurVariance < MIN_BLUR_VARIANCE_FOR_GOOD) {
      issues.push("quality.low_sharpness");
    }
    return issues;
  }

  // ---------- modal dialog helpers ----------

  function openDialog(dialog) {
    if (typeof dialog.showModal === "function" && !dialog.open) {
      dialog.showModal();
    } else if (!dialog.open) {
      dialog.setAttribute("open", "");
    }
  }

  function closeDialog(dialog) {
    if (!dialog.open) return;
    if (typeof dialog.close === "function") {
      dialog.close();
    } else {
      dialog.removeAttribute("open");
    }
  }

  function showConfirmDialog(options) {
    const dialog = elements.confirmDialog;
    elements.confirmDialogTitle.textContent = options.title || "";
    elements.confirmDialogBody.textContent = options.body || "";
    elements.confirmDialogActions.innerHTML = "";

    const opener = document.activeElement;

    return new Promise((resolve) => {
      let resolved = false;
      const buttons = [];

      const finish = (value) => {
        if (resolved) return;
        resolved = true;
        dialog.removeEventListener("close", onClose);
        for (const button of buttons) {
          button.element.removeEventListener("click", button.handler);
        }
        closeDialog(dialog);
        if (opener && typeof opener.focus === "function") {
          try { opener.focus(); } catch (_) {}
        }
        resolve(value);
      };

      const onClose = () => finish(null);
      dialog.addEventListener("close", onClose);

      (options.buttons || []).forEach((spec) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `button${spec.kind ? ` ${spec.kind}` : ""}`;
        button.textContent = spec.label;
        const handler = () => finish(spec.value);
        button.addEventListener("click", handler);
        elements.confirmDialogActions.appendChild(button);
        buttons.push({ element: button, handler });
      });

      openDialog(dialog);

      const focusTarget = elements.confirmDialogActions.querySelector("button.primary")
        || elements.confirmDialogActions.querySelector("button:not(.cancel)")
        || elements.confirmDialogActions.querySelector("button");
      if (focusTarget) {
        try { focusTarget.focus(); } catch (_) {}
      }
    });
  }

  function showShareKeyDialog(shareKey) {
    const dialog = elements.shareKeyDialog;
    elements.shareKeyValue.value = shareKey;
    elements.shareKeyCopyStatus.textContent = "";
    elements.shareKeyCopyStatus.classList.remove("error");

    const opener = document.activeElement;

    return new Promise((resolve) => {
      let resolved = false;

      const finish = (value) => {
        if (resolved) return;
        resolved = true;
        elements.shareKeyCopyButton.removeEventListener("click", onCopy);
        elements.shareKeySaveButton.removeEventListener("click", onSave);
        elements.shareKeyContinueButton.removeEventListener("click", onContinue);
        elements.shareKeyCancelButton.removeEventListener("click", onCancel);
        dialog.removeEventListener("close", onClose);
        elements.shareKeyValue.value = "";
        closeDialog(dialog);
        if (opener && typeof opener.focus === "function") {
          try { opener.focus(); } catch (_) {}
        }
        resolve(value);
      };

      async function onCopy() {
        elements.shareKeyCopyStatus.classList.remove("error");
        try {
          if (window.navigator && window.navigator.clipboard && window.navigator.clipboard.writeText) {
            await window.navigator.clipboard.writeText(shareKey);
            elements.shareKeyCopyStatus.textContent = t("dialog.share_key.copied");
            return;
          }
          throw new Error("clipboard unavailable");
        } catch (_error) {
          elements.shareKeyCopyStatus.classList.add("error");
          elements.shareKeyCopyStatus.textContent = t("dialog.share_key.copy_failed");
          try {
            elements.shareKeyValue.focus();
            elements.shareKeyValue.select();
          } catch (_) {}
        }
      }

      function onSave() {
        const blob = new Blob([shareKey + "\n"], { type: "text/plain;charset=utf-8" });
        downloadBlob(blob, `facetrace-share-key-${exportTimestampSlug()}.txt`);
      }

      const onContinue = () => finish(true);
      const onCancel = () => finish(false);
      const onClose = () => finish(false);

      elements.shareKeyCopyButton.addEventListener("click", onCopy);
      elements.shareKeySaveButton.addEventListener("click", onSave);
      elements.shareKeyContinueButton.addEventListener("click", onContinue);
      elements.shareKeyCancelButton.addEventListener("click", onCancel);
      dialog.addEventListener("close", onClose);

      openDialog(dialog);

      // Pre-select the key so a user can immediately Cmd/Ctrl+C even before
      // discovering the Copy button.
      try {
        elements.shareKeyValue.focus();
        elements.shareKeyValue.select();
      } catch (_) {}
    });
  }

  function promptForShareKey() {
    const dialog = elements.importKeyDialog;
    elements.importKeyInput.value = "";
    elements.importKeyError.textContent = "";

    const opener = document.activeElement;

    return new Promise((resolve) => {
      let resolved = false;

      const finish = (value) => {
        if (resolved) return;
        resolved = true;
        elements.importKeyConfirmButton.removeEventListener("click", onConfirm);
        elements.importKeyCancelButton.removeEventListener("click", onCancel);
        elements.importKeyInput.removeEventListener("keydown", onKeydown);
        dialog.removeEventListener("close", onClose);
        elements.importKeyInput.value = "";
        closeDialog(dialog);
        if (opener && typeof opener.focus === "function") {
          try { opener.focus(); } catch (_) {}
        }
        resolve(value);
      };

      const onConfirm = () => {
        const raw = elements.importKeyInput.value.trim();
        if (!raw) {
          elements.importKeyError.textContent = t("error.searchset_key_required");
          elements.importKeyInput.focus();
          return;
        }
        const encoded = raw.startsWith(SEARCH_SET_KEY_PREFIX) ? raw.slice(SEARCH_SET_KEY_PREFIX.length) : raw;
        try {
          const keyBytes = base64UrlToBytes(encoded);
          if (!keyBytes || keyBytes.length !== AES_GCM_KEY_BYTES) {
            throw new Error("invalid key length");
          }
        } catch (_error) {
          elements.importKeyError.textContent = t("error.searchset_key_invalid");
          elements.importKeyInput.focus();
          elements.importKeyInput.select();
          return;
        }
        finish(raw);
      };
      const onCancel = () => finish(null);
      const onClose = () => finish(null);
      const onKeydown = (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          onConfirm();
        }
      };

      elements.importKeyConfirmButton.addEventListener("click", onConfirm);
      elements.importKeyCancelButton.addEventListener("click", onCancel);
      elements.importKeyInput.addEventListener("keydown", onKeydown);
      dialog.addEventListener("close", onClose);

      openDialog(dialog);

      try { elements.importKeyInput.focus(); } catch (_) {}
    });
  }

  // ---------- input bindings ----------

  function bindDropzone(dropzone, input, onFiles) {
    const openPicker = () => input.click();
    dropzone.addEventListener("click", openPicker);
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPicker();
      }
    });
    input.addEventListener("change", () => {
      onFiles(Array.from(input.files || []));
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const files = Array.from(event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files : []);
      onFiles(files);
    });
  }

  elements.referencePickButton.addEventListener("click", () => elements.referenceInput.click());
  elements.candidatePickButton.addEventListener("click", () => elements.candidateInput.click());
  bindDropzone(elements.referenceDropzone, elements.referenceInput, handleReferenceFiles);
  bindDropzone(elements.candidateDropzone, elements.candidateInput, handleCandidateFiles);
  elements.importSetButton.addEventListener("click", () => elements.searchSetInput.click());
  elements.searchSetInput.addEventListener("change", () => {
    handleSearchSetFiles(Array.from(elements.searchSetInput.files || []));
    elements.searchSetInput.value = "";
  });

  elements.clearButton.addEventListener("click", () => {
    state.runToken += 1;
    state.processing = false;
    releaseAllObjectUrls();
    state.reference = null;
    state.referenceFaceIndex = 0;
    state.candidates = [];
    state.nextCandidateId = 1;
    state.nextImportedSetId = 1;
    hideProgress();
    renderReference();
    renderResults();
    updateCandidateMessage();
  });

  elements.exportButton.addEventListener("click", exportCsv);
  elements.exportSetButton.addEventListener("click", exportSearchSet);
  elements.resultsList.addEventListener("input", (event) => {
    const input = event.target && event.target.closest ? event.target.closest("[data-candidate-display-name]") : null;
    if (!input) return;
    const candidate = state.candidates.find((item) => String(item.id) === input.getAttribute("data-candidate-display-name"));
    if (!candidate) return;
    candidate.displayName = input.value;
    if (candidate.result) {
      candidate.result.displayName = input.value;
    }
  });
  elements.resultsList.addEventListener("click", (event) => {
    const button = event.target && event.target.closest ? event.target.closest("[data-candidate-attribution]") : null;
    if (!button) return;
    openAttributionDialog(button.getAttribute("data-candidate-attribution"));
  });
  elements.attributionDialogClose.addEventListener("click", () => {
    if (elements.attributionDialog.close) {
      elements.attributionDialog.close();
    } else {
      elements.attributionDialog.removeAttribute("open");
    }
  });
  elements.sortSelect.addEventListener("change", () => {
    state.sortMode = elements.sortSelect.value;
    renderResults();
  });

  // ---------- reference / candidate handling ----------

  async function handleReferenceFiles(files) {
    const imageFiles = files.filter(isLikelyImage);
    if (!imageFiles.length) {
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = t("error.image_unreadable_help");
      return;
    }

    state.runToken += 1;
    const token = state.runToken;
    releaseAnalysisUrls(state.reference);
    state.reference = {
      fileName: imageFiles[0].name,
      status: "processing",
      faces: [],
      thumbnail: "",
      errorKey: null
    };
    state.referenceFaceIndex = 0;
    renderReference();
    showProgress(0, 1, "progress.reference.processing");

    try {
      ensureModelsReady();
      await yieldToBrowser();
      const analysis = await analyzeImageFile(imageFiles[0]);
      if (token !== state.runToken) {
        releaseAnalysisUrls(analysis);
        return;
      }
      state.reference = analysis;

      if (analysis.faces.length > 0) {
        state.referenceFaceIndex = chooseDefaultReferenceFace(analysis.faces);
        rescoreAllCandidates();
        await processCandidateQueue();
      }
      // No-face path is handled by updateCandidateMessage() in the finally
      // block, which surfaces a "no face — queue paused" hint when needed.
    } catch (error) {
      if (token !== state.runToken) return;
      state.reference = {
        fileName: imageFiles[0].name,
        status: "error",
        faces: [],
        thumbnail: "",
        errorKey: normalizeErrorKey(error),
        errorVars: normalizeErrorVars(error)
      };
    } finally {
      if (token === state.runToken) {
        hideProgress();
        renderReference();
        renderResults();
        updateCandidateMessage();
      }
    }
  }

  function handleCandidateFiles(files) {
    const imageFiles = files.filter(isLikelyImage);
    const rejectedCount = files.length - imageFiles.length;

    if (!imageFiles.length) {
      elements.candidateMessage.className = "message error";
      elements.candidateMessage.textContent = t("message.files.none_image");
      return;
    }

    for (const file of imageFiles) {
      state.candidates.push({
        id: state.nextCandidateId++,
        file,
        fileName: file.name,
        displayName: "",
        sourceUrl: "",
        attribution: null,
        status: "queued",
        result: null,
        error: null
      });
    }

    elements.candidateMessage.className = rejectedCount ? "message warn" : "message";
    elements.candidateMessage.textContent = rejectedCount
      ? t("message.files.queued_mixed", { accepted: imageFiles.length, rejected: rejectedCount })
      : t("message.files.queued_only", { count: imageFiles.length });

    renderResults();
    processCandidateQueue();
  }

  function ensureModelsReady() {
    if (!state.modelsReady) {
      if (state.modelError) {
        throw makeI18nError("status.model.error", { message: normalizeError(state.modelError) });
      }
      throw makeI18nError("error.model_not_loaded");
    }
  }

  function getReferenceFace() {
    if (!state.reference || !state.reference.faces || !state.reference.faces.length) {
      return null;
    }
    return state.reference.faces[state.referenceFaceIndex] || state.reference.faces[0] || null;
  }

  function hasUsableReference() {
    return Boolean(getReferenceFace());
  }

  async function processCandidateQueue() {
    if (state.processing || !state.candidates.length || !hasUsableReference()) {
      updateCandidateMessage();
      scheduleRenderResults();
      return;
    }

    if (!state.modelsReady) {
      updateCandidateMessage();
      return;
    }

    const token = state.runToken;
    state.processing = true;
    const queue = state.candidates.filter((candidate) => candidate.status === "queued" || candidate.status === "error-read");
    let completed = 0;
    let cancelled = false;

    showProgress(0, Math.max(queue.length, 1), queue.length ? "progress.candidates.processing" : "progress.none_queued");

    for (const candidate of queue) {
      if (token !== state.runToken) {
        cancelled = true;
        break;
      }
      candidate.status = "processing";
      scheduleRenderResults();
      showProgress(completed, queue.length, "progress.candidate.processing", { fileName: candidate.fileName });

      try {
        await yieldToBrowser();
        const analysis = await analyzeImageFile(candidate.file);
        if (token !== state.runToken) {
          releaseAnalysisUrls(analysis);
          candidate.status = "queued";
          cancelled = true;
          break;
        }
        analysis.displayName = candidate.displayName || "";
        analysis.sourceUrl = candidate.sourceUrl || "";
        analysis.attribution = normalizeAttribution(candidate.attribution);
        candidate.result = scoreCandidateAnalysis(analysis);
        candidate.status = "done";
        candidate.error = null;
        candidate.file = null;
      } catch (error) {
        if (token !== state.runToken) {
          candidate.status = "queued";
          cancelled = true;
          break;
        }
        const errorKey = normalizeErrorKey(error);
        const errorVars = normalizeErrorVars(error);
        candidate.result = {
          fileName: candidate.fileName,
          thumbnail: "",
          width: 0,
          height: 0,
          faces: [],
          comparisons: [],
          best: null,
          statusKind: "error",
          statusKey: errorKey,
          statusVars: errorVars,
          faceCount: 0,
          errorKey,
          errorVars
        };
        candidate.status = "done";
        candidate.errorKey = errorKey;
        candidate.errorVars = errorVars;
        candidate.file = null;
      }

      completed += 1;
      showProgress(completed, queue.length, "progress.candidates.done", { completed, total: queue.length });
      scheduleRenderResults();
      await yieldToBrowser();
    }

    state.processing = false;
    if (token === state.runToken || !state.reference || state.reference.status !== "processing") {
      hideProgress();
    }

    if (cancelled || token !== state.runToken) {
      for (const candidate of state.candidates) {
        if (candidate.status === "processing") {
          candidate.status = "queued";
        }
      }
    }

    updateCandidateMessage();
    renderResults();

    if (state.candidates.some((candidate) => candidate.status === "queued")) {
      processCandidateQueue();
    }
  }

  // ---------- per-image analysis ----------

  async function analyzeImageFile(file) {
    return runAnalysisExclusive(() => analyzeImageFileNow(file));
  }

  async function analyzeImageFileNow(file) {
    const loaded = await loadImage(file);
    let analysisCanvas = null;
    let thumbnail = "";
    const faces = [];
    let returned = false;
    try {
      analysisCanvas = drawImageToCanvas(loaded.image, MAX_ANALYSIS_SIDE);
      thumbnail = await canvasToObjectUrl(analysisCanvas, THUMBNAIL_SIDE, 0.82);

      const detectorOptions = new faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: 0.4
      });

      // face-api detection + landmarks. The face-api faceRecognitionNet is no
      // longer used; ArcFace replaces it for descriptor extraction.
      // Wrap in a tensor scope so transient detection tensors are released
      // even when long batches are processed back-to-back.
      const endScope = beginTensorScope();
      let detections;
      try {
        detections = await faceapi
          .detectAllFaces(analysisCanvas, detectorOptions)
          .withFaceLandmarks();
      } finally {
        endScope();
      }

      for (let index = 0; index < detections.length; index += 1) {
        const detection = detections[index];
        const fivePoints = fivePointFrom68(detection.landmarks);
        if (!fivePoints) continue;

        const aligned = alignedFaceCanvas(analysisCanvas, fivePoints);
        let descriptor;
        let blurVariance;
        try {
          descriptor = await descriptorWithFlipTta(aligned);
          blurVariance = laplacianVarianceForCanvas(aligned);
        } finally {
          aligned.width = 0;
          aligned.height = 0;
        }

        const box = detection.detection.box;
        const { yawRatio, pitchRatio, rollDegrees } = poseRatiosFromFivePoints(fivePoints);
        const crop = await cropFaceToObjectUrl(analysisCanvas, box, FACE_CROP_SIDE);

        faces.push({
          index,
          descriptor,
          score: detection.detection.score,
          box: { x: box.x, y: box.y, width: box.width, height: box.height },
          crop,
          quality: {
            detectorScore: detection.detection.score,
            yawRatio,
            pitchRatio,
            rollDegrees,
            blurVariance
          }
        });

        await yieldToBrowser();
      }

      faces.sort((left, right) => (right.box.width * right.box.height) - (left.box.width * left.box.height));

      const analysis = {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type || "unknown",
        width: analysisCanvas.width,
        height: analysisCanvas.height,
        thumbnail,
        faces,
        faceCount: faces.length,
        status: faces.length ? "ok" : "no-face",
        errorKey: faces.length ? null : "error.no_face_detected"
      };
      returned = true;
      return analysis;
    } finally {
      if (!returned) {
        releaseObjectUrl(thumbnail);
        for (const face of faces) {
          releaseObjectUrl(face.crop);
        }
      }
      loaded.release();
      if (analysisCanvas) {
        analysisCanvas.width = 0;
        analysisCanvas.height = 0;
      }
    }
  }

  async function loadImage(file) {
    if ("createImageBitmap" in window) {
      try {
        const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
        return {
          image: bitmap,
          release() {
            if (typeof bitmap.close === "function") {
              bitmap.close();
            }
          }
        };
      } catch (_error) {
        // Fall back to HTMLImageElement decoding below for formats or
        // browsers that do not support createImageBitmap for this file.
      }
    }

    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.decoding = "async";
      image.onload = () => resolve({
        image,
        release() {
          URL.revokeObjectURL(url);
        }
      });
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(makeI18nError("error.image_unreadable"));
      };
      image.src = url;
    });
  }

  function drawImageToCanvas(image, maxSide) {
    const sourceWidth = image.naturalWidth || image.width;
    const sourceHeight = image.naturalHeight || image.height;
    if (!sourceWidth || !sourceHeight) {
      throw makeI18nError("error.image_unreadable");
    }

    const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const context = get2dContext(canvas, READBACK_CONTEXT_OPTIONS);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  async function canvasToObjectUrl(sourceCanvas, maxSide, quality) {
    const scale = Math.min(1, maxSide / Math.max(sourceCanvas.width, sourceCanvas.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sourceCanvas.width * scale));
    canvas.height = Math.max(1, Math.round(sourceCanvas.height * scale));
    const context = get2dContext(canvas, DRAW_CONTEXT_OPTIONS);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
    try {
      const blob = await canvasToBlob(canvas, "image/jpeg", quality);
      return registerObjectUrl(URL.createObjectURL(blob), blob);
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  async function cropFaceToObjectUrl(sourceCanvas, box, size) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const square = Math.max(box.width, box.height) * 1.55;
    const sourceX = clamp(centerX - square / 2, 0, sourceCanvas.width);
    const sourceY = clamp(centerY - square / 2, 0, sourceCanvas.height);
    const sourceRight = clamp(centerX + square / 2, 0, sourceCanvas.width);
    const sourceBottom = clamp(centerY + square / 2, 0, sourceCanvas.height);
    const sourceWidth = Math.max(1, sourceRight - sourceX);
    const sourceHeight = Math.max(1, sourceBottom - sourceY);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = get2dContext(canvas, DRAW_CONTEXT_OPTIONS);
    context.fillStyle = "#eef2f7";
    context.fillRect(0, 0, size, size);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(sourceCanvas, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, size, size);
    try {
      const blob = await canvasToBlob(canvas, "image/jpeg", 0.86);
      return registerObjectUrl(URL.createObjectURL(blob), blob);
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  function canvasToBlob(canvas, type, quality) {
    if (typeof canvas.toBlob === "function") {
      return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(makeI18nError("error.preview_generation"));
        }, type, quality);
      });
    }

    // Very old browsers may lack toBlob. Keep a fallback so previews still
    // work, but most modern browsers take the asynchronous path above.
    try {
      const dataUrl = canvas.toDataURL(type, quality);
      const [header, payload] = dataUrl.split(",");
      const mimeMatch = /^data:([^;]+);base64$/i.exec(header || "");
      const bytes = base64ToUint8Array(payload || "");
      return Promise.resolve(new Blob([bytes], { type: mimeMatch ? mimeMatch[1] : type }));
    } catch (error) {
      return Promise.reject(error);
    }
  }

  async function objectUrlToAsset(url) {
    if (!url) return null;
    if (!window.crypto || !window.crypto.subtle) {
      throw makeI18nError("error.crypto_unavailable");
    }
    const blob = objectUrlBlobs.get(url);
    if (!blob) {
      throw makeI18nError("error.searchset_asset_unavailable");
    }
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return {
      mediaType: blob.type || "application/octet-stream",
      encoding: "base64",
      byteLength: bytes.length,
      sha256: await sha256Base64Url(bytes),
      data: bytesToBase64(bytes)
    };
  }

  function assetToObjectUrl(asset) {
    if (!asset) return "";
    if (asset.encoding !== "base64" || typeof asset.data !== "string") {
      throw makeI18nError("error.searchset_invalid");
    }
    const bytes = base64ToBytes(asset.data);
    const blob = new Blob([bytes], { type: asset.mediaType || "application/octet-stream" });
    return registerObjectUrl(URL.createObjectURL(blob), blob);
  }

  function encodeDescriptor(descriptor) {
    if (!descriptor || descriptor.length !== ARCFACE_EMBEDDING_DIM) {
      throw makeI18nError("error.searchset_invalid");
    }
    const buffer = new ArrayBuffer(ARCFACE_EMBEDDING_DIM * 4);
    const view = new DataView(buffer);
    for (let index = 0; index < ARCFACE_EMBEDDING_DIM; index += 1) {
      view.setFloat32(index * 4, Number(descriptor[index]) || 0, true);
    }
    return {
      dimensions: ARCFACE_EMBEDDING_DIM,
      encoding: "float32-le-base64",
      data: bytesToBase64(new Uint8Array(buffer))
    };
  }

  function decodeDescriptor(encoded) {
    if (!encoded || encoded.encoding !== "float32-le-base64" || encoded.dimensions !== ARCFACE_EMBEDDING_DIM) {
      throw makeI18nError("error.searchset_incompatible");
    }
    const bytes = base64ToBytes(encoded.data);
    if (bytes.byteLength !== ARCFACE_EMBEDDING_DIM * 4) {
      throw makeI18nError("error.searchset_incompatible");
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const descriptor = new Float32Array(ARCFACE_EMBEDDING_DIM);
    for (let index = 0; index < ARCFACE_EMBEDDING_DIM; index += 1) {
      descriptor[index] = view.getFloat32(index * 4, true);
    }
    return descriptor;
  }

  async function compressPayload(bytes) {
    if (typeof CompressionStream !== "function") {
      return { compression: "none", bytes };
    }
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
    return {
      compression: "gzip",
      bytes: new Uint8Array(await new Response(stream).arrayBuffer())
    };
  }

  async function decompressPayload(bytes, compression) {
    if (!compression || compression === "none") {
      return bytes;
    }
    if (compression !== "gzip" || typeof DecompressionStream !== "function") {
      throw makeI18nError("error.searchset_unsupported_compression");
    }
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  function downloadBlob(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Keep the URL alive long enough that even slow browsers / "Save As"
    // dialogs / sandboxed file pickers can resolve the download before the
    // blob is revoked. 250ms used to truncate downloads on slow devices.
    window.setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_TTL_MS);
  }

  function chooseDefaultReferenceFace(faces) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    faces.forEach((face, index) => {
      const area = face.box.width * face.box.height;
      const score = area * Math.max(face.score, 0.01);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });
    return bestIndex;
  }

  function rescoreAllCandidates() {
    for (const candidate of state.candidates) {
      if (candidate.result && candidate.result.faces) {
        candidate.result = scoreCandidateAnalysis(candidate.result);
      }
    }
  }

  // ---------- comparison ----------

  function dotProduct(left, right) {
    if (!left || !right || left.length !== right.length) {
      return Number.NaN;
    }
    let sum = 0;
    for (let i = 0; i < left.length; i += 1) {
      sum += left[i] * right[i];
    }
    return sum;
  }

  function euclideanDistance(left, right) {
    if (!left || !right || left.length !== right.length) {
      return Number.POSITIVE_INFINITY;
    }
    let sum = 0;
    for (let i = 0; i < left.length; i += 1) {
      const delta = left[i] - right[i];
      sum += delta * delta;
    }
    return Math.sqrt(sum);
  }

  function cosineToPercent(cosine) {
    if (!Number.isFinite(cosine)) return 0;
    // Calibrated sigmoid: 50% maps to COSINE_PERCENT_CENTER, slope tuned for
    // SE-MobileFaceNet ArcFace cosine distributions on MS1M-trained weights.
    const z = COSINE_PERCENT_SLOPE * (cosine - COSINE_PERCENT_CENTER);
    const pct = 100 / (1 + Math.exp(-z));
    return Math.round(clamp(pct, 0, 100));
  }

  function interpretSimilarity(percent) {
    if (percent >= 85) return "similarity.very_high";
    if (percent >= 70) return "similarity.high";
    if (percent >= 50) return "similarity.medium";
    if (percent >= 30) return "similarity.low";
    return "similarity.very_low";
  }

  function scoreClass(percent) {
    if (!Number.isFinite(percent)) return "unavailable";
    if (percent >= 85) return "strong";
    if (percent >= 70) return "";
    if (percent >= 50) return "maybe";
    return "low";
  }

  function scoreCandidateAnalysis(analysis) {
    const referenceFace = getReferenceFace();
    const base = {
      ...analysis,
      comparisons: [],
      best: null,
      statusKind: "error",
      statusKey: "",
      statusVars: {},
      errorKey: analysis.errorKey || null,
      errorVars: analysis.errorVars || {}
    };

    if (!referenceFace) {
      base.statusKind = "error";
      base.statusKey = "error.model_or_reference_missing";
      return base;
    }

    if (!analysis.faces || analysis.faces.length === 0) {
      base.statusKind = "error";
      base.statusKey = "error.no_face_detected";
      base.errorKey = "error.no_face_detected";
      return base;
    }

    base.comparisons = analysis.faces.map((face, index) => {
      const cosine = dotProduct(referenceFace.descriptor, face.descriptor);
      const distance = euclideanDistance(referenceFace.descriptor, face.descriptor);
      const similarity = cosineToPercent(cosine);
      return {
        faceIndex: index,
        originalFaceIndex: face.index,
        cosine,
        distance,
        similarity,
        interpretationKey: interpretSimilarity(similarity),
        detectorScore: face.score,
        crop: face.crop,
        qualityIssues: classifyQuality(face.quality),
        quality: face.quality
      };
    });

    base.best = base.comparisons.reduce((best, item) => {
      if (!best || item.similarity > best.similarity) return item;
      if (item.similarity === best.similarity && item.cosine > best.cosine) return item;
      return best;
    }, null);

    if (analysis.faces.length > 1) {
      base.statusKind = "warn";
      base.statusKey = "candidate.state.multiple_faces";
      base.statusVars = { count: analysis.faces.length };
    } else {
      base.statusKind = "ok";
      base.statusKey = "candidate.state.face_detected";
    }

    return base;
  }

  function normalizeErrorKey(error) {
    if (!error) return "error.image_unreadable";
    if (error.i18nKey) return error.i18nKey;
    const message = error.message || String(error);
    if (state.locales.en && state.locales.en[message]) return message;
    if (/could not be read|decode|image/i.test(message)) return "error.image_unreadable";
    return "error.unexpected";
  }

  function normalizeErrorVars(error) {
    if (!error) return {};
    if (error.i18nVars) return error.i18nVars;
    const message = error.message || String(error);
    return message ? { message } : {};
  }

  function normalizeError(error) {
    return t(normalizeErrorKey(error), normalizeErrorVars(error));
  }

  // ---------- progress UI ----------

  function showProgress(value, max, key, vars) {
    state.progressKey = key || "progress.idle";
    state.progressVars = vars || {};
    elements.progressWrap.classList.add("active");
    elements.progressBar.max = Math.max(1, max || 1);
    elements.progressBar.value = clamp(value || 0, 0, elements.progressBar.max);
    refreshProgressText();
  }

  function hideProgress() {
    state.progressKey = "progress.idle";
    state.progressVars = {};
    elements.progressWrap.classList.remove("active");
    elements.progressBar.value = 0;
    refreshProgressText();
  }

  // ---------- rendering ----------

  function scheduleRenderResults() {
    if (renderResultsScheduled) return;
    renderResultsScheduled = true;
    const callback = () => {
      renderResultsScheduled = false;
      renderResults();
    };
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(callback);
    } else {
      window.setTimeout(callback, 16);
    }
  }

  function renderReference() {
    const reference = state.reference;
    elements.referenceFaces.innerHTML = "";

    if (!reference) {
      elements.referencePreview.innerHTML = `<div class="empty-state">${escapeHtml(t("reference.empty"))}</div>`;
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = t("reference.help");
      return;
    }

    if (reference.thumbnail) {
      elements.referencePreview.innerHTML = `<img src="${reference.thumbnail}" alt="${escapeHtml(t("alt.reference_image_preview"))}">`;
    } else {
      elements.referencePreview.innerHTML = `<div class="empty-state">${escapeHtml(t("progress.candidate.processing", { fileName: reference.fileName }))}</div>`;
    }

    if (reference.status === "processing") {
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = t("message.reference.processing");
      return;
    }

    if (reference.errorKey || !reference.faces.length) {
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = t(reference.errorKey || "error.no_face_detected", reference.errorVars || {});
      return;
    }

    if (reference.faces.length > 1) {
      elements.referenceMessage.className = "message warn";
      elements.referenceMessage.textContent = t("message.reference.multiple", { count: reference.faces.length });
    } else {
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = t("message.reference.ready");
    }

    const label = document.createElement("div");
    label.className = "summary-counts";
    label.textContent = t("message.reference.faces_label");
    const strip = document.createElement("div");
    strip.className = "face-strip";

    reference.faces.forEach((face, index) => {
      const issues = classifyQuality(face.quality);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `face-choice ${index === state.referenceFaceIndex ? "active" : ""}`;
      const issueBadge = issues.length ? `<span class="face-quality-warn">${escapeHtml(issueLabel(issues[0]))}</span>` : "";
      button.innerHTML = `
        <img src="${face.crop}" alt="${escapeHtml(t("label.reference_face_with_index", { index: index + 1 }))}">
        <span>${escapeHtml(t("label.face_with_index", { index: index + 1 }))}</span>
        ${issueBadge}
      `;
      button.addEventListener("click", () => {
        state.referenceFaceIndex = index;
        rescoreAllCandidates();
        renderReference();
        renderResults();
      });
      strip.appendChild(button);
    });

    elements.referenceFaces.append(label, strip);
  }

  function renderResults() {
    elements.exportButton.disabled = !state.candidates.some((candidate) => candidate.result);
    elements.exportSetButton.disabled = !state.candidates.some((candidate) => candidate.result && candidate.result.faces && candidate.result.faces.length);

    if (!state.candidates.length) {
      elements.summaryCounts.textContent = t("results.empty");
      elements.resultsList.innerHTML = `<div class="empty-state">${escapeHtml(t("results.list.empty"))}</div>`;
      return;
    }

    const done = state.candidates.filter((candidate) => candidate.result).length;
    const withScores = state.candidates.filter((candidate) => candidate.result && candidate.result.best).length;
    const errors = state.candidates.filter((candidate) => candidate.result && candidate.result.statusKind === "error").length;
    elements.summaryCounts.textContent = t("summary.counts", {
      done,
      total: state.candidates.length,
      scored: withScores,
      errors
    });

    const sorted = getSortedCandidates();
    elements.resultsList.innerHTML = sorted.map(renderCandidateCard).join("");
  }

  function getSortedCandidates() {
    const candidates = [...state.candidates];
    const similarity = (candidate) => candidate.result && candidate.result.best ? candidate.result.best.similarity : -1;
    const name = (candidate) => candidate.fileName.toLocaleLowerCase();
    const status = (candidate) => {
      if (!candidate.result) return candidate.status;
      return t(candidate.result.statusKey || "candidate.state.pending", candidate.result.statusVars || {});
    };

    candidates.sort((left, right) => {
      if (state.sortMode === "similarity-asc") {
        return similarity(left) - similarity(right) || name(left).localeCompare(name(right));
      }
      if (state.sortMode === "name-asc") {
        return name(left).localeCompare(name(right));
      }
      if (state.sortMode === "status") {
        return status(left).localeCompare(status(right)) || similarity(right) - similarity(left);
      }
      return similarity(right) - similarity(left) || name(left).localeCompare(name(right));
    });
    return candidates;
  }

  function renderCandidateCard(candidate) {
    if (!candidate.result) {
      const status = candidate.status === "processing"
        ? t("candidate.state.processing")
        : hasUsableReference()
          ? t("candidate.state.queued")
          : t("candidate.state.waiting_reference");
      return `
        <article class="result-card">
          <div class="thumb"><div class="empty-state">${candidate.status === "processing" ? "..." : ""}</div></div>
          <div class="face-thumb"><div class="empty-state">${escapeHtml(t("label.face"))}</div></div>
          <div class="result-main">
            <div class="filename" title="${escapeHtml(candidate.fileName)}">${escapeHtml(candidate.fileName)}</div>
            <label class="name-field">
              <span class="sr-only">${escapeHtml(t("label.display_name"))}</span>
              <input class="name-input" type="text" value="${escapeHtml(candidate.displayName || "")}" placeholder="${escapeHtml(t("placeholder.display_name"))}" data-candidate-display-name="${candidate.id}">
            </label>
            <div class="status-row"><span class="badge">${status}</span></div>
          </div>
          <div class="score-block">
            <div class="score unavailable">${escapeHtml(t("candidate.state.pending"))}</div>
            <div class="interpretation">${escapeHtml(state.modelsReady ? t("candidate.state.awaiting_processing") : t("candidate.state.model_not_loaded"))}</div>
          </div>
        </article>
      `;
    }

    const result = candidate.result;
    const best = result.best;
    const percent = best ? best.similarity : null;
    const interpretation = best ? t(best.interpretationKey) : t(result.statusKey || "candidate.state.pending", result.statusVars || {});
    const statusKind = result.statusKind || "error";
    const bestFace = best ? result.faces[best.faceIndex] : null;
    const scoreText = best ? `${percent}%` : t("candidate.state.no_score");
    const scoreLabel = best ? t("detail.similarity_value", { percent, interpretation: t(best.interpretationKey) }) : t("candidate.state.no_similarity_score");
    const thumb = result.thumbnail
      ? `<img src="${result.thumbnail}" alt="${escapeHtml(t("alt.candidate_image_preview"))}">`
      : `<div class="empty-state">${escapeHtml(t("candidate.state.no_preview"))}</div>`;
    const faceThumb = bestFace && bestFace.crop
      ? `<img src="${bestFace.crop}" alt="${escapeHtml(t("alt.best_detected_face_preview"))}">`
      : `<div class="empty-state">${escapeHtml(t("candidate.state.no_face"))}</div>`;

    const qualityIssues = best && best.qualityIssues && best.qualityIssues.length
      ? `<span class="badge warn">${escapeHtml(t("label.quality", { issues: qualityListLabel(best.qualityIssues) }))}</span>`
      : "";
    const attribution = normalizeAttribution(result.attribution || candidate.attribution);
    const attributionButton = attribution
      ? `<button class="info-button" type="button" data-candidate-attribution="${candidate.id}" title="${escapeHtml(t("button.attribution_info"))}" aria-label="${escapeHtml(t("button.attribution_info"))}">${escapeHtml(t("button.info"))}</button>`
      : "";

    return `
      <article class="result-card">
        <div>
          <div class="thumb">${thumb}</div>
          <div class="thumb-label">${escapeHtml(t("label.image"))}</div>
        </div>
        <div>
          <div class="face-thumb">${faceThumb}</div>
          <div class="face-label">${escapeHtml(best ? t("label.best_face") : t("label.face"))}</div>
        </div>
        <div class="result-main">
          <div class="filename" title="${escapeHtml(candidate.fileName)}">${escapeHtml(candidate.fileName)}</div>
          <label class="name-field">
            <span class="sr-only">${escapeHtml(t("label.display_name"))}</span>
            <input class="name-input" type="text" value="${escapeHtml(candidate.displayName || result.displayName || "")}" placeholder="${escapeHtml(t("placeholder.display_name"))}" data-candidate-display-name="${candidate.id}">
          </label>
          <div class="status-row">
            <span class="badge ${statusKind}">${escapeHtml(t(result.statusKey || "candidate.state.pending", result.statusVars || {}))}</span>
            <span class="badge">${escapeHtml(t("label.face_count", { count: result.faceCount || 0 }))}</span>
            ${qualityIssues}
            ${attributionButton}
          </div>
        </div>
        <div class="score-block">
          <div class="score ${scoreClass(percent)}" aria-label="${scoreLabel}">${escapeHtml(scoreText)}</div>
          <div class="interpretation">${escapeHtml(interpretation)}</div>
        </div>
        ${renderDetails(result)}
      </article>
    `;
  }

  function renderDetails(result) {
    const best = result.best;
    const attribution = normalizeAttribution(result.attribution);
    const cells = [
      [t("detail.filename"), result.fileName],
      [t("detail.display_name"), result.displayName || ""],
      [t("detail.source_url"), result.sourceUrl || ""],
      [t("detail.detection_status"), t(result.statusKey || "candidate.state.pending", result.statusVars || {})],
      [t("detail.detected_faces"), String(result.faceCount || 0)],
      [t("detail.image_size"), result.width && result.height ? `${result.width} x ${result.height}` : ""]
    ];

    if (attribution) {
      for (const cell of [
        [t("detail.author"), attribution.author],
        [t("detail.license"), attribution.license],
        [t("detail.source_link"), attribution.sourceLink],
        [t("detail.attribution_notes"), attribution.notes]
      ]) {
        if (cell[1]) cells.push(cell);
      }
    }

    if (best) {
      cells.push(
        [t("detail.similarity"), t("detail.similarity_value", { percent: best.similarity, interpretation: t(best.interpretationKey) })],
        [t("detail.cosine"), formatNumber(best.cosine, 5)],
        [t("detail.euclidean"), formatNumber(best.distance, 5)],
        [t("detail.detector_score"), formatNumber(best.detectorScore, 4)]
      );
      if (best.quality) {
        cells.push(
          [t("detail.yaw_ratio"), formatNumber(best.quality.yawRatio, 3)],
          [t("detail.pitch_ratio"), formatNumber(best.quality.pitchRatio, 3)],
          [t("detail.roll"), formatNumber(best.quality.rollDegrees, 2)],
          [t("detail.sharpness"), formatNumber(best.quality.blurVariance, 1)]
        );
      }
    } else if (result.errorKey) {
      cells.push([t("detail.error"), t(result.errorKey, result.errorVars || {})]);
    }

    const faceList = result.comparisons && result.comparisons.length
      ? `
        <div class="candidate-faces">
          ${result.comparisons.map((item) => `
            <div class="candidate-face ${best && item.faceIndex === best.faceIndex ? "best" : ""}">
              ${item.crop
                ? `<img src="${item.crop}" alt="${escapeHtml(t("label.detected_candidate_face_with_index", { index: item.faceIndex + 1 }))}">`
                : `<div class="empty-state">${escapeHtml(t("candidate.state.no_preview"))}</div>`}
              <strong>${item.similarity}%</strong>
              <span>${escapeHtml(t("label.face_with_index", { index: item.faceIndex + 1 }))}</span>
              <span>${escapeHtml(t("detail.cosine_short", { value: formatNumber(item.cosine, 3) }))}</span>
            </div>
          `).join("")}
        </div>
      `
      : "";

    return `
      <details class="details">
        <summary>${escapeHtml(t("detail.summary"))}</summary>
        <div class="detail-grid">
          ${cells.map(([label, value]) => `
            <div class="detail-cell">
              <b>${escapeHtml(label)}</b>
              <span>${escapeHtml(value)}</span>
            </div>
          `).join("")}
        </div>
        ${faceList}
      </details>
    `;
  }

  function openAttributionDialog(candidateId) {
    const candidate = state.candidates.find((item) => String(item.id) === String(candidateId));
    if (!candidate || !candidate.result) return;
    const attribution = normalizeAttribution(candidate.result.attribution || candidate.attribution);
    if (!attribution) return;

    const sourceHref = safeExternalHref(attribution.sourceLink);
    const rows = [
      { label: t("detail.filename"), value: candidate.fileName },
      { label: t("detail.display_name"), value: candidate.displayName || candidate.result.displayName || "" },
      { label: t("detail.author"), value: attribution.author },
      { label: t("detail.license"), value: attribution.license },
      { label: t("detail.source_link"), value: attribution.sourceLink, href: sourceHref },
      { label: t("detail.attribution_notes"), value: attribution.notes }
    ].filter((row) => row.value);

    elements.attributionDialogBody.innerHTML = `
      <div class="attribution-grid">
        ${rows.map((row) => `
          <div class="attribution-row">
            <b>${escapeHtml(row.label)}</b>
            <span>${row.href
              ? `<a href="${escapeHtml(row.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(row.value)}</a>`
              : escapeHtml(row.value)}
            </span>
          </div>
        `).join("")}
      </div>
    `;

    const opener = document.activeElement;
    elements.attributionDialog.addEventListener("close", () => {
      if (opener && typeof opener.focus === "function") {
        try { opener.focus(); } catch (_) {}
      }
    }, { once: true });

    if (typeof elements.attributionDialog.showModal === "function") {
      elements.attributionDialog.showModal();
    } else {
      elements.attributionDialog.setAttribute("open", "");
    }
  }

  function updateCandidateMessage() {
    if (!state.modelsReady) {
      elements.candidateMessage.className = state.modelError ? "message error" : "message";
      elements.candidateMessage.textContent = state.modelError
        ? t("message.candidates.model_unavailable")
        : t("message.candidates.loading_models");
      return;
    }

    if (!hasUsableReference()) {
      const reference = state.reference;
      const referenceLoadedButNoFace = reference
        && reference.status !== "processing"
        && (!reference.faces || reference.faces.length === 0);
      const waitingCount = state.candidates.filter((candidate) => {
        if (candidate.status === "queued") return true;
        return candidate.result
          && !candidate.result.best
          && candidate.result.statusKey === "candidate.state.waiting_reference";
      }).length;

      if (referenceLoadedButNoFace && waitingCount > 0) {
        elements.candidateMessage.className = "message warn";
        elements.candidateMessage.textContent = t("message.reference.no_face_queue_paused", { count: waitingCount });
        return;
      }

      elements.candidateMessage.className = "message";
      elements.candidateMessage.textContent = state.candidates.length
        ? t("message.candidates.queue_wait_reference", { count: state.candidates.length })
        : t("candidate.help.initial");
      return;
    }

    const queued = state.candidates.filter((candidate) => candidate.status === "queued").length;
    const processed = state.candidates.filter((candidate) => candidate.result).length;
    elements.candidateMessage.className = "message";
    elements.candidateMessage.textContent = queued
      ? t("message.files.queued_only", { count: queued })
      : t("message.candidates.processed", { count: processed });
  }

  // ---------- search set import/export ----------

  function searchSetPipelineMetadata() {
    return {
      id: "facetrace-arcface-256-v1",
      embeddingDim: ARCFACE_EMBEDDING_DIM,
      descriptorEncoding: "float32-le-base64",
      detector: "face-api tiny_face_detector_model",
      landmarks: "face-api face_landmark_68_model",
      recognizer: "SE-MobileFaceNet ArcFace TF.js GraphModel",
      alignment: "insightface-5point-112",
      tta: "horizontal-flip-average-l2",
      maxAnalysisSide: MAX_ANALYSIS_SIDE,
      thumbnailSide: THUMBNAIL_SIDE,
      faceCropSide: FACE_CROP_SIDE,
      similarity: {
        metric: "cosine",
        percentCenter: COSINE_PERCENT_CENTER,
        percentSlope: COSINE_PERCENT_SLOPE
      }
    };
  }

  async function buildSearchSetPayload(candidates, onProgress) {
    const items = [];
    let faceCount = 0;

    for (let i = 0; i < candidates.length; i += 1) {
      const candidate = candidates[i];
      const result = candidate.result;
      const attribution = normalizeAttribution(result.attribution || candidate.attribution);
      const faces = [];
      for (const face of result.faces || []) {
        faces.push({
          index: face.index,
          descriptor: encodeDescriptor(face.descriptor),
          detectorScore: face.score,
          box: face.box,
          quality: face.quality,
          crop: await objectUrlToAsset(face.crop)
        });
      }
      faceCount += faces.length;
      items.push({
        id: `item-${items.length + 1}`,
        fileName: candidate.fileNameIsPlaceholder ? "" : (candidate.fileName || result.fileName || ""),
        displayName: candidate.displayName || result.displayName || "",
        sourceUrl: result.sourceUrl || candidate.sourceUrl || "",
        attribution: attribution || undefined,
        fileSize: result.fileSize || null,
        fileType: result.fileType || "unknown",
        width: result.width || 0,
        height: result.height || 0,
        status: faces.length ? "ok" : "no-face",
        thumbnail: await objectUrlToAsset(result.thumbnail),
        faces
      });
      if (typeof onProgress === "function") {
        onProgress(i + 1, candidates.length);
      }
      // Yield between items so the progress bar paints and the user can still
      // interact with the page on a large export.
      await yieldToBrowser();
    }

    return {
      format: SEARCH_SET_FORMAT,
      schemaVersion: SEARCH_SET_SCHEMA_VERSION,
      createdAt: new Date().toISOString(),
      producer: {
        name: "FaceTrace Offline",
        runtime: "browser"
      },
      model: searchSetPipelineMetadata(),
      counts: {
        items: items.length,
        faces: faceCount
      },
      privacy: {
        containsBiometricDescriptors: true,
        containsFullOriginalImages: false,
        containsPreviewImages: true
      },
      items
    };
  }

  async function encryptedEnvelopeForPayload(payload) {
    if (!window.crypto || !window.crypto.subtle || typeof window.crypto.getRandomValues !== "function") {
      throw makeI18nError("error.crypto_unavailable");
    }

    const payloadBytes = textToBytes(JSON.stringify(payload));
    const compressed = await compressPayload(payloadBytes);
    const keyBytes = window.crypto.getRandomValues(new Uint8Array(AES_GCM_KEY_BYTES));
    const iv = window.crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));
    const key = await window.crypto.subtle.importKey("raw", keyBytes, { name: "AES-GCM" }, false, ["encrypt"]);
    const header = {
      format: SEARCH_SET_ENVELOPE_FORMAT,
      envelopeVersion: SEARCH_SET_ENVELOPE_VERSION,
      createdAt: new Date().toISOString(),
      payload: {
        format: SEARCH_SET_FORMAT,
        schemaVersion: SEARCH_SET_SCHEMA_VERSION,
        compression: compressed.compression,
        byteLength: compressed.bytes.length
      },
      encryption: {
        algorithm: "AES-256-GCM",
        keyFormat: "ftsk1",
        iv: bytesToBase64Url(iv),
        tagLength: AES_GCM_TAG_BITS
      }
    };
    const additionalData = textToBytes(stableStringify(header));
    const ciphertext = new Uint8Array(await window.crypto.subtle.encrypt({
      name: "AES-GCM",
      iv,
      additionalData,
      tagLength: AES_GCM_TAG_BITS
    }, key, compressed.bytes));

    return {
      shareKey: `${SEARCH_SET_KEY_PREFIX}${bytesToBase64Url(keyBytes)}`,
      envelope: {
        format: SEARCH_SET_ENVELOPE_FORMAT,
        envelopeVersion: SEARCH_SET_ENVELOPE_VERSION,
        header,
        ciphertext: bytesToBase64Url(ciphertext)
      }
    };
  }

  async function exportSearchSet() {
    const exportable = state.candidates.filter((candidate) => candidate.result && candidate.result.faces && candidate.result.faces.length);
    if (!exportable.length) {
      elements.candidateMessage.className = "message warn";
      elements.candidateMessage.textContent = t("message.searchset.none_exportable");
      return;
    }

    const itemCount = exportable.length;
    const faceCount = exportable.reduce((sum, candidate) => sum + (candidate.result.faces ? candidate.result.faces.length : 0), 0);

    const decision = await showConfirmDialog({
      title: t("dialog.export.title"),
      body: t("dialog.export.body", { items: itemCount, faces: faceCount }),
      buttons: [
        { label: t("dialog.export.cancel"), value: null, kind: "cancel" },
        { label: t("dialog.export.confirm"), value: "go", kind: "primary" }
      ]
    });
    if (decision !== "go") {
      elements.candidateMessage.className = "message";
      elements.candidateMessage.textContent = t("message.searchset.export_canceled");
      return;
    }

    elements.exportSetButton.disabled = true;
    showProgress(0, exportable.length, "progress.searchset.exporting");
    try {
      const payload = await buildSearchSetPayload(exportable, (done, total) => {
        showProgress(done, total, "progress.candidates.done", { completed: done, total });
      });
      showProgress(exportable.length, exportable.length, "progress.searchset.exporting");
      const encrypted = await encryptedEnvelopeForPayload(payload);
      hideProgress();

      const acknowledged = await showShareKeyDialog(encrypted.shareKey);
      if (!acknowledged) {
        elements.candidateMessage.className = "message warn";
        elements.candidateMessage.textContent = t("message.searchset.export_canceled");
        return;
      }

      const fileBase = safeFileName(payload.items[0] && (payload.items[0].displayName || payload.items[0].fileName), "facetrace-search-set");
      downloadBlob(new Blob([JSON.stringify(encrypted.envelope, null, 2)], {
        type: "application/vnd.facetrace.search-set+json"
      }), `${fileBase}-${exportTimestampSlug()}.facetrace-set`);

      elements.candidateMessage.className = "message";
      elements.candidateMessage.textContent = t("message.searchset.exported", {
        items: payload.counts.items,
        faces: payload.counts.faces
      });
    } catch (error) {
      elements.candidateMessage.className = "message error";
      elements.candidateMessage.textContent = normalizeError(error);
    } finally {
      hideProgress();
      renderResults();
    }
  }

  async function handleSearchSetFiles(files) {
    if (!files.length) return;

    if (files.length > 1) {
      elements.candidateMessage.className = "message warn";
      elements.candidateMessage.textContent = t("error.searchset_multi_file");
      return;
    }

    const file = files[0];
    if (file.size > MAX_SEARCH_SET_FILE_BYTES) {
      elements.candidateMessage.className = "message error";
      elements.candidateMessage.textContent = t("error.searchset_too_large", {
        maxMb: Math.round(MAX_SEARCH_SET_FILE_BYTES / (1024 * 1024))
      });
      return;
    }

    state.runToken += 1;
    state.processing = false;
    showProgress(0, 1, "progress.searchset.importing");

    let payload;
    try {
      payload = await readSearchSetFile(file);
    } catch (error) {
      const errorKey = normalizeErrorKey(error);
      elements.candidateMessage.className = errorKey === "message.searchset.import_canceled" ? "message" : "message error";
      elements.candidateMessage.textContent = t(errorKey, normalizeErrorVars(error));
      hideProgress();
      return;
    }

    let mode = "replace";
    const incomingItems = (payload.items && payload.items.length) || 0;
    const currentItems = state.candidates.length;
    if (currentItems > 0) {
      const choice = await showConfirmDialog({
        title: t("dialog.import.title"),
        body: t("dialog.import.body", { currentItems, newItems: incomingItems }),
        buttons: [
          { label: t("dialog.import.cancel"), value: null, kind: "cancel" },
          { label: t("dialog.import.merge"), value: "merge" },
          { label: t("dialog.import.replace"), value: "replace", kind: "primary" }
        ]
      });
      if (!choice) {
        elements.candidateMessage.className = "message";
        elements.candidateMessage.textContent = t("message.searchset.import_canceled");
        hideProgress();
        return;
      }
      mode = choice;
    }

    try {
      const imported = await candidatesFromSearchSet(payload);

      if (mode === "replace") {
        releaseCandidateObjectUrls();
        state.candidates = [];
      }
      for (const candidate of imported) {
        candidate.id = state.nextCandidateId++;
        state.candidates.push(candidate);
      }

      if (hasUsableReference()) {
        rescoreAllCandidates();
      }

      const itemsCount = payload.counts ? payload.counts.items : imported.length;
      const facesCount = payload.counts
        ? payload.counts.faces
        : imported.reduce((sum, candidate) => sum + (candidate.result && candidate.result.faces ? candidate.result.faces.length : 0), 0);

      if (!hasUsableReference()) {
        elements.candidateMessage.className = "message";
        elements.candidateMessage.textContent = t("message.searchset.import_no_reference", { items: itemsCount });
      } else {
        elements.candidateMessage.className = "message";
        elements.candidateMessage.textContent = t("message.searchset.imported", { items: itemsCount, faces: facesCount });
      }
    } catch (error) {
      elements.candidateMessage.className = "message error";
      elements.candidateMessage.textContent = normalizeError(error);
    } finally {
      hideProgress();
      renderResults();
    }
  }

  async function readSearchSetFile(file) {
    let bytes = new Uint8Array(await file.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      bytes = await decompressPayload(bytes, "gzip");
    }

    let parsed;
    try {
      const text = bytesToText(bytes);
      parsed = JSON.parse(text);
    } catch (_error) {
      throw makeI18nError("error.searchset_invalid");
    }

    if (parsed && parsed.format === SEARCH_SET_ENVELOPE_FORMAT) {
      parsed = await decryptSearchSetEnvelope(parsed);
    }
    validateSearchSetPayload(parsed);
    return parsed;
  }

  async function parseSearchSetShareKey() {
    const raw = await promptForShareKey();
    if (raw === null) {
      throw makeI18nError("message.searchset.import_canceled");
    }
    if (!String(raw).trim()) {
      throw makeI18nError("error.searchset_key_required");
    }
    const value = String(raw).trim();
    const encoded = value.startsWith(SEARCH_SET_KEY_PREFIX) ? value.slice(SEARCH_SET_KEY_PREFIX.length) : value;
    let keyBytes;
    try {
      keyBytes = base64UrlToBytes(encoded);
    } catch (_error) {
      throw makeI18nError("error.searchset_key_invalid");
    }
    if (!keyBytes || keyBytes.length !== AES_GCM_KEY_BYTES) {
      throw makeI18nError("error.searchset_key_invalid");
    }
    return keyBytes;
  }

  async function decryptSearchSetEnvelope(envelope) {
    if (!window.crypto || !window.crypto.subtle) {
      throw makeI18nError("error.crypto_unavailable");
    }
    if (!envelope.header || !envelope.header.encryption || !envelope.ciphertext || envelope.header.encryption.algorithm !== "AES-256-GCM") {
      throw makeI18nError("error.searchset_invalid");
    }

    const keyBytes = await parseSearchSetShareKey();
    const key = await window.crypto.subtle.importKey("raw", keyBytes, { name: "AES-GCM" }, false, ["decrypt"]);
    const iv = base64UrlToBytes(envelope.header.encryption.iv);
    const additionalData = textToBytes(stableStringify(envelope.header));
    let decrypted;
    try {
      decrypted = new Uint8Array(await window.crypto.subtle.decrypt({
        name: "AES-GCM",
        iv,
        additionalData,
        tagLength: envelope.header.encryption.tagLength || AES_GCM_TAG_BITS
      }, key, base64UrlToBytes(envelope.ciphertext)));
    } catch (_error) {
      throw makeI18nError("error.searchset_decrypt_failed");
    }

    const payloadBytes = await decompressPayload(decrypted, envelope.header.payload.compression);
    try {
      return JSON.parse(bytesToText(payloadBytes));
    } catch (_error) {
      throw makeI18nError("error.searchset_invalid");
    }
  }

  function validateSearchSetPayload(payload) {
    if (!payload || payload.format !== SEARCH_SET_FORMAT || payload.schemaVersion !== SEARCH_SET_SCHEMA_VERSION || !Array.isArray(payload.items)) {
      throw makeI18nError("error.searchset_invalid");
    }
    const model = payload.model || {};
    if (model.embeddingDim !== ARCFACE_EMBEDDING_DIM || model.descriptorEncoding !== "float32-le-base64") {
      throw makeI18nError("error.searchset_incompatible");
    }
  }

  async function candidatesFromSearchSet(payload) {
    const candidates = [];
    const placeholderName = t("searchset.unnamed_file");
    for (const item of payload.items) {
      const faces = [];
      const rawFaces = Array.isArray(item.faces) ? item.faces : [];
      for (let index = 0; index < rawFaces.length; index += 1) {
        const face = rawFaces[index];
        const detectorScoreRaw = Number(face.detectorScore);
        const qualityScoreRaw = face.quality ? Number(face.quality.detectorScore) : NaN;
        faces.push({
          index: Number.isFinite(face.index) ? face.index : index,
          descriptor: decodeDescriptor(face.descriptor),
          score: Number.isFinite(detectorScoreRaw)
            ? detectorScoreRaw
            : (Number.isFinite(qualityScoreRaw) ? qualityScoreRaw : 0),
          box: face.box || { x: 0, y: 0, width: 0, height: 0 },
          crop: assetToObjectUrl(face.crop),
          quality: face.quality || {}
        });
      }

      const attribution = normalizeAttribution(item.attribution);
      const fileNameIsPlaceholder = !item.fileName;
      const fileName = item.fileName || placeholderName;
      const analysis = {
        importedSetId: state.nextImportedSetId,
        fileName,
        displayName: item.displayName || "",
        sourceUrl: item.sourceUrl || "",
        attribution,
        fileSize: item.fileSize || 0,
        fileType: item.fileType || "unknown",
        width: item.width || 0,
        height: item.height || 0,
        thumbnail: assetToObjectUrl(item.thumbnail),
        faces,
        faceCount: faces.length,
        status: faces.length ? "ok" : "no-face",
        comparisons: [],
        best: null,
        statusKind: faces.length ? "" : "error",
        statusKey: faces.length ? "candidate.state.waiting_reference" : "error.no_face_detected",
        statusVars: {},
        errorKey: faces.length ? null : "error.no_face_detected",
        errorVars: {}
      };

      candidates.push({
        id: 0,
        file: null,
        fileName: analysis.fileName,
        fileNameIsPlaceholder,
        displayName: analysis.displayName,
        sourceUrl: analysis.sourceUrl,
        attribution,
        status: "done",
        result: hasUsableReference() ? scoreCandidateAnalysis(analysis) : analysis,
        error: null
      });
    }
    state.nextImportedSetId += 1;
    return candidates;
  }

  // ---------- CSV export ----------

  function exportCsv() {
    const rows = [[
      t("csv.filename"),
      t("csv.display_name"),
      t("csv.author"),
      t("csv.license"),
      t("csv.source_link"),
      t("csv.status"),
      t("csv.face_count"),
      t("csv.similarity_percent"),
      t("csv.interpretation"),
      t("csv.cosine_similarity"),
      t("csv.euclidean_distance"),
      t("csv.best_face_index"),
      t("csv.detector_score"),
      t("csv.yaw_ratio"),
      t("csv.pitch_ratio"),
      t("csv.roll_degrees"),
      t("csv.blur_variance"),
      t("csv.quality_issues"),
      t("csv.error")
    ]];

    for (const candidate of getSortedCandidates()) {
      const result = candidate.result;
      const best = result && result.best ? result.best : null;
      const quality = best ? best.quality : null;
      const attribution = normalizeAttribution((result && result.attribution) || candidate.attribution);
      rows.push([
        candidate.fileName,
        candidate.displayName || "",
        attribution ? attribution.author : "",
        attribution ? attribution.license : "",
        attribution ? attribution.sourceLink : "",
        result ? t(result.statusKey || "candidate.state.pending", result.statusVars || {}) : t("candidate.state.pending"),
        result ? String(result.faceCount || 0) : "",
        best ? String(best.similarity) : "",
        best ? t(best.interpretationKey) : "",
        best ? formatNumber(best.cosine, 6) : "",
        best ? formatNumber(best.distance, 6) : "",
        best ? String(best.faceIndex + 1) : "",
        best ? formatNumber(best.detectorScore, 6) : "",
        quality ? formatNumber(quality.yawRatio, 3) : "",
        quality ? formatNumber(quality.pitchRatio, 3) : "",
        quality ? formatNumber(quality.rollDegrees, 2) : "",
        quality ? formatNumber(quality.blurVariance, 1) : "",
        best && best.qualityIssues ? best.qualityIssues.map((key) => t(key)).join("; ") : "",
        result && result.errorKey ? t(result.errorKey, result.errorVars || {}) : ""
      ]);
    }

    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
    // U+FEFF BOM so Excel on Windows opens UTF-8 cleanly without mojibake on
    // accented attribution fields.
    const blob = new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" });
    downloadBlob(blob, `${t("csv.filename_prefix")}-${exportTimestampSlug()}.csv`);
  }

  function csvEscape(value) {
    const text = String(value == null ? "" : value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
  }

  // ---------- bootstrap ----------

  state.locales = window.FACETRACE_EMBEDDED_LOCALES || {};
  state.locale = getInitialLocale();
  if (elements.languageSelect) {
    elements.languageSelect.value = state.locale;
    elements.languageSelect.addEventListener("change", () => {
      setLocale(elements.languageSelect.value, true);
    });
  }
  applyI18nToDom();

  renderReference();
  renderResults();
  updateCandidateMessage();
  loadModels();
})();
