# FaceTrace Offline

<p align="center">
  <a href="docs/screenshots/facetrace-offline-marketing-card.png">
    <img src="docs/screenshots/facetrace-offline-marketing-card-preview.jpg" alt="FaceTrace Offline marketing card: private, powerful, 100% offline face similarity comparison." width="900">
  </a>
</p>

<p align="center">
  <a href="https://pascal-gujer.github.io/facetrace-standalone/"><strong>Open the live app</strong></a>
  ·
  <a href="https://github.com/pascal-gujer/facetrace-standalone/releases/latest">Download the latest release</a>
</p>

<p align="center">
  <sub>No install, build step, account, upload, backend, or localhost server required. The GitHub Pages copy is the same standalone browser app and still processes selected images locally on your device.</sub>
</p>

FaceTrace Offline is a standalone browser app for local face similarity comparison. It runs entirely client-side from a self-contained `index.html`; there is no server, upload, telemetry, CDN, cloud API, or remote model download.

## Example Run

<p align="center">
  <a href="docs/screenshots/facetrace-offline-example-full.png">
    <img src="docs/screenshots/facetrace-offline-example-preview.jpg" alt="FaceTrace Offline example results showing high similarity scores, a lower sunglasses score, and a masked image with no detected face." width="900">
  </a>
</p>

<p align="center">
  <sub>Example result list from the offline app. High percentages indicate visual similarity; occlusion can lower scores or prevent face detection. Click the image for the full screenshot.</sub>
</p>

## Run Offline

Fastest path: open the hosted copy at `https://pascal-gujer.github.io/facetrace-standalone/`. It runs fully client-side; selected images are not uploaded.

For a fully offline copy:

1. Download `index.html` from the latest release or clone this repository.
2. Open `index.html` directly in a modern browser.
3. Do not start a local server; no `localhost` connection is required or used.
4. Select one reference image, then select one or many candidate images.

The app is designed to work from a direct `file://` open. The ONNX Runtime Web script, the ONNX Runtime WASM sidecar/backend, and all YuNet/SFace model weights are inlined into `index.html`, so the browser does not need to load additional JavaScript, fetch model files from the local filesystem, or contact the network.

## Source Layout

`index.html` is generated. For normal maintenance, edit these inputs instead:

- `src/index.template.html`: HTML shell with inline placeholders.
- `src/styles.css`: application styles.
- `src/canvas-readback-patch.js`: early Canvas 2D readback hint patch.
- `src/app.js`: application logic.
- `src/locales/*.json`: flat UI string maps for the embedded language selector; the build checks that all locale files have matching keys.
- `vendor/onnxruntime-web.wasm.min.js`: vendored ONNX Runtime Web browser bundle.
- `models/embedded-models.js`: generated JavaScript that holds a gzip-compressed, base64-encoded JSON map of all model manifests and weight shards; do not edit it by hand.
- `docs/search-set-format.md`: portable encrypted search set format for browser exports and external generators.

The unpacked files in `models/` are kept as source/audit copies of the model assets used to generate `models/embedded-models.js`:

- `models/onnxruntime/ort-wasm-simd-threaded.mjs` (ONNX Runtime Web WASM sidecar module)
- `models/onnxruntime/ort-wasm-simd-threaded.wasm` (ONNX Runtime Web WASM backend)
- `models/yunet/face_detection_yunet_2026may.onnx` (OpenCV YuNet detector)
- `models/opencv_sface/face_recognition_sface_2021dec_int8.onnx` (OpenCV SFace recognizer)

## Build

The build uses only Python 3.9+ from the standard library. No npm install, package manager, local web server, or network access is required.

```bash
python3 tools/build.py
```

This writes two generated artifacts:

- `models/embedded-models.js`, built from the unpacked model manifests and shards in `models/`. The map is gzip-compressed before base64 encoding.
- `index.html`, built from `src/`, `vendor/onnxruntime-web.wasm.min.js`, and the generated model bundle.

To verify that both generated files match the source inputs:

```bash
python3 tools/build.py --check
```

The build script intentionally rejects generated HTML that reintroduces external-loading tags such as `<script src>`, `<link>`, or `<iframe>`.

## GitHub Automation

`.github/workflows/build-pages-release.yml` keeps the unusual offline build honest in CI:

- Pull requests and pushes verify `index.html` and `models/embedded-models.js` with `python3 tools/build.py --check`.
- The workflow runs JavaScript syntax checks on the maintained source and generated model bundle.
- Manual workflow runs can deploy the generated `index.html` and this README as the GitHub Pages site.
- Pushing a tag named `v*` creates a GitHub Release with `index.html` and `README.md` attached.
- Manual workflow runs can also create a release tag by filling `release_tag`, for example `v1.2.3`. The workflow tags the selected commit, uploads the standalone release assets, and the tag push remains compatible with the automatic release path.

Pages deployment is deliberately manual so ordinary pushes do not accidentally publish an unfinished private build. To publish it, first open the GitHub repository settings and choose `Settings > Pages > Build and deployment > Source > GitHub Actions`, then run the `Build, Pages, Release` workflow manually with `deploy_pages` enabled. The workflow does not use `actions/configure-pages` automatic enablement because that requires a token other than the default `GITHUB_TOKEN`; keeping it manual avoids storing a Pages administration token in the repository.

The Pages deployment serves the same self-contained HTML artifact. The app still performs all face processing in the browser and does not require a server at runtime. The top-corner GitHub sponsor ribbon is a plain link only; it does not load external scripts, images, fonts, or tracking pixels.

## Why This Build Is Unusual

The target environment is restricted: it may allow opening one local HTML file but reject `localhost`, local servers, remote URLs, CDNs, and even sibling `file://` script/model loads because file URLs can be treated as unique security origins. A normal web build that emits separate JavaScript chunks or model files is therefore less reliable for this deployment.

For that reason, the repository keeps maintainable source files, then compiles them into one large HTML artifact. The final `index.html` is intentionally big because it includes the app, ONNX Runtime Web, the WASM inference backend, and the model weights needed for offline face comparison.

## Browser Compatibility

Use a recent Chrome, Edge, Firefox, or Safari with JavaScript, Canvas, Blob, File API, WebAssembly with SIMD, `Response`, `atob`, and `DecompressionStream` support enabled. The app runs ONNX Runtime Web through its embedded WASM backend. Very locked-down enterprise browsers must allow JavaScript and WebAssembly execution in the opened local HTML file.

## Recognition Pipeline

Each candidate image goes through the same local pipeline:

1. **Detection** (OpenCV YuNet `face_detection_yunet_2026may.onnx`): finds face boxes and 5 landmarks.
2. **5-point alignment** (closed-form 2D similarity transform): warps the face to the canonical 112×112 OpenCV SFace template before feeding it to the recognizer.
3. **Embedding** (OpenCV SFace `face_recognition_sface_2021dec_int8.onnx`): produces a 128-value L2-normalized face descriptor.
4. **Quality signals**: estimates yaw and pitch ratios from the 5 points, and a Laplacian-variance sharpness score on the aligned crop. Marginal faces are flagged in the UI but never rejected.

## Similarity Percentage

Each detected face is compared with the selected reference descriptor using cosine similarity (since both descriptors are L2-normalized, this is a simple dot product). The user-facing percentage is a calibrated sigmoid:

```text
similarity = 100 / (1 + exp(-14 * (cosine - 0.363)))
```

The center 0.363 follows the OpenCV SFace same-identity cosine threshold used by OpenCV examples; the slope 14 spreads the curve into a UI percentage. This percentage is specific to the SFace pipeline and is not comparable with older FaceTrace ArcFace percentages.

Raw cosine similarity, raw Euclidean distance, detector score, yaw / pitch ratios, roll, and sharpness are all shown inside each result's technical details section.

Interpretation bands:

- `85-100%`: very similar
- `70-84%`: similar
- `50-69%`: possibly similar
- `30-49%`: low similarity
- Below `30%`: very low similarity

These thresholds are deliberately conservative and are not forensic proof.

## Search Sets

Candidate images can be exported as encrypted FaceTrace search sets. A search
set contains filenames, optional real names, thumbnails, face crops, quality
metrics, optional attribution metadata, and precomputed face descriptors. It
does not include the selected reference image or the full original candidate
photos. Once imported, a new reference face can be selected and compared
against the descriptors without reprocessing the original images.

### Exporting

Click **Export set** in the candidate panel. After confirming the export count,
the app encrypts the payload with browser-native AES-256-GCM and shows a
single-use share key in the form `ftsk1_<base64url>`. The key is generated
in-browser by `crypto.getRandomValues`, never persisted, and never sent
anywhere. The dialog offers:

- **Copy key** — copy via the Clipboard API.
- **Save key file** — download the share key as a small `.txt` for password
  managers, USB sticks, or printed backup.
- **I saved the key — download set** — only then is the `.facetrace-set` file
  written to disk. Cancelling at this step abandons the encrypted payload.

Store and share the key separately from the `.facetrace-set` file. There is no
recovery if the key is lost.

### Importing

Click **Import set** in the candidate panel and select a `.facetrace-set` file
(max 512 MB). For encrypted files the app prompts for the share key; paste the
`ftsk1_...` key and click **Decrypt**. If the candidate list already has items,
a follow-up dialog asks whether to **Replace** the current list or **Add** the
imported items alongside.

If no reference face is loaded, imported current-model descriptors sit ready
until a reference image is added; the app will then score them automatically.
Older `facetrace-arcface-256-v1` search sets are still accepted, but their
stored 256-value descriptors are not compared directly. Instead, FaceTrace asks
for confirmation and tries to reprocess stored thumbnails and face crops with
YuNet + SFace. Exporting the migrated set writes the current
`facetrace-yunet-sface-128-v1` format.

### Editing

Click **Edit entries** in the candidate panel to curate the current candidate
list. Editing mode exposes per-entry edit and delete controls without cluttering
the normal matching view. The display name is also editable inline on every
result card regardless of edit mode.

The edit dialog can update filenames, real names, source URLs, and attribution
fields (`author`, `license`, `sourceLink`, and `notes`). Generated recognition
data such as descriptors, thumbnails, crops, quality metrics, and similarity
scores remains read-only. Press `⌘`/`Ctrl` + `Enter` to save the dialog without
reaching for the mouse.

URL fields (source URL and source link) are normalised on save: `http(s)://`
URLs are accepted as-is, host-only entries get an implicit `https://` prefix,
and any other scheme (`javascript:`, `data:`, `ftp:`, ...) is dropped silently.

Clearing the filename field falls back to a localised placeholder
("Imported image" / "Importiertes Bild" / "Image importée"). The placeholder
re-translates automatically on language switch.

Edits live for the current browser session and any subsequent CSV or
`.facetrace-set` export. They are discarded by **Clear session**, by importing
a search set with **Replace**, or by reloading the page.

### Format

The portable on-disk format is documented in [`docs/search-set-format.md`](docs/search-set-format.md)
so external crawlers or batch generators can produce compatible sets with
per-item author, license, and source link information. Imported records with
attribution expose that source/license information through an info button in
the results UI.

## Languages

The header language picker switches the UI between English, Deutsch, and
Français. The selection persists in `localStorage`. A `?lang=de` or `?lang=fr`
query parameter overrides both the saved preference and the browser's
`navigator.language`, which is useful when sharing pre-localized links into the
hosted Pages build.

## Privacy

All processing happens locally in your browser. No data leaves your device. The app does not upload files, call external URLs, include telemetry, or require browser permissions beyond selecting local files.

## Limitations

- Results are probabilistic and must not be used for legal, forensic, employment, access-control, or identity-verification decisions.
- All face recognition models, including OpenCV SFace, can show demographic bias on cross-ethnicity and cross-age comparisons.
- Lighting, pose, blur, age differences, occlusion, image compression, and low resolution can change scores.
- The app compares visible detected faces only. If a face is not detected, no descriptor can be computed.
- If a candidate image contains multiple faces, all detected faces are scored and the closest face is used for the main result.
- If the reference image contains multiple faces, the app defaults to the largest/highest-confidence face and lets you choose a different detected reference face.

## Included Models And Library

- `index.html`: self-contained runtime application with embedded ONNX Runtime Web, embedded WASM sidecar/backend, and embedded model data.
- `vendor/onnxruntime-web.wasm.min.js`: ONNX Runtime Web 1.26.0 browser bundle, MIT-licensed. License copy: `vendor/onnxruntime-web.LICENSE`.
- `models/onnxruntime/ort-wasm-simd-threaded.mjs`: ONNX Runtime Web WASM sidecar module.
- `models/onnxruntime/ort-wasm-simd-threaded.wasm`: ONNX Runtime Web WASM inference backend.
- `models/yunet/face_detection_yunet_2026may.onnx`: OpenCV YuNet face detector from OpenCV Zoo. License copy: `models/yunet/MODEL_LICENSE.txt`.
- `models/opencv_sface/face_recognition_sface_2021dec_int8.onnx`: OpenCV SFace recognizer from OpenCV Zoo. License copy: `models/opencv_sface/MODEL_LICENSE.txt`.
- `models/embedded-models.js`: generated, gzip-compressed local bundle of the same runtime and model assets, also embedded into `index.html`.

Review ONNX Runtime Web, OpenCV Zoo model licenses, and the source dataset terms for any redistribution outside your own use case.

## Troubleshooting

- **Model not loaded**: confirm you are opening the generated self-contained `index.html`, not an older copy that still referenced external local scripts.
- **Canvas readback warning**: current `index.html` applies the Canvas 2D `willReadFrequently` hint before image analysis runs. If an old browser still logs this warning, it is a performance hint rather than a correctness failure.
- **Image could not be read**: try a common browser-readable format such as JPEG, PNG, WebP, AVIF, or BMP.
- **No face detected**: use a clearer image with a larger, front-facing face and fewer occlusions.
- **WebAssembly disabled**: ONNX Runtime Web needs WebAssembly execution. Some kiosk and enterprise browsers disable it by policy.
- **Slow large batches**: the app processes images sequentially and yields between files to keep the interface responsive. Very large images and large folders can still take time.
- **No CSV download**: confirm the browser allows downloads initiated by local pages.
