# FaceTrace Search Set Format

FaceTrace search sets are portable candidate-image indexes for FaceTrace
Offline. They let the browser app or an external generator precompute searchable
candidate face descriptors, thumbnails, and face crops, then later compare a new
reference face against that set without reprocessing the original candidate
images.

The format intentionally does not include the reference/search target image.

## File Types

The browser exports encrypted files with the extension:

```text
.facetrace-set
```

The exported file is UTF-8 JSON containing an encrypted envelope. The encrypted
payload is compressed JSON. Import also accepts plaintext search-set JSON and
gzip-compressed plaintext JSON for tooling and debugging, but encrypted exports
are the default.

## Envelope

Encrypted files use this outer shape:

```json
{
  "format": "facetrace-search-set-envelope",
  "envelopeVersion": 1,
  "header": {
    "format": "facetrace-search-set-envelope",
    "envelopeVersion": 1,
    "createdAt": "2026-05-03T20:00:00.000Z",
    "payload": {
      "format": "facetrace-search-set",
      "schemaVersion": 1,
      "compression": "gzip",
      "byteLength": 12345
    },
    "encryption": {
      "algorithm": "AES-256-GCM",
      "keyFormat": "ftsk1",
      "iv": "base64url-encoded-96-bit-iv",
      "tagLength": 128
    }
  },
  "ciphertext": "base64url-encoded-ciphertext-plus-gcm-tag"
}
```

The AES-GCM additional authenticated data is the canonical JSON form of
`header`, with object keys sorted recursively and no extra whitespace. This is
the same logical structure as `stableStringify(header)` in `src/app.js`.

The share key format is:

```text
ftsk1_<base64url-encoded-256-bit-raw-aes-key>
```

The payload is compressed before encryption. Compression is usually `gzip`; a
producer may use `none`, and importers should support both.

## Plain Payload

After decryption and decompression, the payload is JSON:

```json
{
  "format": "facetrace-search-set",
  "schemaVersion": 1,
  "createdAt": "2026-05-03T20:00:00.000Z",
  "producer": {
    "name": "FaceTrace Offline",
    "runtime": "browser"
  },
  "model": {
    "id": "facetrace-yunet-sface-128-v1",
    "embeddingDim": 128,
    "descriptorEncoding": "float32-le-base64",
    "detector": "OpenCV YuNet face_detection_yunet_2026may.onnx",
    "detectorThreshold": 0.7,
    "detectorNmsThreshold": 0.3,
    "landmarks": "YuNet 5-point landmarks",
    "recognizer": "OpenCV SFace face_recognition_sface_2021dec_int8.onnx",
    "alignment": "opencv-sface-5point-112",
    "tta": "none",
    "descriptorNormalization": "l2",
    "maxAnalysisSide": 1600,
    "thumbnailSide": 260,
    "faceCropSide": 144,
    "similarity": {
      "metric": "cosine",
      "sameIdentityThreshold": 0.363,
      "percentCenter": 0.363,
      "percentSlope": 14,
      "calibration": "SFace-specific heuristic; not comparable with legacy ArcFace percentages"
    }
  },
  "counts": {
    "items": 1,
    "faces": 1
  },
  "privacy": {
    "containsBiometricDescriptors": true,
    "containsFullOriginalImages": false,
    "containsPreviewImages": true
  },
  "items": []
}
```

Importers must reject incompatible descriptor dimensions or descriptor
encodings. Producers should keep `model.id` stable only when they produce
descriptors with the same model assets, alignment, test-time augmentation, and
normalization behavior.

The browser importer also recognizes legacy
`facetrace-arcface-256-v1` / 256-dimension search sets exported by older
FaceTrace builds. Those legacy descriptors are not compared directly with the
current SFace reference. The app asks for confirmation and attempts to
reprocess stored thumbnails or face crops into the current
`facetrace-yunet-sface-128-v1` descriptor format.

## Items

Each item represents one candidate image or source record:

```json
{
  "id": "item-1",
  "fileName": "IMG_0012.jpg",
  "displayName": "Alice Example",
  "sourceUrl": "https://example.com/about/alice",
  "attribution": {
    "author": "Example Photographer",
    "license": "CC BY 4.0",
    "sourceLink": "https://example.com/photos/IMG_0012",
    "notes": "Optional credit line or dataset note"
  },
  "fileSize": 123456,
  "fileType": "image/jpeg",
  "width": 1200,
  "height": 900,
  "status": "ok",
  "thumbnail": {
    "mediaType": "image/jpeg",
    "encoding": "base64",
    "byteLength": 8192,
    "sha256": "base64url-sha256",
    "data": "base64-jpeg-bytes"
  },
  "faces": []
}
```

`displayName`, `sourceUrl`, and `attribution` are optional for private local
exports. The browser app currently exports an empty `sourceUrl` for local files.
External crawlers should set `sourceUrl` when the item came from a page, profile,
or manifest URL.

`sourceUrl` is provenance for the source record. `attribution.sourceLink` is the
legal/source credit link shown in the FaceTrace UI.

The browser export does not include full original images. It includes a small
thumbnail and face crops only.

## Attribution And Licensing

Datasets that are redistributed, bundled as examples, or generated from
third-party material should include item-level attribution:

```json
{
  "attribution": {
    "author": "Example Photographer",
    "license": "CC BY 4.0",
    "sourceLink": "https://example.com/photo",
    "notes": "Optional extra credit or dataset context"
  }
}
```

For redistributed datasets, `author`, `license`, and `sourceLink` must be filled
for every item where legal attribution is required. `notes` is optional.
The browser import preserves these fields, exposes them in each result's
technical details, and shows a source/license info popup for records that carry
attribution data.

If one original image contains multiple detected faces, the attribution applies
to that image item and therefore to the thumbnail and face crops derived from it.
If individual faces require different legal metadata, generators should emit
separate items.

## Faces

Each face entry contains one searchable face descriptor:

```json
{
  "index": 0,
  "descriptor": {
    "dimensions": 128,
    "encoding": "float32-le-base64",
    "data": "base64-little-endian-float32-bytes"
  },
  "detectorScore": 0.93,
  "box": {
    "x": 100,
    "y": 80,
    "width": 220,
    "height": 220
  },
  "quality": {
    "detectorScore": 0.93,
    "yawRatio": 1.1,
    "pitchRatio": 1.0,
    "rollDegrees": 2.3,
    "blurVariance": 82.4
  },
  "crop": {
    "mediaType": "image/jpeg",
    "encoding": "base64",
    "byteLength": 4096,
    "sha256": "base64url-sha256",
    "data": "base64-jpeg-bytes"
  }
}
```

Descriptors are stored as exactly 128 little-endian IEEE-754 `float32` values.
The bytes are base64-encoded using regular base64, not base64url.

## External Generator Requirements

An external generator should only claim `model.id:
"facetrace-yunet-sface-128-v1"` when it matches the browser pipeline:

- face detection: OpenCV YuNet `face_detection_yunet_2026may.onnx`
- detector score threshold: 0.7
- detector NMS threshold: 0.3
- landmarks: YuNet 5-point output
- alignment: OpenCV SFace 5-point 112x112 template
- recognizer: OpenCV SFace `face_recognition_sface_2021dec_int8.onnx`
- descriptor: single aligned crop embedding, then L2-normalized
- descriptor encoding: 128 little-endian float32 values

If a tool uses a different model, alignment, embedding dimension, or TTA policy,
it should use a different `model.id`. The current browser importer rejects
non-128 or non-`float32-le-base64` current descriptors. Older
`facetrace-arcface-256-v1` files are treated as migration inputs only.

## Privacy

Search sets contain biometric face descriptors and preview images. They should
be treated as sensitive personal data even when full original photos are not
included. Encrypted `.facetrace-set` files should be shared separately from
their `ftsk1_...` share keys.
