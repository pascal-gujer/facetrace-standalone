#!/usr/bin/env python3
"""Build the self-contained FaceTrace Offline HTML file.

This project intentionally ships as one large HTML file for restricted
environments that cannot rely on localhost, a web server, remote URLs, CDNs, or
file:// subresource loading. The editable source stays split under src/, while
this script inlines the CSS, application JavaScript, vendored face-api UMD
bundle (which itself ships TensorFlow.js v4 internally), and the
gzip-compressed local model bundle into index.html.

The model bundle (face detection, landmarks, and ArcFace recognition) is built
from the unpacked manifests/shards in models/, packed into a single JSON map,
gzip-compressed, base64-encoded, and decompressed in the browser via the
WHATWG DecompressionStream API. This shrinks the embedded payload meaningfully
versus base64-of-uncompressed.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = ROOT / "src" / "index.template.html"
STYLE = ROOT / "src" / "styles.css"
CANVAS_PATCH = ROOT / "src" / "canvas-readback-patch.js"
APP = ROOT / "src" / "app.js"
LOCALES_DIR = ROOT / "src" / "locales"
FACE_API = ROOT / "vendor" / "face-api.min.js"
MODEL_BUNDLE = ROOT / "models" / "embedded-models.js"
DEFAULT_OUTPUT = ROOT / "index.html"

REPLACEMENTS = {
    "{{FACETRACE_CSS}}": STYLE,
    "{{FACETRACE_CANVAS_PATCH_JS}}": CANVAS_PATCH,
    "{{FACETRACE_FACE_API_JS}}": FACE_API,
    "{{FACETRACE_APP_JS}}": APP,
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing build input: {path.relative_to(ROOT)}") from exc


def inline_css(text: str) -> str:
    if "</style" in text.lower():
        raise SystemExit("Refusing to inline CSS containing a closing </style tag")
    return text.rstrip() + "\n"


def inline_script(text: str) -> str:
    # Prevent an accidental literal closing script tag inside a vendored bundle
    # or generated model blob from terminating the surrounding HTML script block.
    safe = text.replace("</script", "<\\/script")
    return safe.rstrip() + "\n"


def collect_face_api_assets(entries: dict[str, dict[str, str]]) -> None:
    """Embed face-api.js detector and landmark assets keyed by filename."""
    models_dir = ROOT / "models"
    manifests = sorted(models_dir.glob("*-weights_manifest.json"))
    if not manifests:
        raise SystemExit("No face-api weight manifests found in models/")

    for manifest_path in manifests:
        manifest_text = read_text(manifest_path)
        entries[manifest_path.name] = {"kind": "json", "text": manifest_text}

        try:
            manifest = json.loads(manifest_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid model manifest JSON: {manifest_path.relative_to(ROOT)}"
            ) from exc

        for group in manifest:
            for shard_name in group.get("paths", []):
                shard_path = manifest_path.parent / shard_name
                if not shard_path.exists():
                    raise SystemExit(
                        f"Missing model shard: {shard_path.relative_to(ROOT)}"
                    )
                entries[shard_path.name] = {
                    "kind": "binary",
                    "base64": base64.b64encode(shard_path.read_bytes()).decode("ascii"),
                }


def collect_arcface_assets(entries: dict[str, dict[str, str]]) -> None:
    """Embed the converted ArcFace TF.js GraphModel: model.json + shards."""
    arcface_dir = ROOT / "models" / "arcface"
    model_json_path = arcface_dir / "model.json"
    if not model_json_path.exists():
        raise SystemExit(
            "Missing ArcFace model.json. Run conversion (see tools/README) "
            "and place model.json + shards under models/arcface/."
        )

    model_json_text = read_text(model_json_path)
    entries[model_json_path.name] = {"kind": "json", "text": model_json_text}

    try:
        spec = json.loads(model_json_text)
    except json.JSONDecodeError as exc:
        raise SystemExit("Invalid ArcFace model.json") from exc

    for group in spec.get("weightsManifest", []):
        for shard_name in group.get("paths", []):
            shard_path = arcface_dir / shard_name
            if not shard_path.exists():
                raise SystemExit(
                    f"Missing ArcFace shard: {shard_path.relative_to(ROOT)}"
                )
            # Filename collision guard: face-api shards already in entries
            # share a flat namespace with the ArcFace shards.
            if shard_path.name in entries:
                raise SystemExit(
                    f"Filename collision in embedded bundle: {shard_path.name}"
                )
            entries[shard_path.name] = {
                "kind": "binary",
                "base64": base64.b64encode(shard_path.read_bytes()).decode("ascii"),
            }


def load_locale_maps() -> dict[str, dict[str, str]]:
    required = ("en", "de", "fr")
    locales: dict[str, dict[str, str]] = {}

    for code in required:
        path = LOCALES_DIR / f"{code}.json"
        try:
            raw = json.loads(read_text(path))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid locale JSON: {path.relative_to(ROOT)}") from exc
        if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
            raise SystemExit(f"Locale file must be a flat string map: {path.relative_to(ROOT)}")
        locales[code] = raw

    base_keys = set(locales["en"].keys())
    for code in required:
        keys = set(locales[code].keys())
        missing = sorted(base_keys - keys)
        extra = sorted(keys - base_keys)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
            if extra:
                details.append(f"extra keys: {', '.join(extra[:8])}{'...' if len(extra) > 8 else ''}")
            raise SystemExit(f"Locale key mismatch in {code}.json ({'; '.join(details)})")

    return locales


def build_locales_script(locales: dict[str, dict[str, str]]) -> str:
    payload = json.dumps(locales, separators=(",", ":"), ensure_ascii=False)
    return (
        "/* Embedded locale resources for FaceTrace Offline. */\n"
        f"window.FACETRACE_EMBEDDED_LOCALES = {payload};\n"
    )


def build_model_bundle() -> str:
    entries: dict[str, dict[str, str]] = {}
    collect_face_api_assets(entries)
    collect_arcface_assets(entries)

    payload = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    # mtime=0 keeps most of the gzip header deterministic. Some Python/zlib
    # combinations still stamp platform-specific OS metadata into byte 9, so
    # normalize that byte to 255 ("unknown") for reproducible builds.
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    b64 = base64.b64encode(compressed).decode("ascii")

    return (
        "/* Generated local model bundle for FaceTrace Offline. Do not edit by hand.\n"
        f" * raw json bytes: {len(payload)}\n"
        f" * gzip bytes:     {len(compressed)}\n"
        f" * base64 bytes:   {len(b64)}\n"
        " */\n"
        f'window.FACETRACE_EMBEDDED_MODELS_GZIP_B64 = "{b64}";\n'
    )


def build_html(model_bundle: str, locales_script: str) -> str:
    html = read_text(TEMPLATE)

    for marker, path in REPLACEMENTS.items():
        if marker not in html:
            raise SystemExit(f"Template marker missing: {marker}")
        text = read_text(path)
        replacement = inline_css(text) if path.suffix == ".css" else inline_script(text)
        html = html.replace(marker, replacement, 1)

    model_marker = "{{FACETRACE_MODEL_BUNDLE_JS}}"
    if model_marker not in html:
        raise SystemExit(f"Template marker missing: {model_marker}")
    html = html.replace(model_marker, inline_script(model_bundle), 1)

    locale_marker = "{{FACETRACE_LOCALES_JS}}"
    if locale_marker not in html:
        raise SystemExit(f"Template marker missing: {locale_marker}")
    html = html.replace(locale_marker, inline_script(locales_script), 1)

    leftover = [m for m in (*REPLACEMENTS.keys(), model_marker, locale_marker) if m in html]
    if leftover:
        raise SystemExit(f"Unreplaced template marker(s): {', '.join(leftover)}")

    generated_note = (
        "<!--\n"
        "  Generated by tools/build.py. Edit src/*, vendor/face-api.min.js,\n"
        "  or the unpacked model assets in models/, then rebuild. index.html\n"
        "  is intentionally self-contained for offline file:// execution on\n"
        "  restricted systems. models/embedded-models.js is generated too.\n"
        "-->\n"
    )

    if html.startswith("<!doctype html>"):
        html = html.replace("<!doctype html>\n", "<!doctype html>\n" + generated_note, 1)
    else:
        html = generated_note + html

    validate_generated_html(html)
    return html


def validate_generated_html(html: str) -> None:
    disallowed = {
        "<script src>": re.compile(r"<script\b[^>]*\bsrc\s*=", re.IGNORECASE),
        "<link>": re.compile(r"<link\b", re.IGNORECASE),
        "<iframe>": re.compile(r"<iframe\b", re.IGNORECASE),
    }
    found = [name for name, pattern in disallowed.items() if pattern.search(html)]
    if found:
        raise SystemExit(
            f"Generated HTML contains disallowed external-loading tag(s): {', '.join(found)}"
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build self-contained index.html")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML path, default: index.html",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; fail if the output file is not up to date",
    )
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    generated_model_bundle = build_model_bundle()
    locales = load_locale_maps()
    generated_locales_script = build_locales_script(locales)
    generated = build_html(generated_model_bundle, generated_locales_script)

    if args.check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        existing_model_bundle = (
            MODEL_BUNDLE.read_text(encoding="utf-8") if MODEL_BUNDLE.exists() else ""
        )
        failed = False

        if existing_model_bundle != generated_model_bundle:
            print(f"{MODEL_BUNDLE.relative_to(ROOT)} is not up to date", file=sys.stderr)
            print(f"expected sha256 {sha256_text(generated_model_bundle)}", file=sys.stderr)
            print(f"existing sha256 {sha256_text(existing_model_bundle)}", file=sys.stderr)
            failed = True

        if existing != generated:
            print(f"{output.relative_to(ROOT)} is not up to date", file=sys.stderr)
            print(f"expected sha256 {sha256_text(generated)}", file=sys.stderr)
            print(f"existing sha256 {sha256_text(existing)}", file=sys.stderr)
            failed = True

        return 1 if failed else 0

    MODEL_BUNDLE.write_text(generated_model_bundle, encoding="utf-8")
    output.write_text(generated, encoding="utf-8")
    print(f"wrote {MODEL_BUNDLE.relative_to(ROOT)} ({len(generated_model_bundle):,} bytes)")
    print(f"wrote {output.relative_to(ROOT)} ({len(generated):,} bytes)")
    print(f"sha256 {sha256_text(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
