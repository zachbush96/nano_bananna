#!/usr/bin/env python3
import os
import uuid
from io import BytesIO
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError

from google import genai

# ------------------------------
# Config
# ------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-image-preview")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default 10 MB cap (override with env)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))

# HTTP session w/ realistic headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "NanoBanana/1.0 (+https://example.com) PythonRequests",
    "Accept": "image/avif,image/webp,image/apng,image/*;q=0.8,*/*;q=0.5",
})

app = Flask(__name__)
client = genai.Client(api_key=GOOGLE_API_KEY)

# ------------------------------
# Helpers
# ------------------------------
def _infer_suffix_from_mime(mime: str) -> str:
    if not mime or "/" not in mime:
        return "png"
    subtype = mime.split("/")[-1].lower().strip()
    if subtype in ("jpeg", "jpg", "pjpeg"):
        return "jpg"
    if subtype in ("png", "gif", "webp", "bmp", "tiff"):
        return subtype
    return "png"

def _save_pil(img: Image.Image, suffix: str = "png") -> str:
    fname = f"{uuid.uuid4().hex}.{suffix}"
    path = os.path.join(OUTPUT_DIR, fname)
    # Ensure RGB for formats like JPEG; keep mode otherwise when possible
    to_save = img
    if suffix.lower() in ("jpg", "jpeg") and img.mode not in ("RGB", "L"):
        to_save = img.convert("RGB")
    to_save.save(path)
    return path

def _verify_image_bytes(img_bytes: bytes) -> Image.Image:
    """
    Verify that img_bytes is a real image and return a *fresh* PIL.Image object
    ready to be consumed by the Google client.
    """
    # First quick verify (detects truncation/invalid headers)
    try:
        probe = Image.open(BytesIO(img_bytes))
        probe.verify()
    except UnidentifiedImageError:
        raise ValueError("Provided data is not a valid image.")
    except Exception as e:
        raise ValueError(f"Image verification failed: {e}")

    # Re-open after verify (verify() leaves the parser closed)
    try:
        img = Image.open(BytesIO(img_bytes))
        img.load()  # fully load into memory
        return img
    except Exception as e:
        raise ValueError(f"Could not reopen image after verification: {e}")

def _download_image(image_url: str) -> tuple[Image.Image, str]:
    """
    Download and validate an image from HTTP(S), enforcing size limit.
    Returns (PIL.Image, mime_type).
    """
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must be http(s)")

    try:
        with SESSION.get(image_url, stream=True, timeout=25) as r:
            if r.status_code >= 400:
                raise ValueError(f"Image URL returned HTTP {r.status_code}")

            mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            buf = BytesIO()
            total = 0
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image too large (> {MAX_IMAGE_BYTES} bytes).")
                buf.write(chunk)
        data = buf.getvalue()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {e}") from e

    img = _verify_image_bytes(data)

    # If server didn't give a usable mime, infer from PIL
    fmt = (img.format or "").lower()
    if not (mime.startswith("image/") and len(mime) > 6):
        if fmt:
            mime = f"image/{'jpeg' if fmt in ('jpeg', 'jpg') else fmt}"
        else:
            mime = "image/png"

    return img, mime

def _read_upload_file(file_storage) -> tuple[Image.Image, str]:
    """
    Read an uploaded file object safely with size limit, return (PIL.Image, mime)
    """
    # Enforce size limit while reading
    buf = BytesIO()
    total = 0
    while True:
        chunk = file_storage.stream.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError(f"Uploaded image too large (> {MAX_IMAGE_BYTES} bytes).")
        buf.write(chunk)
    data = buf.getvalue()
    img = _verify_image_bytes(data)

    mime = (file_storage.mimetype or "").split(";")[0].strip().lower()
    fmt = (img.format or "").lower()
    if not (mime.startswith("image/") and len(mime) > 6):
        mime = f"image/{'jpeg' if fmt in ('jpeg', 'jpg') else fmt or 'png'}"
    return img, mime

def download_images_from_zillow(zillow_url: str) -> tuple[Image.Image, str]:
    """
    Specialized downloader for Zillow images, which may have specific URL patterns.
    """
    # Example implementation; adjust as needed based on actual Zillow URL structures
    return _download_image(zillow_url)

def find_all_images_in_html(html: str) -> list[str]:
    """
    Simple parser to find all image URLs in an HTML string.
    """
    from html.parser import HTMLParser

    class ImgSrcParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.img_srcs = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() == "img":
                attr_dict = dict(attrs)
                src = attr_dict.get("src")
                if src:
                    self.img_srcs.append(src)

    parser = ImgSrcParser()
    parser.feed(html)
    return parser.img_srcs

# ------------------------------
# Route
# ------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    """
    Accepts:
      - JSON: {"prompt": "text...", "image_url": "https://..."}  (image_url optional)
      - multipart/form-data: prompt=<text>, image=<file>         (image optional)

    Behavior:
      - If an image is provided (via URL or file), send [prompt, image] to model (image edit/variation).
      - Else send [prompt] (pure text-to-image / text output, depending on model behavior).
    """
    model = DEFAULT_MODEL
    prompt = None
    img = None
    mime = None

    ctype = request.content_type or ""

    # Multipart form with optional file
    if ctype.startswith("multipart/form-data"):
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            return jsonify(error="prompt required"), 400

        file = request.files.get("image")
        if file and file.filename:
            try:
                img, mime = _read_upload_file(file)
            except ValueError as ve:
                return jsonify(error=str(ve)), 400

    # JSON body with optional image_url
    else:
        if not request.is_json:
            return jsonify(error="Send JSON or multipart/form-data"), 400
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify(error="prompt required"), 400

        image_url = (data.get("image_url") or "").strip()
        if image_url:
            try:
                img, mime = _download_image(image_url)
            except ValueError as ve:
                return jsonify(error=str(ve)), 400

    # Build contents for the API call
    contents = [prompt]
    if img is not None:
        # The Google genai client accepts a PIL.Image directly in contents
        contents.append(img)

    # Call Gemini
    try:
        resp = client.models.generate_content(model=model, contents=contents)
    except Exception as e:
        return jsonify(error=f"Model error: {e}"), 502

    # Parse response safely
    texts = []
    images = []

    try:
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            return jsonify(model=model, texts=[], images=[], note="No candidates returned."), 200

        parts = getattr(candidates[0].content, "parts", []) or []
        for part in parts:
            # Text part
            if getattr(part, "text", None) is not None:
                texts.append(part.text)

            # Inline image data part
            elif getattr(part, "inline_data", None) is not None:
                part_mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                suffix = _infer_suffix_from_mime(part_mime)
                data = getattr(part.inline_data, "data", None)
                if not data:
                    continue
                try:
                    out_img = Image.open(BytesIO(data))
                    out_img.load()
                    out_path = _save_pil(out_img, suffix)
                    images.append(out_path)
                except Exception as e:
                    texts.append(f"[warn] Failed to save returned image: {e}")
    except Exception as e:
        return jsonify(error=f"Failed to parse model response: {e}"), 500

    return jsonify(model=model, texts=texts, images=images), 200

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    # Bind all interfaces for easy LAN testing
    app.run(host="0.0.0.0", port=5000, debug=True)
