#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NanoBanana API (production-leaning)
- /generate: Call Google Gemini with prompt and optional image (URL or upload)
- /zillow/download: Scrape Zillow listing images, zip them, and return metadata

Notes:
- Set GOOGLE_API_KEY in environment (or .env)
- Consider running behind a production WSGI server (e.g., gunicorn/uvicorn)
- Playwright is optional; /zillow/download will error with a clear message if missing
"""

# ==============================
# Standard Library Imports
# ==============================
import os
import sys
import uuid
import re
import time
import zipfile
import logging
from io import BytesIO
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from html.parser import HTMLParser

# ==============================
# Third-Party Imports
# ==============================
from dotenv import load_dotenv
import requests
from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError

# Google Generative AI client (pip install google-genai)
from google import genai

# Playwright is optional; we detect availability
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# ==============================
# Environment & Config
# ==============================
load_dotenv()  # Load from .env if present

class Config:
    # App
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", 5000))
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    JSON_SORT_KEYS: bool = False

    # Limits
    MAX_IMAGE_BYTES: int = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))  # 10MB
    # Global incoming request size (Flask will reject larger)
    MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_CONTENT_LENGTH", 25 * 1024 * 1024))  # 25MB

    # Output
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "generated")

    # Google GenAI
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GOOGLE_GENAI_MODEL: str = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-image-preview")

    # Network
    HTTP_TIMEOUT: int = int(os.environ.get("HTTP_TIMEOUT", 25))
    HTTP_CHUNK: int = int(os.environ.get("HTTP_CHUNK", 8192))

# Validate critical env
if not Config.GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

# Ensure output directory exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==============================
# Logging
# ==============================
LOG_LEVEL = logging.DEBUG if Config.DEBUG else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("nanobanana")

# ==============================
# HTTP Session
# ==============================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "NanoBanana/1.0 (+https://example.com) PythonRequests",
    "Accept": "image/avif,image/webp,image/apng,image/*;q=0.8,*/*;q=0.5",
})

# ==============================
# Flask App & Client
# ==============================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = Config.JSON_SORT_KEYS

client = genai.Client(api_key=Config.GOOGLE_API_KEY)

# ==============================
# Utilities
# ==============================
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
    path = os.path.join(Config.OUTPUT_DIR, fname)
    to_save = img
    if suffix.lower() in ("jpg", "jpeg") and img.mode not in ("RGB", "L"):
        to_save = img.convert("RGB")
    to_save.save(path)
    logger.debug("Saved image: %s", path)
    return path

def _verify_image_bytes(img_bytes: bytes) -> Image.Image:
    """
    Verify image bytes and return a *fresh* PIL.Image object.
    """
    try:
        probe = Image.open(BytesIO(img_bytes))
        probe.verify()
    except UnidentifiedImageError:
        raise ValueError("Provided data is not a valid image.")
    except Exception as e:
        raise ValueError(f"Image verification failed: {e}")

    try:
        img = Image.open(BytesIO(img_bytes))
        img.load()
        return img
    except Exception as e:
        raise ValueError(f"Could not reopen image after verification: {e}")

def _download_image(image_url: str) -> Tuple[Image.Image, str]:
    """
    Download and validate an image from HTTP(S), enforcing size limit.
    Returns (PIL.Image, mime_type).
    """
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must be http(s)")

    try:
        with SESSION.get(image_url, stream=True, timeout=Config.HTTP_TIMEOUT) as r:
            if r.status_code >= 400:
                raise ValueError(f"Image URL returned HTTP {r.status_code}")

            mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            buf = BytesIO()
            total = 0
            for chunk in r.iter_content(chunk_size=Config.HTTP_CHUNK):
                if not chunk:
                    continue
                total += len(chunk)
                if total > Config.MAX_IMAGE_BYTES:
                    raise ValueError(f"Image too large (> {Config.MAX_IMAGE_BYTES} bytes).")
                buf.write(chunk)
        data = buf.getvalue()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {e}") from e

    img = _verify_image_bytes(data)

    fmt = (img.format or "").lower()
    if not (mime.startswith("image/") and len(mime) > 6):
        if fmt:
            mime = f"image/{'jpeg' if fmt in ('jpeg', 'jpg') else fmt}"
        else:
            mime = "image/png"

    return img, mime

def _read_upload_file(file_storage) -> Tuple[Image.Image, str]:
    """
    Read an uploaded file object safely with size limit, return (PIL.Image, mime).
    """
    buf = BytesIO()
    total = 0
    while True:
        chunk = file_storage.stream.read(Config.HTTP_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > Config.MAX_IMAGE_BYTES:
            raise ValueError(f"Uploaded image too large (> {Config.MAX_IMAGE_BYTES} bytes).")
        buf.write(chunk)

    data = buf.getvalue()
    img = _verify_image_bytes(data)

    mime = (file_storage.mimetype or "").split(";")[0].strip().lower()
    fmt = (img.format or "").lower()
    if not (mime.startswith("image/") and len(mime) > 6):
        mime = f"image/{'jpeg' if fmt in ('jpeg', 'jpg') else (fmt or 'png')}"
    return img, mime

def download_images_from_zillow(zillow_url: str) -> Tuple[Image.Image, str]:
    """
    Placeholder for specialized Zillow image download. Currently delegates to _download_image.
    """
    return _download_image(zillow_url)

def _scrape_zillow_and_zip(
    zillow_url: str,
    target_format: str = "jpeg",
    target_size: str = "1536",
) -> Tuple[str, int]:
    """
    Open a Zillow listing, navigate to the media wall, collect image URLs, download them,
    and save into a zip under OUTPUT_DIR. Returns (zip_path, num_images).
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright is required for Zillow scraping. Install with:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )

    # Normalize options
    target_format = (target_format or "jpeg").lower().strip()
    if target_format not in {"jpeg", "webp"}:
        target_format = "jpeg"
    target_size = (str(target_size) or "1536").strip()
    if not target_size.isdigit():
        target_size = "1536"

    urls: List[str] = []
    size_re = re.compile(r"_(\d+)\.(?:jpg|jpeg|webp)$", re.IGNORECASE)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        logger.info("Navigating to Zillow URL: %s", zillow_url)
        page.goto(zillow_url, wait_until="domcontentloaded", timeout=90_000)

        try:
            btn = page.locator('[data-testid="gallery-see-all-photos-button"]')
            if btn.count() > 0:
                btn.first.click(timeout=15_000)
        except Exception:
            pass  # Some pages already show the media wall

        page.wait_for_selector('div[data-testid="hollywood-vertical-media-wall"]', timeout=30_000)

        start = time.time()
        while time.time() - start < 5.0:
            try:
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                page.locator('div[data-testid="hollywood-vertical-media-wall"]').evaluate(
                    "el => { el.scrollTop = el.scrollHeight; }"
                )
            except Exception:
                pass
            time.sleep(0.3)

        selector = f'div[data-testid="hollywood-vertical-media-wall"] source[type="image/{target_format}"]'
        try:
            srcsets = page.eval_on_selector_all(
                selector,
                "nodes => nodes.map(n => n.getAttribute('srcset') || '')"
            )
        except Exception:
            srcsets = []

        cand_urls: List[str] = []
        for srcset in srcsets:
            if not srcset:
                continue
            last_part = srcset.split(',')[-1].strip()
            url = last_part.split(" ")[0]
            url = size_re.sub(f"_{target_size}.{target_format}", url)
            cand_urls.append(url)

        seen = set()
        for u in cand_urls:
            if u and u not in seen:
                seen.add(u)
                urls.append(u)

        context.close()
        browser.close()

    if not urls:
        raise RuntimeError("No images found on Zillow media wall.")

    zip_name = f"zillow_images_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(Config.OUTPUT_DIR, zip_name)

    written = 0
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, url in enumerate(urls, start=1):
            try:
                r = SESSION.get(url, timeout=30)
                if r.status_code >= 400:
                    continue
                ext = target_format if target_format != "jpeg" else "jpg"
                zf.writestr(f"images/image_{idx}.{ext}", r.content)
                written += 1
            except Exception as e:
                logger.warning("Skipping image due to error: %s", e)

    return zip_path, written

def find_all_images_in_html(html: str) -> List[str]:
    """
    Simple parser to find all <img src="..."> URLs in an HTML string.
    """
    class ImgSrcParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.img_srcs: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() == "img":
                attr_dict = dict(attrs)
                src = attr_dict.get("src")
                if src:
                    self.img_srcs.append(src)

    parser = ImgSrcParser()
    parser.feed(html)
    return parser.img_srcs

# ==============================
# Error Handlers
# ==============================
@app.errorhandler(400)
def _bad_request(e):
    return jsonify(error="Bad Request", detail=str(e)), 400

@app.errorhandler(404)
def _not_found(e):
    return jsonify(error="Not Found"), 404

@app.errorhandler(413)
def _too_large(e):
    return jsonify(error="Payload Too Large"), 413

@app.errorhandler(500)
def _server_error(e):
    logger.exception("Internal server error: %s", e)
    return jsonify(error="Internal Server Error"), 500

# ==============================
# Routes
# ==============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/zillow/download", methods=["POST"])
def zillow_download():
    """
    POST JSON:
      {
        "url": "https://www.zillow.com/...",
        "format": "jpeg|webp",   // optional, default "jpeg"
        "size": "1536"           // optional
      }

    Returns: { "zip": "<path>", "count": <int> }
    """
    if not request.is_json:
        return jsonify(error="Send JSON body."), 400
    data = request.get_json(silent=True) or {}
    zurl = (data.get("url") or "").strip()
    if not zurl:
        return jsonify(error="url required"), 400

    target_format = (data.get("format") or "jpeg").strip()
    target_size = (data.get("size") or "1536").strip()

    try:
        zip_path, count = _scrape_zillow_and_zip(zurl, target_format, target_size)
        logger.info("Zillow zip created: %s (images: %d)", zip_path, count)
    except Exception as e:
        logger.exception("Zillow scraping failed")
        return jsonify(error=str(e)), 500

    # Tip: if you'd rather stream the zip, return send_file(zip_path, as_attachment=True)
    return jsonify(zip=zip_path, count=count), 200

@app.route("/generate", methods=["POST"])
def generate():
    """
    Accepts:
      - JSON: {"prompt": "text...", "image_url": "https://..."}  (image_url optional)
      - multipart/form-data: prompt=<text>, image=<file>         (image optional)

    Behavior:
      - If an image is provided (via URL or file), send [prompt, image] to model (edit/variation).
      - Else send [prompt] only.
    """
    model = Config.GOOGLE_GENAI_MODEL
    prompt: Optional[str] = None
    img: Optional[Image.Image] = None

    ctype = (request.content_type or "").lower()

    # Multipart form with optional file
    if ctype.startswith("multipart/form-data"):
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            return jsonify(error="prompt required"), 400

        file = request.files.get("image")
        if file and file.filename:
            try:
                img, _mime = _read_upload_file(file)
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
                img, _mime = _download_image(image_url)
            except ValueError as ve:
                return jsonify(error=str(ve)), 400

    # Build contents for the API call
    contents: List = [prompt]
    if img is not None:
        contents.append(img)

    # Call Gemini
    try:
        resp = client.models.generate_content(model=model, contents=contents)
    except Exception as e:
        logger.exception("Model error")
        return jsonify(error=f"Model error: {e}"), 502

    # Parse response safely
    texts: List[str] = []
    images: List[str] = []

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
                    logger.warning("Failed to save returned image: %s", e)
                    texts.append(f"[warn] Failed to save returned image: {e}")
    except Exception as e:
        logger.exception("Failed to parse model response")
        return jsonify(error=f"Failed to parse model response: {e}"), 500

    return jsonify(model=model, texts=texts, images=images), 200

# ==============================
# Entrypoint
# ==============================
def main() -> None:
    logger.info("Starting NanoBanana on %s:%d (debug=%s)", Config.HOST, Config.PORT, Config.DEBUG)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

if __name__ == "__main__":
    main()
