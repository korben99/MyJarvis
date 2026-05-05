#!/opt/jarvis/venv/bin/python3
"""
Upload files to OpenWebUI via its API, then create/update a Knowledge base.
OpenWebUI handles chunking, embedding, and Qdrant storage with proper file_ids.

Usage:
  python3 upload-to-openwebui.py --api-key YOUR_API_KEY
  python3 upload-to-openwebui.py --api-key YOUR_API_KEY --knowledge-name "Jarvis"
  python3 upload-to-openwebui.py --api-key YOUR_API_KEY --dry-run
  python3 upload-to-openwebui.py --api-key YOUR_API_KEY --reset-tracking
  python3 upload-to-openwebui.py --api-key YOUR_API_KEY --file-number 20
"""

import argparse
import json
import mimetypes
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv("/opt/jarvis/.env")

# Optional extraction libraries — used as fallback when OpenWebUI returns "empty content"
try:
    import logging as _logging

    import pypdf

    _logging.getLogger("pypdf").setLevel(
        _logging.ERROR
    )  # silence "Ignoring wrong pointing object" warnings
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from docx import Document as DocxDocument

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from html.parser import HTMLParser

    HAS_HTML = True  # stdlib, always available
except ImportError:
    HAS_HTML = False


OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://localhost:3000")
DATA_DIR = os.getenv("DATA_DIR", "/opt/jarvis/RAGData")
TRACKING_FILE = "/opt/jarvis/logs/uploaded-files.json"
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")

# no pdf at the begining no .py, .html, .yml...
EXTS = {".xls", ".txt", ".md", ".csv", ".docx", ".pdf"}
MAX_FILE_SIZE_MB = 1  # skip files larger than this
UPLOAD_TIMEOUT = 180  # seconds per upload
ADD_TIMEOUT = 300  # seconds for knowledge add call (embedding pipeline can be slow)
MAX_RETRIES = 3  # retries per API call
SAVE_EVERY = 1  # save tracking file every N successful uploads


class _HTMLTextExtractor(HTMLParser if HAS_HTML else object):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts)


def extract_text_locally(path: Path) -> str | None:
    """Try to extract plain text from a file using local libraries.
    Returns extracted text string, or None if extraction is not possible."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        if not HAS_PYPDF:
            return None
        try:
            reader = pypdf.PdfReader(str(path))
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n\n".join(parts) if parts else None
        except Exception as e:
            print(f"  [pypdf error: {e}]")
            return None

    if ext in (".docx", ".doc"):
        if not HAS_DOCX:
            return None
        try:
            doc = DocxDocument(str(path))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(parts) if parts else None
        except Exception as e:
            print(f"  [python-docx error: {e}]")
            return None

    if ext == ".html":
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            text = parser.get_text().strip()
            return text if text else None
        except Exception as e:
            print(f"  [html parse error: {e}]")
            return None

    # For plain-text formats, just read the file
    if ext in (
        ".txt",
        ".md",
        ".rst",
        ".csv",
        ".py",
        ".js",
        ".sh",
        ".yaml",
        ".yml",
        ".json",
    ):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return text if text else None
        except Exception as e:
            print(f"  [read error: {e}]")
            return None

    return None


# Graceful shutdown on Ctrl+C
_shutdown = False


def _sigint(sig, frame):
    global _shutdown
    print("\n[Interrupted — will stop after current file]")
    _shutdown = True


signal.signal(signal.SIGINT, _sigint)


def make_session():
    """HTTP session with retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class APIError(Exception):
    """Raised for permanent HTTP client errors (4xx) that should not be retried."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:200]}")


def api(session, method, path, headers, timeout=60, **kwargs):
    """Make an API call, return parsed JSON or None on transient error.
    Raises APIError for permanent 4xx failures."""
    url = f"{OPENWEBUI_URL}{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.request(method, url, headers=headers, timeout=timeout, **kwargs)
            if r.ok:
                return r.json()
            # Non-retryable client errors — raise so caller can inspect status/body
            if r.status_code in (400, 401, 403, 404, 422):
                raise APIError(r.status_code, r.text)
            # Server-side — retry
            print(
                f"  HTTP {r.status_code} (attempt {attempt}/{MAX_RETRIES}): {r.text[:100]}"
            )
        except APIError:
            raise  # don't swallow permanent errors
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            print(f"  Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError as e:
            # ReadTimeoutError is sometimes wrapped as ConnectionError
            if "ReadTimeout" in type(e.__cause__).__name__ if e.__cause__ else False:
                print(f"  Read timeout (attempt {attempt}/{MAX_RETRIES})")
            else:
                print(f"  Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
        except Exception as e:
            print(f"  Request error (attempt {attempt}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES:
            time.sleep(2**attempt)  # 2s, 4s backoff
    return None


def get_or_create_knowledge(session, headers, name, uploaded):
    # Use cached kb_id from tracking file if available
    cached_id = uploaded.get("_kb_id")
    if cached_id:
        # Verify it still exists
        resp = api(session, "GET", f"/api/v1/knowledge/{cached_id}", headers=headers)
        if resp and resp.get("id"):
            print(f"Reusing knowledge base: '{name}' (id: {cached_id})")
            return cached_id
        print(f"Cached kb_id {cached_id} not found, looking up by name...")

    # Fall back to name lookup
    resp = api(session, "GET", "/api/v1/knowledge/", headers=headers) or []
    kb_list = resp if isinstance(resp, list) else resp.get("data", [])
    for kb in kb_list:
        if kb.get("name") == name:
            print(f"Found existing knowledge base: '{name}' (id: {kb['id']})")
            uploaded["_kb_id"] = kb["id"]
            return kb["id"]

    # Create new
    kb = api(
        session,
        "POST",
        "/api/v1/knowledge/create",
        headers=headers,
        json={"name": name, "description": f"Auto-indexed from {DATA_DIR}"},
    )
    if kb:
        print(f"Created knowledge base: '{name}' (id: {kb['id']})")
        uploaded["_kb_id"] = kb["id"]
        return kb["id"]
    return None


def collect_files(data_dir, uploaded):
    """Collect all eligible files that haven't been uploaded yet."""
    files = []
    skipped_size = []
    data_path = Path(data_dir)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    _excluded = {data_path / "Trade"}

    for path in sorted(data_path.rglob("*")):
        if not path.is_file():
            continue
        if any(path.is_relative_to(exc) for exc in _excluded):
            continue
        if path.name.startswith("._"):  # macOS resource fork — not a real file
            continue
        if path.suffix.lower() not in EXTS:
            continue
        size = path.stat().st_size
        if size > max_bytes:
            skipped_size.append(str(path.relative_to(data_path)))
            continue
        mtime = str(path.stat().st_mtime)
        key = f"{path}:{mtime}"
        if key not in uploaded:
            files.append((path, key))

    if skipped_size:
        print(
            f"Skipped {len(skipped_size)} files exceeding {MAX_FILE_SIZE_MB}MB size limit"
        )

    return files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=OPENWEBUI_API_KEY, help="OpenWebUI API KEY")
    p.add_argument(
        "--knowledge-name", default="Jarvis Knowledge", help="Knowledge base name"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="List files without uploading"
    )
    p.add_argument(
        "--reset-tracking",
        action="store_true",
        help="Clear tracking file and re-upload all",
    )
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory to index")
    p.add_argument(
        "--file-number",
        type=int,
        default=0,
        help="Max files to process per run (0 = unlimited)",
    )
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry files previously marked as errors",
    )
    return p.parse_args()


def load_tracking():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not read tracking file ({e}), starting fresh")
    return {}


def save_tracking(uploaded):
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    tmp = TRACKING_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(uploaded, f)
    os.replace(tmp, TRACKING_FILE)  # atomic write


def main():
    args = parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"}
    session = make_session()

    # Verify connectivity
    me = api(session, "GET", "/api/v1/auths/", headers=headers)
    if me is None:
        print(
            "Cannot reach OpenWebUI or invalid API key. Check --api-key and OPENWEBUI_URL."
        )
        sys.exit(1)

    # Tracking
    uploaded = {} if args.reset_tracking else load_tracking()
    if args.reset_tracking:
        print("Tracking reset — will re-upload all files.")
    elif args.retry_errors:
        errors_cleared = {
            k: v
            for k, v in uploaded.items()
            if isinstance(v, str) and v.startswith("ERROR:")
        }
        for k in errors_cleared:
            del uploaded[k]
        print(f"Cleared {len(errors_cleared)} error entries — will retry them.")

    # Collect files
    files = collect_files(args.data_dir, uploaded)
    total_pending = len(files)
    if args.file_number > 0:
        files = files[: args.file_number]
    done_count = sum(1 for k in uploaded if not k.startswith("_"))
    print(
        f"Files to upload: {len(files)} (of {total_pending} pending, {done_count} already done)"
    )

    if args.dry_run:
        for p, _ in files[:50]:
            print(f"  {p.relative_to(args.data_dir)}")
        if len(files) > 50:
            print(f"  ... and {len(files) - 50} more")
        return

    if not files:
        print("Nothing new to upload.")
        return

    # Get/create knowledge base
    kb_id = get_or_create_knowledge(session, headers, args.knowledge_name, uploaded)
    if not kb_id:
        print("Could not get/create knowledge base. Aborting.")
        sys.exit(1)
    save_tracking(uploaded)  # persist kb_id immediately

    ok, errors, skipped_empty = 0, 0, 0

    for i, (path, key) in enumerate(files):
        if _shutdown:
            break

        label = str(path.relative_to(args.data_dir))
        label_short = (label[:67] + "...") if len(label) > 70 else label
        print(f"[{i + 1}/{len(files)}] {label_short}", end=" ", flush=True)

        try:
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

            with open(path, "rb") as fh:
                upload = api(
                    session,
                    "POST",
                    "/api/v1/files/",
                    headers=headers,
                    timeout=UPLOAD_TIMEOUT,
                    files={"file": (path.name, fh, mime)},
                )

            if not upload:
                print("UPLOAD FAILED")
                errors += 1
                uploaded[key] = "ERROR:upload_failed"
                continue

            file_id = upload.get("id")
            if not file_id:
                print(f"UPLOAD FAILED (no id in response: {str(upload)[:100]})")
                errors += 1
                uploaded[key] = "ERROR:no_file_id"
                continue

            try:
                result = api(
                    session,
                    "POST",
                    f"/api/v1/knowledge/{kb_id}/file/add",
                    headers=headers,
                    timeout=ADD_TIMEOUT,
                    json={"file_id": file_id},
                )
            except APIError as e:
                if e.status_code == 400 and "duplicate" in e.body.lower():
                    # Already in the knowledge base — treat as success, stop retrying
                    print(f"OK (duplicate — already indexed, id={file_id})")
                    uploaded[key] = file_id
                    ok += 1
                    continue
                if e.status_code == 400 and "empty" in e.body.lower():
                    # OpenWebUI couldn't extract text — try locally and re-upload as .txt
                    print(
                        f"(OpenWebUI parse failed, trying local extraction...)",
                        end=" ",
                        flush=True,
                    )
                    text = extract_text_locally(path)
                    if text and text.strip():
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                suffix=".txt", delete=False, mode="w", encoding="utf-8"
                            ) as tmp:
                                tmp.write(text)
                                tmp_path = tmp.name
                            with open(tmp_path, "rb") as fh:
                                upload2 = api(
                                    session,
                                    "POST",
                                    "/api/v1/files/",
                                    headers=headers,
                                    timeout=UPLOAD_TIMEOUT,
                                    files={
                                        "file": (path.stem + ".txt", fh, "text/plain")
                                    },
                                )
                            if upload2 and upload2.get("id"):
                                file_id2 = upload2["id"]
                                try:
                                    result2 = api(
                                        session,
                                        "POST",
                                        f"/api/v1/knowledge/{kb_id}/file/add",
                                        headers=headers,
                                        timeout=ADD_TIMEOUT,
                                        json={"file_id": file_id2},
                                    )
                                except APIError as e2:
                                    if (
                                        e2.status_code == 400
                                        and "empty" in e2.body.lower()
                                    ):
                                        print(
                                            f"SKIPPED (OpenWebUI still sees empty after local extraction)"
                                        )
                                        uploaded[key] = "SKIPPED:empty_content"
                                        skipped_empty += 1
                                    else:
                                        print(
                                            f"FAILED (fallback knowledge add HTTP {e2.status_code})"
                                        )
                                        errors += 1
                                        uploaded[key] = (
                                            f"ERROR:fallback_knowledge_add_{e2.status_code}"
                                        )
                                    result2 = None
                                if result2 is not None:
                                    print(f"OK via local extraction (id={file_id2})")
                                    uploaded[key] = file_id2
                                    ok += 1
                                    continue
                        except Exception as ex:
                            print(f"  [fallback upload error: {ex}]")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        if key not in uploaded:
                            print(f"FAILED (local extraction ok but re-upload failed)")
                            errors += 1
                            uploaded[key] = "ERROR:fallback_upload_failed"
                    else:
                        print(f"SKIPPED (truly empty or unreadable)")
                        uploaded[key] = "SKIPPED:empty_content"  # never retry
                        skipped_empty += 1
                else:
                    print(f"KNOWLEDGE ADD FAILED HTTP {e.status_code}: {e.body[:120]}")
                    errors += 1
                    uploaded[key] = f"ERROR:knowledge_add_{e.status_code}"
                continue

            if result is not None:
                print(f"OK (id={file_id})")
                uploaded[key] = file_id
                ok += 1
            else:
                print(f"KNOWLEDGE ADD FAILED (transient, id={file_id})")
                errors += 1
                uploaded[key] = "ERROR:knowledge_add_transient"

        except FileNotFoundError:
            print("FILE NOT FOUND (skipping)")
            errors += 1
            uploaded[key] = "ERROR:file_not_found"
        except PermissionError:
            print("PERMISSION DENIED (skipping)")
            errors += 1
            uploaded[key] = "ERROR:permission_denied"
        except APIError as e:
            print(f"UPLOAD FAILED HTTP {e.status_code}: {e.body[:120]}")
            errors += 1
            uploaded[key] = f"ERROR:api_{e.status_code}"
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            errors += 1
            uploaded[key] = f"ERROR:{type(e).__name__}"

        # Save after every file so a hard freeze never loses progress
        save_tracking(uploaded)

    # Final save
    save_tracking(uploaded)

    status = "Interrupted" if _shutdown else "Done"
    print(
        f"\n{status}. Uploaded: {ok}, Skipped (no text): {skipped_empty}, Errors: {errors}, Total tracked: {len(uploaded)}"
    )
    print(
        f"Open OpenWebUI → Workspace → Knowledge → '{args.knowledge_name}' to use it."
    )


if __name__ == "__main__":
    main()
