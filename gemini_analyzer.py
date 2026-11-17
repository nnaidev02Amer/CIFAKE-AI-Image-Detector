"""Gemini visual second opinion for the CIFAKE app.

Gemini is NOT a validated AI-image detector. It is asked for a cautious visual
assessment (POSSIBLY_AI_GENERATED / POSSIBLY_REAL / INCONCLUSIVE) with the evidence
it can see. Its accuracy on this task has not been measured in this project.

Design rules (B.Tech academic prototype):
  * The API key comes from the environment (GEMINI_API_KEY, optionally loaded from a
    local .env) or from Streamlit secrets. It is never logged, shown or returned.
  * The full-resolution image is sent - NOT the CNN's 32x32 copy - but only after all
    metadata (EXIF incl. GPS, XMP, IPTC, comments, text chunks) has been removed.
  * Exactly one request per user click: no automatic retries.
  * Every failure becomes a GeminiResult with ok=False and a readable message;
    nothing in this module raises into the Streamlit page.
  * No Streamlit import here, so everything can be unit-tested with a fake client.

SDK: google-genai (>=3.4,<4), client.models.generate_content with inline image bytes
and a Pydantic response schema (structured JSON output).
"""
from __future__ import annotations

import importlib
import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError

from image_utils import ImageValidationError, load_image

logger = logging.getLogger("cifake.gemini")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_S = 60
PLACEHOLDER_KEYS = {"", "your-gemini-api-key-here", "your_api_key_here", "changeme"}
MAX_REQUEST_IMAGE_BYTES = 18 * 1024 * 1024   # Gemini inline limit is 20 MB per request
MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
ASSESSMENTS = ("POSSIBLY_AI_GENERATED", "POSSIBLY_REAL", "INCONCLUSIVE")

# Error kinds (shown in the UI and used by the tests)
MISSING_KEY, SDK_MISSING, INVALID_IMAGE = "missing_key", "sdk_missing", "invalid_image"
AUTH, QUOTA, TIMEOUT, NETWORK = "auth", "quota", "timeout", "network"
MODEL_UNAVAILABLE, SERVER, BLOCKED = "model_unavailable", "server", "blocked"
INVALID_RESPONSE, UNKNOWN = "invalid_response", "unknown"


# ── Configuration ───────────────────────────────────────────────────────────
def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def get_timeout_s() -> int:
    try:
        return max(5, int(os.environ.get("GEMINI_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def resolve_api_key(secrets: Mapping[str, Any] | None = None) -> str | None:
    """GEMINI_API_KEY from the environment first, then from Streamlit secrets.
    Placeholders (e.g. the value in .env.example) count as 'not configured'."""
    candidates = [os.environ.get("GEMINI_API_KEY")]
    if secrets is not None:
        try:
            candidates.append(secrets.get("GEMINI_API_KEY"))
        except Exception:  # noqa: BLE001 - a broken secrets file must not crash the app
            pass
    for value in candidates:
        key = str(value or "").strip().strip('"').strip("'")
        if key.lower() not in PLACEHOLDER_KEYS:
            return key
    return None


def sdk_available() -> bool:
    try:
        importlib.import_module("google.genai")
        return True
    except ImportError:
        return False


# ── Structured output schema (no confidence field on purpose) ──────────────────
class GeminiAssessment(BaseModel):
    assessment: Literal["POSSIBLY_AI_GENERATED", "POSSIBLY_REAL", "INCONCLUSIVE"] = Field(
        description="Cautious overall visual assessment. Use INCONCLUSIVE when uncertain.")
    explanation: str = Field(
        description="Two to four sentences explaining the assessment in plain language.")
    evidence: list[str] = Field(
        description="Specific, observable details in this image that support the assessment "
                    "(location and what is seen). Empty list if there is no clear evidence.")
    limitations: str = Field(
        description="Why this visual assessment may be wrong or incomplete for this image.")


SYSTEM_INSTRUCTION = """You give a cautious visual second opinion on whether an image may be \
AI-generated, for a university project demonstration. You are not a forensic tool.

Examine only what is visible in the image:
- lighting and shadow consistency (direction, softness, reflections);
- texture and fine detail (skin, hair, foliage, fabric, text, edges, noise);
- anatomical or structural irregularities (hands, teeth, eyes, ears, limbs, architecture, \
perspective, object geometry);
- repeated patterns, smeared or melted regions, unusual artifacts;
- possible signs of editing or manipulation (cloning, inconsistent blur, mismatched edges).

Rules:
- Visual artifacts are clues, not proof of AI generation. Say "possibly".
- A realistic appearance is not evidence that an image is authentic.
- Do not give confidence percentages or probabilities.
- Do not guess the image's source, author, generator or provenance without visible evidence.
- No metadata is provided; do not speculate about it.
- Very small, low-resolution, heavily compressed or ambiguous images: answer INCONCLUSIVE.
- If you are uncertain, answer INCONCLUSIVE.
- Only report evidence you can actually see in this image."""

USER_PROMPT = ("Assess this image following your instructions. Return the assessment, a brief "
               "explanation, the observable evidence and the limitations of your assessment.")


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    lossy_reencode: bool          # True only for rotated JPEGs (quality 95)
    note: str


@dataclass(frozen=True)
class GeminiResult:
    ok: bool
    model: str
    assessment: str | None = None
    explanation: str = ""
    evidence: tuple[str, ...] = ()
    limitations: str = ""
    error_kind: str | None = None
    message: str = ""
    image_note: str = ""
    seconds: float = 0.0


@dataclass(frozen=True)
class Comparison:
    state: Literal["AGREE", "DISAGREE", "UNAVAILABLE", "INCONCLUSIVE"]
    title: str
    message: str
    notes: tuple[str, ...] = field(default_factory=tuple)


# ── Metadata removal ──────────────────────────────────────────────────────────
def _strip_jpeg_segments(data: bytes) -> bytes:
    """Copy a JPEG while dropping metadata segments. Pixel data is NOT re-encoded.

    Kept: APP0 (JFIF), APP2 only if ICC colour profile, APP14 (Adobe colour transform)
    and all image-coding segments. Dropped: APP1 (EXIF incl. GPS, XMP), other APPn
    (IPTC/Photoshop, MPF, maker data), COM comments, and anything after the first
    End-Of-Image marker (e.g. appended secondary images or thumbnails).
    """
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out, i, n = [b"\xff\xd8"], 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        while i + 1 < n and data[i + 1] == 0xFF:      # fill bytes
            i += 1
        marker = data[i + 1]
        if marker == 0xD9:                              # End Of Image
            out.append(b"\xff\xd9")
            return b"".join(out)
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:    # stand-alone markers
            out.append(data[i:i + 2])
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        segment = data[i:i + 2 + length]
        if len(segment) != 2 + length or length < 2:
            raise ValueError("truncated JPEG segment")
        keep = True
        if marker == 0xFE:
            keep = False
        elif 0xE1 <= marker <= 0xEF:
            keep = ((marker == 0xE2 and segment[4:16] == b"ICC_PROFILE\x00")
                    or (marker == 0xEE and segment[4:9] == b"Adobe"))
        if keep:
            out.append(segment)
        i += 2 + length
        if marker == 0xDA:                              # Start Of Scan: copy coded data
            j = i
            while True:
                j = data.index(b"\xff", j)
                nxt = data[j + 1]
                if nxt == 0x00 or 0xD0 <= nxt <= 0xD7:  # stuffed byte / restart marker
                    j += 2
                elif nxt == 0xFF:
                    j += 1
                else:
                    break
            out.append(data[i:j])
            i = j
    raise ValueError("JPEG has no End-Of-Image marker")


def _has_metadata(img: Image.Image) -> bool:
    return bool(len(img.getexif()) or img.info.get("xmp") or img.info.get("comment")
                or any(isinstance(v, str) for v in img.info.values()))


def prepare_for_gemini(data: bytes) -> PreparedImage:
    """Validate the upload (same rules as the CNN path) and build a metadata-free copy.

    The caller's original bytes are not modified (bytes are immutable) and the CNN's
    own preprocessing is untouched.
      JPEG, no rotation flag -> metadata segments removed, pixels bit-identical
      JPEG with EXIF rotation -> rotated upright, re-encoded at quality 95 (documented)
      PNG                     -> re-saved as lossless PNG without metadata
      WEBP                    -> first frame re-saved as lossless WEBP without metadata
    Raises ImageValidationError with a user-friendly message.
    """
    load_image(data)                                    # existing validation: format, size, corruption
    with Image.open(io.BytesIO(data)) as src:
        fmt = src.format
        orientation = src.getexif().get(0x0112, 1)
        src.load()
        icc = src.info.get("icc_profile")
        if fmt == "JPEG" and orientation in (0, 1):
            try:
                out = _strip_jpeg_segments(data)
                with Image.open(io.BytesIO(out)) as chk:
                    chk.load()
                    if chk.size != src.size or _has_metadata(chk):
                        raise ValueError("verification failed")
                return PreparedImage(out, MIME_TYPES[fmt], src.width, src.height, False,
                                     "Metadata removed without re-encoding (pixels unchanged).")
            except (ValueError, IndexError, OSError):
                pass                                    # fall through to a clean re-encode
        upright = ImageOps.exif_transpose(src)
        upright.info = {}                               # never carry EXIF/XMP/comments over
        buf = io.BytesIO()
        extra = {"icc_profile": icc} if icc else {}
        if fmt == "PNG":
            upright.save(buf, "PNG", optimize=False, **extra)
            note = "Re-saved as lossless PNG without metadata (pixels unchanged)."
        elif fmt == "WEBP":
            upright.save(buf, "WEBP", lossless=True, quality=100, exact=True, **extra)
            note = "Re-saved as lossless WEBP without metadata (first frame only if animated)."
        else:
            if upright.mode not in ("RGB", "L", "CMYK"):
                upright = upright.convert("RGB")
            upright.save(buf, "JPEG", quality=95, **extra)
            note = ("Rotated upright using the EXIF orientation and re-encoded as JPEG "
                    "(quality 95) without metadata; slight compression change.")
        out = buf.getvalue()
        if len(out) > MAX_REQUEST_IMAGE_BYTES:
            raise ImageValidationError("The image is too large to send to Gemini after "
                                       "metadata removal. Try a smaller image.")
        return PreparedImage(out, MIME_TYPES[fmt], upright.width, upright.height,
                             fmt == "JPEG", note)


# ── Error handling ────────────────────────────────────────────────────────────
class _Blocked(Exception):
    """Gemini refused to answer (safety filters)."""


class _BadResponse(Exception):
    """Gemini answered, but the answer failed validation."""


_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def _redact(text: str, api_key: str | None) -> str:
    text = str(text)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", text)


def _classify(exc: Exception, model: str) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    text = f"{status} {getattr(exc, 'message', '') or ''} {exc}"
    names = {c.__name__ for c in type(exc).__mro__}
    if isinstance(code, int):
        if code == 429 or "RESOURCE_EXHAUSTED" in status:
            return QUOTA, ("Gemini quota or rate limit reached for this API key. Wait and try "
                           "again later, or check your quota in Google AI Studio. "
                           "The app does not retry automatically.")
        if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
            return AUTH, ("Gemini rejected the API key (invalid, expired or without "
                          "permission). Check GEMINI_API_KEY in your local .env file.")
        if code == 404:
            return MODEL_UNAVAILABLE, (f"Model '{model}' is not available for this API key. "
                                       "Set GEMINI_MODEL to a model currently listed in the "
                                       "Gemini API documentation.")
        if code in (408, 504) or "DEADLINE_EXCEEDED" in status:
            return TIMEOUT, "Gemini did not respond in time. Try again later."
        if code >= 500:
            return SERVER, "The Gemini service is temporarily unavailable. Try again later."
        if code == 400:
            return INVALID_IMAGE, "Gemini could not process this request (bad request)."
    if names & {"TimeoutException", "TimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return TIMEOUT, "Gemini did not respond in time. Try again later."
    if names & {"ConnectError", "NetworkError", "TransportError", "ConnectionError",
                "ClientConnectorError", "gaierror"}:
        return NETWORK, "Could not reach the Gemini API. Check your internet connection."
    return UNKNOWN, f"Unexpected error while contacting Gemini ({type(exc).__name__})."


def _default_client_factory(api_key: str, timeout_s: int):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=timeout_s * 1000,                                 # milliseconds
        retry_options=types.HttpRetryOptions(attempts=1)))        # no automatic retries


def _build_config():
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=GeminiAssessment)            # structured JSON output, validated again below


def _image_part(prepared: PreparedImage):
    from google.genai import types
    return types.Part.from_bytes(data=prepared.data, mime_type=prepared.mime_type)


def _parse(response) -> GeminiAssessment:
    """Validate the structured response; raises _Blocked / _BadResponse if unusable."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise _Blocked(str(feedback.block_reason))
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        reason = str(getattr(candidates[0], "finish_reason", "") or "")
        if any(r in reason for r in ("SAFETY", "BLOCKLIST", "PROHIBITED", "SPII", "IMAGE_SAFETY")):
            raise _Blocked(reason)
        if "MAX_TOKENS" in reason:
            raise _BadResponse("response was cut off")
    parsed = getattr(response, "parsed", None)
    try:
        if isinstance(parsed, GeminiAssessment):
            result = parsed
        elif isinstance(parsed, dict):
            result = GeminiAssessment.model_validate(parsed)
        else:
            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                raise _BadResponse("empty response")
            result = GeminiAssessment.model_validate_json(str(text))
    except ValidationError as exc:
        raise _BadResponse(f"response did not match the schema ({exc.error_count()} problem(s))") from exc
    explanation = result.explanation.strip()[:1500]
    if not explanation:
        raise _BadResponse("empty explanation")
    evidence = [e.strip()[:300] for e in result.evidence if e and e.strip()][:8]
    limitations = result.limitations.strip()[:1000] or "No limitations were stated."
    return GeminiAssessment(assessment=result.assessment, explanation=explanation,
                            evidence=evidence, limitations=limitations)


# ── Main entry point ──────────────────────────────────────────────────────────
def analyze_image(data: bytes, api_key: str | None, model: str | None = None,
                  client_factory: Callable[[str, int], Any] | None = None) -> GeminiResult:
    """One user-triggered Gemini analysis. Never raises; never retries."""
    model = model or get_model_name()
    if not api_key:
        return GeminiResult(False, model, error_kind=MISSING_KEY,
                            message="Gemini is not configured: GEMINI_API_KEY is not set. "
                                    "The CNN prediction still works.")
    if not sdk_available():
        return GeminiResult(False, model, error_kind=SDK_MISSING,
                            message="The google-genai package is not installed "
                                    "(pip install -r requirements.txt).")
    try:
        prepared = prepare_for_gemini(data)
    except ImageValidationError as exc:
        return GeminiResult(False, model, error_kind=INVALID_IMAGE, message=str(exc))

    t0 = time.perf_counter()
    stage = "connect"
    try:
        client = (client_factory or _default_client_factory)(api_key, get_timeout_s())
        stage = "model check"
        info = client.models.get(model=model)                   # verify model before analysing
        actions = getattr(info, "supported_actions", None)
        if actions and "generateContent" not in actions:
            return GeminiResult(False, model, error_kind=MODEL_UNAVAILABLE,
                                message=f"Model '{model}' does not support image analysis "
                                        "(generateContent). Set GEMINI_MODEL to another model.")
        stage = "analysis"
        response = client.models.generate_content(
            model=model, contents=[_image_part(prepared), USER_PROMPT], config=_build_config())
        parsed = _parse(response)
    except _Blocked as exc:
        logger.info("gemini blocked model=%s reason=%s", model, _redact(exc, api_key))
        return GeminiResult(False, model, error_kind=BLOCKED, image_note=prepared.note,
                            message="Gemini declined to analyse this image "
                                    f"({_redact(exc, api_key)}).")
    except _BadResponse as exc:
        logger.info("gemini invalid response model=%s reason=%s", model, _redact(exc, api_key))
        return GeminiResult(False, model, error_kind=INVALID_RESPONSE, image_note=prepared.note,
                            message="Gemini returned a response that could not be validated "
                                    f"({_redact(exc, api_key)}). No result is shown.")
    except Exception as exc:  # noqa: BLE001 - network/API errors become messages
        kind, message = _classify(exc, model)
        logger.warning("gemini %s failed at %s: kind=%s type=%s", model, stage, kind,
                       type(exc).__name__)
        return GeminiResult(False, model, error_kind=kind, message=_redact(message, api_key),
                            image_note=prepared.note)

    seconds = time.perf_counter() - t0
    logger.info("gemini ok model=%s mime=%s bytes=%d assessment=%s %.1fs", model,
                prepared.mime_type, len(prepared.data), parsed.assessment, seconds)
    return GeminiResult(True, model, assessment=parsed.assessment,
                        explanation=parsed.explanation, evidence=tuple(parsed.evidence),
                        limitations=parsed.limitations, image_note=prepared.note,
                        seconds=seconds)


# ── CNN vs Gemini comparison (no combined score) ──────────────────────────────
CNN_TO_GEMINI = {"FAKE": "POSSIBLY_AI_GENERATED", "REAL": "POSSIBLY_REAL"}


def compare(cnn_label: str | None, gemini: GeminiResult | None,
            cnn_confidence: float | None = None) -> Comparison:
    notes = []
    if cnn_confidence is not None and cnn_confidence < 0.70:
        notes.append("The CNN's own confidence is low for this image.")
    if cnn_label not in CNN_TO_GEMINI or gemini is None or not gemini.ok:
        missing = "The CNN" if cnn_label not in CNN_TO_GEMINI else "Gemini"
        return Comparison("UNAVAILABLE", "One system unavailable",
                          f"{missing} result is not available, so no comparison is made.",
                          tuple(notes))
    if gemini.assessment == "INCONCLUSIVE":
        return Comparison("INCONCLUSIVE", "Inconclusive",
                          "Gemini could not reach a visual assessment, so the two systems "
                          "cannot be compared for this image.", tuple(notes))
    if CNN_TO_GEMINI[cnn_label] == gemini.assessment:
        return Comparison("AGREE", "Both systems agree",
                          "The CNN and Gemini point the same way. This is two weak signals "
                          "agreeing, not proof of how the image was made.", tuple(notes))
    return Comparison("DISAGREE", "Systems disagree",
                      "The CNN and Gemini point in different directions. Treat the image as "
                      "uncertain.", tuple(notes))
