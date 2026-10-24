"""Canonicalize full URLs for the URL-text model.

Unlike the previous hostname-only extractor, this retains the hostname,
path, query, and fragment. The training script and live predictor share this
normalization function so they see the same text representation.
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


def normalize_url_text(value: str, *, allow_bare_domain: bool = True) -> str:
    """Return normalized URL text without the scheme, preserving path/query.

    For dataset records without a scheme, a temporary http:// scheme is added
    only for parsing. The returned text excludes scheme to avoid making an
    input https:// URL differ from a schemeless dataset record just because of
    the protocol prefix.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("URL must be a non-empty string.")

    value = html.unescape(value).strip()
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("URL must not contain whitespace or control characters.")

    value = value.lower()
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        if not allow_bare_domain:
            raise ValueError("Enter a complete URL beginning with http:// or https://.")
        value = "http://" + value

    try:
        parsed = urlsplit(value)
        _ = parsed.port  # Forces invalid ports to raise ValueError.
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("URL could not be parsed.") from exc

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("Enter a complete URL beginning with http:// or https://.")

    # Retain path and query; these contain many useful lexical indicators.
    # Exclude protocol because urlset.csv commonly stores schemeless values.
    normalized = parsed.netloc.lower() + parsed.path.lower()
    if parsed.query:
        normalized += "?" + parsed.query.lower()
    if parsed.fragment:
        normalized += "#" + parsed.fragment.lower()

    if not normalized:
        raise ValueError("URL has no usable hostname.")
    return normalized


def extract_features(url: str) -> dict[str, str]:
    """Interface compatible with the current main.py.

    The character n-gram vectorizer transforms url_text inside predict.py.
    """
    return {"url_text": normalize_url_text(url, allow_bare_domain=False)}
