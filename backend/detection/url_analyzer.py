
import ipaddress
import re
from urllib.parse import urlsplit


SUSPICIOUS_WORDS = {
    "login", "verify", "password", "account",
    "secure", "update", "wallet", "signin",
}


def analyse_url(url: str) -> dict:
    """Analyse a URL using explainable heuristic rules."""

    reasons = []
    score = 0

    # Validate the input before parsing.
    if not isinstance(url, str) or not url.strip():
        return {
            "verdict": "Invalid URL",
            "risk_score": 0,
            "reasons": ["Please enter a URL."],
        }

    url = url.strip()

    # Reject whitespace and control characters inside a URL.
    if any(char.isspace() or ord(char) < 32 for char in url):
        return {
            "verdict": "Invalid URL",
            "risk_score": 0,
            "reasons": [
                "The URL contains whitespace or control characters."
            ],
        }

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return {
            "verdict": "Invalid URL",
            "risk_score": 0,
            "reasons": ["The URL could not be parsed correctly."],
        }

    if (
        parsed.scheme.lower() not in ("http", "https")
        or not parsed.netloc
        or not hostname
    ):
        return {
            "verdict": "Invalid URL",
            "risk_score": 0,
            "reasons": [
                "Enter a complete URL starting with http:// or https://."
            ],
        }

    hostname = hostname.lower().rstrip(".")

    # Check whether the hostname is an IP address.
    try:
        ipaddress.ip_address(hostname)
        score += 25
        reasons.append("The hostname is an IP address.")
    except ValueError:
        pass

    # HTTP does not encrypt the connection like HTTPS.
    # However, HTTP alone does not prove a site is phishing.
    if parsed.scheme.lower() == "http":
        score += 10
        reasons.append("The URL uses HTTP rather than HTTPS.")

    # Detect user information before the actual hostname.
    if parsed.username is not None or parsed.password is not None:
        score += 25
        reasons.append(
            "The URL contains user information before the hostname."
        )

    if len(url) > 100:
        score += 10
        reasons.append("The URL is unusually long.")

    # Match suspicious terms as tokens instead of arbitrary substrings.
    url_text = hostname + parsed.path + " " + parsed.query
    tokens = set(
        re.findall(r"[a-z0-9]+", url_text.lower())
    )
    found_words = sorted(SUSPICIOUS_WORDS.intersection(tokens))

    if found_words:
        score += min(10 * len(found_words), 20)
        reasons.append(
            "Contains potentially sensitive keywords: "
            + ", ".join(found_words)
            + "."
        )

    # Multiple hostname components may warrant further inspection.
    hostname_parts = hostname.split(".")
    if len(hostname_parts) >= 5:
        score += 15
        reasons.append(
            "The hostname has many subdomain components."
        )

    # Punycode hostnames deserve inspection, but are not automatically
    # malicious; some legitimate international domains use them.
    if any(part.startswith("xn--") for part in hostname_parts):
        score += 10
        reasons.append(
            "The hostname contains a punycode-encoded component."
        )

    score = min(score, 100)

    if score >= 50:
        verdict = "High risk indicators"
    elif score >= 20:
        verdict = "Needs review"
    else:
        verdict = "Few risk indicators detected"

    if not reasons:
        reasons.append(
            "No indicators covered by the current rules were detected."
        )

    return {
        "verdict": verdict,
        "risk_score": score,
        "reasons": reasons,
    }
