
from detection.url_analyzer import analyse_url


def test_normal_example_domain():
    result = analyse_url("https://example.com")

    assert result["verdict"] == "Few risk indicators detected"
    assert result["risk_score"] == 0


def test_http_url_gets_warning():
    result = analyse_url("http://example.com")

    assert result["risk_score"] == 10
    assert any("HTTP rather than HTTPS" in reason
               for reason in result["reasons"])


def test_ip_hostname_is_flagged():
    result = analyse_url("http://192.0.2.10/login")

    assert result["risk_score"] > 0
    assert any("IP address" in reason
               for reason in result["reasons"])


def test_invalid_scheme_is_rejected():
    result = analyse_url("ftp://example.com")

    assert result["verdict"] == "Invalid URL"


def test_missing_hostname_is_rejected():
    result = analyse_url("https:///login")

    assert result["verdict"] == "Invalid URL"


def test_score_never_exceeds_100():
    url = (
        "http://user:pass@192.0.2.10/"
        + "login-verify-password-account-secure-wallet-" * 5
    )
    result = analyse_url(url)

    assert 0 <= result["risk_score"] <= 100
