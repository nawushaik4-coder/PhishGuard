
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadHistory() {
    try {
      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error("Could not load scan history.");
      }

      const data = await response.json();
      setHistory(data.history ?? []);
    } catch {
      setError(
        "Cannot connect to the backend or load history. Check that FastAPI and MySQL are running."
      );
    }
  }

  async function scanUrl(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/scan`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: url.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail
            ? JSON.stringify(data.detail)
            : "The scan request failed."
        );
      }

      setResult(data);
      await loadHistory();
    } catch (err) {
      setError(
        err.message ||
          "Unable to scan. Check that the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  const score = result?.risk_score ?? 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">P</span>
          <span>Phish<span className="brand-accent">Guard</span></span>
        </a>

        <span className="status-pill">
          <span className="status-dot" />
          URL Protection
        </span>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">
            <span>SECURITY</span>
            <span className="eyebrow-line" />
            <span>URL ANALYSIS</span>
          </div>

          <h1>
            Don't get <span className="gradient-text">hooked.</span>
            <br />
            Check before you click.
          </h1>

          <p className="hero-description">
            Analyse suspicious URLs with transparent security rules.
            Review risk indicators and keep track of previous scans.
          </p>
        </section>

        <section className="scanner-card">
          <div className="section-heading">
            <div className="heading-icon">⌕</div>
            <div>
              <h2>URL Scanner</h2>
              <p>Enter a URL to analyse its characteristics</p>
            </div>
          </div>

          <form onSubmit={scanUrl}>
            <label htmlFor="url-input">Website URL</label>
            <div className="input-row">
              <span className="input-icon">↗</span>
              <input
                id="url-input"
                type="text"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com"
                maxLength={2048}
                required
              />
              <button type="submit" disabled={loading}>
                {loading ? "Scanning..." : "Scan URL"}
                {!loading && <span> →</span>}
              </button>
            </div>
            <p className="input-hint">
              <span>ⓘ</span> The URL is analysed as text. PhishGuard
              does not visit the submitted website.
            </p>
          </form>

          {error && (
            <div className="error-message" role="alert">
              <strong>Something went wrong</strong>
              <p>{error}</p>
            </div>
          )}
        </section>

        {result && (
          <section className="result-card">
            <div className="result-top">
              <div>
                <span className="section-kicker">SCAN RESULT</span>
                <h2>{result.verdict}</h2>
                <p className="result-url">{result.url}</p>
              </div>

              <div className="score-circle">
                <strong>{score}</strong>
                <span>/ 100</span>
              </div>
            </div>

            <div className="score-track">
              <div
                className={`score-fill ${
                  score >= 50
                    ? "score-high"
                    : score >= 20
                    ? "score-medium"
                    : "score-low"
                }`}
                style={{ width: `${score}%` }}
              />
            </div>

            <h3>Indicators detected</h3>
            <ul className="reason-list">
              {result.reasons.map((reason, index) => (
                <li key={`${index}-${reason}`}>
                  <span className="reason-marker">!</span>
                  {reason}
                </li>
              ))}
            </ul>

            <p className="disclaimer">
              {result.notice ||
                "A low score does not prove that a URL is safe."}
            </p>
            <p className="scan-reference">
              Scan reference: #{result.scan_id}
            </p>
          </section>
        )}

        <section className="history-section">
          <div className="history-heading">
            <div>
              <span className="section-kicker">YOUR ACTIVITY</span>
              <h2>Recent scans</h2>
            </div>
            <button
              className="refresh-button"
              onClick={loadHistory}
              type="button"
            >
              ↻ Refresh
            </button>
          </div>

          <div className="history-card">
            {history.length === 0 ? (
              <div className="empty-state">
                <span className="empty-icon">◷</span>
                <h3>No scans yet</h3>
                <p>Your saved scan history will appear here.</p>
              </div>
            ) : (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>URL</th>
                      <th>Verdict</th>
                      <th>Score</th>
                      <th>Scan ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item) => (
                      <tr key={item.scan_id}>
                        <td className="history-url" title={item.url}>
                          {item.url}
                        </td>
                        <td>
                          <span className="verdict-label">
                            {item.verdict}
                          </span>
                        </td>
                        <td>{item.risk_score}/100</td>
                        <td>#{item.scan_id}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        <footer className="footer">
          <span>PhishGuard</span>
          <span>Explainable URL analysis · Student research project</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
