"use client";
import { useState } from "react";
import "./globals.css";
import { requestSota } from "../lib/api";
import ResultPanel from "../components/ResultPanel";

export default function Page() {
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ topic: string; status: string; text: string } | null>(null);

  const apiBase = process.env.NEXT_PUBLIC_SOTAFORGE_API_URL || "http://localhost:8000";

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const t = topic.trim();
    if (!t) {
      setError("Please enter a topic.");
      return;
    }

    try {
      setLoading(true);
      const data = await requestSota(apiBase, t);
      if (data.status !== "completed") {
        setError("Unexpected response status: " + data.status);
      } else {
        setResult(data.result);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate SOTA.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">SOTAforge</h1>
        <p className="subtitle">
          Generate a concise State-of-the-Art summary for any research topic.
        </p>
        <p style={{ marginTop: "0.5rem" }}>
          Source: <a className="link" href="https://github.com/elianmangin0/SOTAforge" target="_blank" rel="noreferrer">GitHub – SOTAforge</a>
        </p>
      </header>

      <section className="card">
        <form className="form" onSubmit={onSubmit}>
          <input
            className="input"
            placeholder="e.g. Diffusion models for image generation"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
          <button className="button" type="submit" disabled={loading}>
            {loading ? "Generating…" : "Generate SOTA"}
          </button>
        </form>
        {error && <div className="status">{error}</div>}
        {!error && loading && (
          <div className="status">Working on your topic… this can take a minute.</div>
        )}

        {result && (
          <ResultPanel result={result} />
        )}
      </section>
    </div>
  );
}
