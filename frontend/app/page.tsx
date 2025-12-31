"use client";
import { useState, useEffect, useRef } from "react";
import "./globals.css";
import { requestSota } from "../lib/api";
import ResultPanel from "../components/ResultPanel";
import ProgressPanel from "../components/ProgressPanel";

interface ProgressEvent {
  status: string;
  message: string;
  step?: string;
  timestamp: string;
  result?: { topic: string; status: string; text: string };
}

export default function Page() {
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ topic: string; status: string; text: string } | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  const apiBase = process.env.NEXT_PUBLIC_SOTAFORGE_API_URL || "http://localhost:8000";

  // Listen to SSE stream for progress updates
  useEffect(() => {
    if (!taskId || !loading) {
      return;
    }

    const eventSource = new EventSource(`${apiBase}/api/sota/stream/${taskId}`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data);
        
        // Add event to progress list
        setProgressEvents((prev) => [...prev, data]);

        // Handle completion
        if (data.status === "completed" && data.result) {
          setResult(data.result);
          setLoading(false);
          setTaskId(null);
          eventSource.close();
        } else if (data.status === "failed") {
          setError(data.message || "Generation failed");
          setLoading(false);
          setTaskId(null);
          eventSource.close();
        }
      } catch (err) {
        console.error("Error parsing SSE data:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [taskId, loading, apiBase]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgressEvents([]);

    const t = topic.trim();
    if (!t) {
      setError("Please enter a topic.");
      return;
    }

    try {
      setLoading(true);
      const data = await requestSota(apiBase, t);
      setTaskId(data.task_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate SOTA.");
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
        {error && <div className="status error">{error}</div>}
        
        {loading && progressEvents.length > 0 && (
          <ProgressPanel events={progressEvents} isActive={loading} />
        )}

        {result && (
          <ResultPanel result={result} />
        )}
      </section>
    </div>
  );
}
