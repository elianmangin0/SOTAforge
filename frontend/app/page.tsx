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
  const [email, setEmail] = useState("");
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

    const e_mail = email.trim();
    
    // Validate email only if provided
    if (e_mail) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(e_mail)) {
        setError("Please enter a valid email address.");
        return;
      }
    }

    try {
      setLoading(true);
      const data = await requestSota(apiBase, t, e_mail || undefined);
      setTaskId(data.task_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate SOTA.");
      setLoading(false);
    }
  }

  // Calculate progress percentage
  const getProgressPercentage = () => {
    if (progressEvents.length === 0) return 0;
    
    const lastEvent = progressEvents[progressEvents.length - 1];
    const totalEvents = progressEvents.length;
    
    // If completed, return 100%
    if (lastEvent.status === "completed") return 100;
    
    // Simple linear progression based on event count
    // SOTA generations typically have 30-80 events
    // Each event adds ~1.2% progress, capped at 90% until completion
    
    if (lastEvent.status === "sending_email") {
      return 95;
    }
    
    // Calculate based on total events (cap at 90% until sending_email)
    const estimatedProgress = Math.min(5 + (totalEvents * 2), 90);
    return Math.round(estimatedProgress);
  };

  const stopGeneration = async () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    // Call backend to cancel the task
    if (taskId) {
      try {
        await fetch(`${apiBase}/api/sota/${taskId}`, {
          method: "DELETE",
        });
      } catch (err) {
        console.error("Error cancelling task:", err);
      }
    }
    
    setLoading(false);
    setTaskId(null);
    setProgressEvents([]);
    setError("Generation stopped by user.");
  };

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">SOTAforge</h1>
        <p className="subtitle">
          Generate a concise State-of-the-Art summary for any research topic.
        </p>
        <p className="subtitle" style={{ fontSize: "0.9rem", marginTop: "0.5rem", opacity: 0.8 }}>
        Generation typically takes 10 minutes to 1 hour depending on topic complexity.
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
            disabled={loading}
          />
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button className="button" type="submit" disabled={loading} style={{ flex: 1 }}>
              {loading ? "Generating…" : "Generate SOTA"}
            </button>
            {loading && (
              <button 
                className="button" 
                type="button" 
                onClick={stopGeneration}
                style={{ 
                  flex: "0 0 auto",
                  backgroundColor: "#dc2626",
                  borderColor: "#dc2626"
                }}
              >
                Stop
              </button>
            )}
          </div>
        </form>
        {error && <div className="status error">{error}</div>}
        
        {progressEvents.length > 0 && !result && (
          <div style={{ marginTop: "1.5rem" }}>
            <div style={{ 
              width: "100%", 
              height: "8px", 
              backgroundColor: "#2a2a2a", 
              borderRadius: "4px",
              overflow: "hidden",
              marginBottom: "1rem"
            }}>
              <div style={{
                width: `${getProgressPercentage()}%`,
                height: "100%",
                backgroundColor: "#10b981",
                transition: "width 0.3s ease",
                boxShadow: "0 0 10px rgba(16, 185, 129, 0.5)"
              }} />
            </div>
            <p style={{ 
              fontSize: "0.9rem", 
              color: "#10b981",
              textAlign: "center",
              marginBottom: "1rem"
            }}>
              {getProgressPercentage()}% Complete
            </p>
          </div>
        )}
        
        {loading && progressEvents.length > 0 && (
          <ProgressPanel events={progressEvents} isActive={loading} />
        )}

        {result && (
          <ResultPanel result={result} />
        )}
      </section>

      {!loading && (
        <section className="card" style={{ marginTop: "1.5rem" }}>
          <h3 style={{ 
            fontSize: "1rem", 
            marginBottom: "0.75rem",
            color: "#10b981"
          }}>
            Optional Email Delivery
          </h3>
          <p style={{ 
            fontSize: "0.85rem", 
            marginBottom: "1rem",
            opacity: 0.8,
            lineHeight: "1.5"
          }}>
            Provide your email to receive the SOTA summary as PDF and Markdown files. 
            You&apos;ll still see the results here instantly - email is just a convenient way to save them.
          </p>
          <input
            className="input"
            type="email"
            placeholder="your.email@example.com (optional)"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ marginBottom: 0 }}
          />
        </section>
      )}
    </div>
  );
}
