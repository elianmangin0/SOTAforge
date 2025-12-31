import { useEffect, useRef } from "react";

interface ProgressEvent {
  status: string;
  message: string;
  step?: string;
  timestamp: string;
}

interface ProgressPanelProps {
  events: ProgressEvent[];
  isActive: boolean;
}

export default function ProgressPanel({ events, isActive }: ProgressPanelProps) {
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [events]);

  const getStepLabel = (step?: string) => {
    if (!step) return "";
    const labels: Record<string, string> = {
      search: "SEARCH",
      filter: "FILTER",
      parser: "PARSE",
      analyzer: "ANALYZE",
      synthesizer: "SYNTHESIZE",
      db: "DATABASE",
      initializing: "INIT",
      loading: "LOAD",
      running: "RUN"
    };
    return labels[step] || step.toUpperCase();
  };

  return (
    <div className="progress-panel">
      <div className="progress-header">
        <h3>Pipeline Execution Log</h3>
        {isActive && <div className="pulse-dot"></div>}
      </div>
      
      <div className="log-container">
        {events.map((event, idx) => (
          <div key={idx} className="log-line">
            {event.step && (
              <span className="log-step">[{getStepLabel(event.step)}]</span>
            )}
            <span className="log-message">{event.message}</span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
      
      {isActive && (
        <div className="log-footer">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <span className="log-status">Processing...</span>
        </div>
      )}
    </div>
  );
}
