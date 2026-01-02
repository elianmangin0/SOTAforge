import ReactMarkdown from 'react-markdown';

interface SotaResult {
  topic: string;
  status: string;
  text: string;
}

type Props = { result: SotaResult };

export default function ResultPanel({ result }: Props) {
  return (
    <div className="result">
      <div className="result-text markdown-content">
        <ReactMarkdown>{result.text}</ReactMarkdown>
      </div>
    </div>
  );
}
