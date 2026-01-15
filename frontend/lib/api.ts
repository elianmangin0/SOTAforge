interface SotaResult {
  topic: string;
  status: string;
  text: string;
}

interface SOTAResponse {
  task_id: string;
}

interface SOTAStatusResponse {
  task_id: string;
  status: string;
  progress: string;
  result: SotaResult | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export async function requestSota(apiBase: string, topic: string, email?: string): Promise<SOTAResponse>
{
  const body: { topic: string; email?: string } = { topic };
  if (email) {
    body.email = email;
  }

  const res = await fetch(`${apiBase}/api/sota`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error (${res.status}): ${text}`);
  }

  return res.json();
}

export async function getSOTAStatus(apiBase: string, taskId: string): Promise<SOTAStatusResponse>
{
  const res = await fetch(`${apiBase}/api/sota/status/${taskId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error (${res.status}): ${text}`);
  }

  return res.json();
}
