interface SotaResult {
  topic: string;
  status: string;
  text: string;
}

export async function requestSota(apiBase: string, topic: string): Promise<{ status: string; result: SotaResult; }>
{
  const res = await fetch(`${apiBase}/api/sota`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ topic }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error (${res.status}): ${text}`);
  }

  return res.json();
}
