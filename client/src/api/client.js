const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export async function fetchGraphRoot() {
  const res = await fetch(`${API_BASE}/graph/root`);
  return res.json();
}

export async function expandNode(nodeId) {
  const res = await fetch(`${API_BASE}/graph/expand/${encodeURIComponent(nodeId)}`);
  return res.json();
}

export async function fetchNodeDetail(nodeId) {
  const res = await fetch(`${API_BASE}/graph/node/${encodeURIComponent(nodeId)}`);
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/graph/stats`);
  return res.json();
}

export async function fetchTopProducts(limit = 10) {
  const res = await fetch(`${API_BASE}/graph/products/top?limit=${limit}`);
  return res.json();
}

export async function fetchIncompleteOrders() {
  const res = await fetch(`${API_BASE}/graph/orders/incomplete`);
  return res.json();
}

export async function* streamChat(message, conversationHistory = []) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_history: conversationHistory }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data: ')) {
        try {
          yield JSON.parse(trimmed.slice(6));
        } catch { /* skip malformed */ }
      }
    }
  }
}
