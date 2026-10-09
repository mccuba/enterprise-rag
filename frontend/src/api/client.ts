const BASE = import.meta.env.VITE_API_URL ?? "/api";

export type SourceCitation = {
  citation_index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  text: string;
  score: number;
};

export type AskResponse = {
  session_id: string;
  answer: string;
  answer_with_citations: string;
  grounded: boolean;
  insufficient_context: boolean;
  sources: SourceCitation[];
  used_chunk_ids: string[];
};

export type DocumentInfo = {
  id: string;
  filename: string;
  pages: number;
  chunk_count: number;
};

export type DocumentText = {
  id: string;
  filename: string;
  full_text: string;
  chunks: Array<{
    chunk_id: string;
    chunk_index: number;
    page: number | null;
    text: string;
  }>;
};

export async function listDocuments(): Promise<DocumentInfo[]> {
  const r = await fetch(`${BASE}/documents`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function uploadDocument(file: File): Promise<void> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/documents`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
}

export async function ask(
  question: string,
  sessionId: string,
): Promise<AskResponse> {
  const r = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getDocumentText(id: string): Promise<DocumentText> {
  const r = await fetch(`${BASE}/documents/${id}/text`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getHistory(sessionId: string) {
  const r = await fetch(`${BASE}/history/${sessionId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
