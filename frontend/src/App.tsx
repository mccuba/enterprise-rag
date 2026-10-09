import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ask,
  DocumentInfo,
  DocumentText,
  getDocumentText,
  listDocuments,
  SourceCitation,
  uploadDocument,
} from "./api/client";

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
  insufficient?: boolean;
};

function sessionId(): string {
  const key = "rag_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

/** Renderiza texto con badges [n] clickeables. */
function CitedText({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: SourceCitation[];
  onCite: (s: SourceCitation) => void;
}) {
  const byIdx = useMemo(() => {
    const m = new Map<number, SourceCitation>();
    sources.forEach((s) => m.set(s.citation_index, s));
    return m;
  }, [sources]);

  const parts = text.split(/(\[\d+\])/g);
  return (
    <span className="cited-text">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (!m) return <span key={i}>{part}</span>;
        const n = Number(m[1]);
        const src = byIdx.get(n);
        return (
          <button
            key={i}
            type="button"
            className="cite-badge"
            title={src ? `${src.filename}${src.page ? ` · p.${src.page}` : ""}` : `Cita ${n}`}
            onClick={() => src && onCite(src)}
            disabled={!src}
          >
            {n}
          </button>
        );
      })}
    </span>
  );
}

export default function App() {
  const sid = useMemo(() => sessionId(), []);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [panelOpen, setPanelOpen] = useState(false);
  const [activeDoc, setActiveDoc] = useState<DocumentText | null>(null);
  const [highlightChunkId, setHighlightChunkId] = useState<string | null>(null);
  const highlightRef = useRef<HTMLElement | null>(null);

  async function refreshDocs() {
    try {
      setDocs(await listDocuments());
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    void refreshDocs();
  }, []);

  useEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightChunkId, activeDoc]);

  async function onUpload(files: FileList | null) {
    if (!files?.length) return;
    const list = Array.from(files);
    setUploading(true);
    setError(null);
    try {
      for (let i = 0; i < list.length; i++) {
        const f = list[i];
        setUploadStatus(
          `Indexando ${i + 1}/${list.length}: ${f.name} (PDF→MD + embeddings; PDFs grandes tardan)…`,
        );
        await uploadDocument(f);
        await refreshDocs();
      }
      setUploadStatus(null);
    } catch (e) {
      setError(String(e));
      setUploadStatus(null);
    } finally {
      setUploading(false);
    }
  }

  async function openCitation(src: SourceCitation) {
    setPanelOpen(true);
    setHighlightChunkId(src.chunk_id);
    if (!activeDoc || activeDoc.id !== src.document_id) {
      const text = await getDocumentText(src.document_id);
      setActiveDoc(text);
    }
  }

  async function onAsk(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    try {
      const res = await ask(q, sid);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer_with_citations,
          sources: res.sources,
          insufficient: res.insufficient_context,
        },
      ]);
    } catch (err) {
      setError(String(err));
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Error al consultar el asistente.", insufficient: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`shell ${panelOpen ? "shell--split" : ""}`}>
      <aside className="sources-rail">
        <header className="rail-header">
          <h1>Fuentes</h1>
          <label className="add-btn">
            {uploading ? "Indexando…" : "+ Añadir PDF"}
            <input
              type="file"
              accept="application/pdf"
              multiple
              hidden
              disabled={uploading}
              onChange={(e) => void onUpload(e.target.files)}
            />
          </label>
        </header>
        {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.id}>
              <button
                type="button"
                className={activeDoc?.id === d.id ? "doc active" : "doc"}
                onClick={() => {
                  void getDocumentText(d.id).then((t) => {
                    setActiveDoc(t);
                    setPanelOpen(true);
                    setHighlightChunkId(null);
                  });
                }}
              >
                <span className="pdf-ico">PDF</span>
                <span>
                  <strong>{d.filename}</strong>
                  <small>
                    {d.pages} pág. · {d.chunk_count} chunks
                  </small>
                </span>
              </button>
            </li>
          ))}
          {!docs.length && !uploadStatus && (
            <li className="empty">
              Carga 1–2 PDFs públicos de ~10–40 pág. (no libros de 500+).
            </li>
          )}
        </ul>
      </aside>

      {panelOpen && (
        <section className="source-panel">
          <header className="panel-header">
            <div>
              <h2>{activeDoc?.filename ?? "Documento"}</h2>
              <p>Haz clic en una cita del chat para resaltar el chunk.</p>
            </div>
            <button type="button" className="ghost" onClick={() => setPanelOpen(false)}>
              Cerrar
            </button>
          </header>
          <div className="panel-body">
            {activeDoc?.chunks.map((c) => (
              <article
                key={c.chunk_id}
                ref={(el) => {
                  if (c.chunk_id === highlightChunkId) highlightRef.current = el;
                }}
                className={
                  c.chunk_id === highlightChunkId ? "chunk chunk--hi" : "chunk"
                }
              >
                <div className="chunk-meta">
                  chunk {c.chunk_index}
                  {c.page != null ? ` · p.${c.page}` : ""}
                </div>
                <p>{c.text}</p>
              </article>
            ))}
          </div>
        </section>
      )}

      <main className="chat">
        <header className="chat-header">
          <div>
            <p className="brand">Enterprise RAG</p>
            <h2>Asistente con grounding</h2>
          </div>
          <span className="pill">{docs.length} fuentes</span>
        </header>

        <div className="messages">
          {!messages.length && (
            <div className="welcome">
              <p>
                Pregunta sobre tus PDFs. Las respuestas incluyen citas clicables
                que abren el fragmento fuente correspondiente.
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`bubble ${msg.role}${msg.insufficient ? " warn" : ""}`}
            >
              {msg.role === "assistant" && msg.sources ? (
                <CitedText
                  text={msg.content}
                  sources={msg.sources}
                  onCite={(s) => void openCitation(s)}
                />
              ) : (
                msg.content
              )}
            </div>
          ))}
          {loading && <div className="bubble assistant muted">Recuperando contexto…</div>}
        </div>

        {error && <div className="error">{error}</div>}

        <form className="composer" onSubmit={onAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Haz una pregunta o crea algo"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            Enviar
          </button>
        </form>
      </main>
    </div>
  );
}
