# Enterprise RAG Assistant

Prototipo de asistente RAG (Retrieval-Augmented Generation) con:

- Ingesta de PDFs → **Markdown** (`pymupdf4llm`) → chunking → **FastEmbed** local → Chroma
- Chat LLM vía **Ollama Cloud** (embeddings locales con FastEmbed)
- Consulta `/ask` con **grounding** y rechazo explícito si no hay contexto
- **Trazabilidad de fuentes** con citas `[n]` y panel de documento
- Historial de sesión en **SQLite**
- Backend FastAPI + Frontend React (Vite)
- Docker Compose, pytest, Ruff

## Arquitectura

```
frontend (React) ──HTTP──► backend (FastAPI)
                              ├── services/rag.py      (orquestación)
                              ├── services/ingest.py
                              ├── providers/llm.py     (puerto → Ollama Cloud)
                              ├── providers/embeddings.py
                              ├── providers/vector_store.py (puerto → Chroma)
                              └── db/ (SQLite: docs, chunks, messages)
```

Ver decisiones detalladas en [`DECISION_LOG.md`](./DECISION_LOG.md).

## Documentos de conocimiento

Corpus de papers públicos de ML/física generativa (todos ≥10 páginas), usados para demostrar
ingesta, retrieval, grounding y trazabilidad. Archivos locales de referencia en
`PostDoct/PAPERS/`; fuentes públicas (arXiv) abajo.

| Título | Autores (resumen) | Fuente pública | Págs. | Descripción breve |
|--------|-------------------|----------------|-------|-------------------|
| Active Flow Expansion for Out-of-Distribution Discovery: from Theory to Molecules | De Santi et al. | [arXiv:2606.08802](https://arxiv.org/pdf/2606.08802) | 43 | Expansión activa de flujos para descubrimiento OOD; de teoría a moléculas. |
| Carré du champ flow matching: better quality-generalisation tradeoff in generative models | Bamberger et al. | [arXiv:2510.05930](https://arxiv.org/pdf/2510.05930) | 24 | Flow matching con *carré du champ*; trade-off calidad vs generalización. |
| Conservation Laws for Diffusion Models | Aharoni & Pfister | [arXiv:2607.10067](https://arxiv.org/pdf/2607.10067) | 23 | Leyes de conservación aplicadas al análisis de modelos de difusión. |
| DiffTaichi: Differentiable Programming for Physical Simulation | Hu et al. (ICLR 2020) | [arXiv:1910.00935](https://arxiv.org/pdf/1910.00935) | 20 | Programación diferenciable para simulación física (DiffTaichi). |
| Flow Matching Meets PDEs: A Unified Framework for Physics-Constrained Generation | Baldan, Liu, Guardone, Thuerey | Preprint (archivo `FLOW MATCHING MEETS PDES.pdf`) | 24 | Marco unificado que combina flow matching con restricciones de EDPs. |
| Flow Matching for Generative Modeling | Lipman et al. (FAIR / Weizmann) | [arXiv:2210.02747](https://arxiv.org/pdf/2210.02747) | 28 | Paper fundacional de *flow matching* para modelado generativo. |
| Physics-Informed Diffusion Models | Bastek, Sun, Kochmann (ICLR 2025) | [arXiv:2403.14404](https://arxiv.org/pdf/2403.14404) | 26 | Difusión informada por física (PINNs + diffusion). |

Dominio elegido: **IA generativa / flow matching / física-ML**, con suficiente extensión para
consultas in-document, multi-chunk y out-of-document.

## Requisitos

- Python 3.11+
- Node 20+
- Cuenta Ollama Cloud + tokens API
- (Opcional) Docker / Docker Compose

## Configuración

```bash
cd backend
cp .env.example .env
# Edita OLLAMA_TOKENS=token1,token2,...
```

**Importante:** no subas `.env` ni keys al repositorio.

## Ejecución local

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://127.0.0.1:5173 (proxy `/api` → backend).

## Docker Compose

```bash
# backend/.env debe existir con OLLAMA_TOKENS
docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:5173  

## API mínima

| Endpoint | Propósito |
|----------|-----------|
| `POST /documents` | Carga y procesa un PDF |
| `POST /ask` | Pregunta + `session_id` → respuesta grounded + fuentes |
| `GET /history/{session_id}` | Historial de la sesión |
| `GET /documents/{id}/text` | Texto/chunks para resaltar citas en el FE |

## Trazabilidad (resumen)

1. Retrieval asigna IDs `[1]…[n]` a chunks.
2. El LLM cita con esos números.
3. El backend mapea a `chunk_id` / página / texto.
4. El FE muestra badges; al hacer clic abre el panel y resalta el chunk.

## Evaluación

Ver [`evaluation/eval_set.json`](./evaluation/eval_set.json): ≥10 preguntas (in-doc, multi-chunk, out-of-doc).

## Tests y calidad

```bash
cd backend
pytest
ruff check app tests
```

## Estructura

```
enterprise-rag/
├── backend/          # FastAPI
├── frontend/         # React + Vite
├── evaluation/       # Dataset de evaluación
├── DECISION_LOG.md
├── docker-compose.yml
└── README.md
```
