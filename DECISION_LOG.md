# DECISION_LOG — Enterprise RAG Assistant

Registro de decisiones técnicas del prototipo.
Formato cercano a ADR: contexto → decisión → alternativas → consecuencias.

---

## 1. ¿Por qué elegiste el modelo/LLM utilizado?

**Decisión:** Ollama Cloud (`gpt-oss:20b`) con rotación de API keys.

**Motivos:**
- Disponibilidad práctica: se contaba con API keys de Ollama Cloud y no con
  credenciales de otros proveedores (p. ej. Anthropic Claude / OpenAI).
- No se dispone de GPU local suficiente para servir un LLM grande en máquina propia;
  la nube evita ese costo de infraestructura para el prototipo.
- Rotación ante HTTP 401/403/429/500 para mitigar límites de cupo durante demos.
- Credenciales solo en `.env` (nunca en el código fuente).

**Alternativas consideradas:** APIs comerciales (Anthropic/OpenAI) si hubiera keys;
**SLM locales** (p. ej. Phi, Gemma, Llama 8B) si hubiera hardware suficiente.
No se privilegió un SLM concreto: son ejemplos típicos de modelos más pequeños
para inferencia local con menor costo computacional que un LLM grande.
La arquitectura desacopla el proveedor (`LLMProvider`), así que cambiar de
modelo no reescribe el flujo RAG.

---

## 2. ¿Por qué elegiste la estrategia de chunking / extracción PDF?

**Decisión extracción:** `pymupdf4llm.to_markdown` (PDF → Markdown) y, sobre ese
texto, recursive character splitter: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=160` (~20%),
con cortes preferentes en párrafos / headings Markdown.

**Cómo funciona (no es “párrafo fijo”):**
- Se toma una ventana de hasta ~800 caracteres; si hay `\n\n`, heading o `. `
  cerca del final, se corta ahí (puede quedar un poco menor a 800).
- El siguiente chunk **no** añade 160 antes y 160 después: arranca
  `160` caracteres **antes del fin** del chunk anterior (`start = end - overlap`),
  de modo que los últimos ~160 chars se repiten al inicio del siguiente.
- Así una frase partida entre dos trozos sigue apareciendo completa en al menos uno.

**Motivos (experiencia práctica en pipelines RAG):**
- Los loaders PDF “por defecto” suelen degradar tablas y estructura; convertir a
  Markdown y trabajar sobre MD reduce ruido para el LLM y mejora la recuperación.
- Hay que chunkear porque los modelos de embedding no manejan bien documentos
  enteros: necesitan fragmentos con semántica coherente.
- Valores típicos de laboratorio: p. ej. 500/100 (~20%). Aquí 800/160 mantiene
  esa proporción (~20%) con trozos un poco más largos (también es habitual usar
  hasta ~1000). No hay bala de plata; 20% de overlap equilibra continuidad vs.
  índice más grande.
- El tamaño se mantiene moderado para no saturar el embedder ni el contexto del
  LLM al recuperar `top_k` fragmentos. Para el prototipo bastan PDFs acotados
  (p. ej. decenas de páginas / pocos MB), no libros enormes.
- Se persiste el `.md` junto al upload para inspección y resaltado de citas en UI.

**Alternativas:** solo `pypdf` (más simple, peor estructura), Unstructured,
LlamaParse; chunking semántico (mejor calidad posible, más costo/complejidad).

## 2b. ¿Por qué no embeddings vía Ollama Cloud ni sentence-transformers?

**Decisión actual:** FastEmbed local (ver §3).

Ollama Cloud se descartó para embeddings tras probar la API (`/api/embed` → 401;
`/api/embeddings` → 404; el catálogo solo lista modelos de chat).
`sentence-transformers` / HuggingFace queda como alternativa válida si se acepta PyTorch.

---

## 3. ¿Por qué elegiste la solución de búsqueda/vectorial?

**Decisión:** Chroma + **FastEmbed** local (ONNX/CPU), modelo
`paraphrase-multilingual-MiniLM-L12-v2`.

**Motivos:**
- Ollama Cloud no expuso modelos de embedding en las pruebas realizadas.
- FastEmbed evita PyTorch/CUDA y no consume tokens de chat al indexar PDFs.
- Modelo multilingüe (ES/EN y otros) razonable para el prototipo.
- La generación de respuestas sigue en Ollama Cloud (`gpt-oss:20b`) con rotación de keys.

**Alternativas:** HuggingFace/`sentence-transformers` (trae torch),
Ollama embed si el plan lo habilita, FAISS/Qdrant.

**Nota ops:** en entornos con sqlite del sistema &lt; 3.35, Chroma requiere
`pysqlite3-binary` + parche en `app/core/sqlite_patch.py`.

---

## 4. ¿Cómo definiste el proceso de retrieval?

1. Embed de la pregunta.
2. `top_k=7` vecinos en Chroma (valor habitual en ejercicios RAG; un `k` un poco
   mayor ayuda a preguntas que combinan varios fragmentos).
3. Filtro por `MIN_RELEVANCE` (default 0.25).
4. Reindexación de citas `[1]…[n]` solo sobre chunks que pasan el umbral.
5. Prompt con bloque de contexto etiquetado + generación (`temperature` baja).
6. Persistencia Q/A en SQLite.

**Nota:** En práctica docente/profesional de RAG suele usarse un `k` del orden
de 5–7; se eligió **7** para dar más margen de contexto sin disparar demasiado
el tamaño del prompt.

---

## 5. ¿Cómo intentaste reducir alucinaciones?

- System prompt: responder solo con el contexto recuperado.
- Señal explícita `INSUFFICIENT_CONTEXT` cuando no hay base suficiente.
- Umbral de score: si ningún chunk es relevante → respuesta de insuficiencia sin inventar.
- **`LLM_TEMPERATURE=0.2`**: en tareas factuales (responder solo con evidencia del
  documento), la experiencia en LLM/RAG indica bajar la temperatura — valores bajos
  hacen al modelo más “enfático” e inventan menos; subirla aumenta creatividad y
  riesgo de alucinación. Se usa 0.2 (baja, no creativa) en lugar de forzar 0.0.
  El parámetro se envía a la API, no solo se menciona en el prompt.
- Evaluación manual con preguntas *out_of_document* en `evaluation/eval_set.json`.

---

## 6. ¿Cómo implementaste o controlaste el grounding?

**Grounding** = la respuesta debe anclarse al contexto recuperado, no al conocimiento paramétrico del LLM.

Mecanismos:
- Contexto numerado inyectado en el prompt.
- Flag `grounded` / `insufficient_context` en la API.
- Citas `[n]` ligadas a `chunk_id` reales emitidos por el backend (el LLM no inventa IDs).

---

## 7. ¿Cómo se obtienen y muestran las fuentes?

**Backend**
1. Asigna `citation_index` estable a cada chunk recuperado.
2. El LLM escribe citas `[n]` en el texto.
3. El backend parsea `[n]`, mapea a `RetrievedChunk` y devuelve:
   - `answer_with_citations`
   - `sources[]` con `chunk_id`, `document_id`, `filename`, `page`, `text`, `score`

**Frontend**
- Badges clicables en el chat.
- Panel lateral abre el documento y resalta el chunk citado.

Esto implementa la trazabilidad de fuentes en la respuesta (requisito 4.3).

---

## 8. ¿Qué ocurriría si el documento no contiene la respuesta?

- Retrieval vacío o bajo umbral → mensaje explícito de insuficiencia, `insufficient_context=true`.
- Si hay chunks pero el LLM declara `INSUFFICIENT_CONTEXT` → mismo tratamiento.
- No se inventan hechos; se documenta en el eval set (preguntas out-of-document).

---

## 9. ¿Cómo soportarías múltiples consultas simultáneas?

Cuellos de botella: modelo de embeddings, LLM remoto, escritura Chroma/SQLite.

Evolución a producción:
- Pool/cola para llamadas LLM (rate limit + backoff).
- Workers async para ingesta de PDFs.
- Vector DB servidor (Qdrant) y Postgres en lugar de SQLite.
- Cache de embeddings de queries frecuentes.
- Timeouts + reintentos idempotentes en el proveedor LLM (ya hay rotación de keys).
- Escalar horizontalmente API y workers; el store vectorial como servicio compartido.

---

## 10. ¿Qué cambiarías para llevar esta solución a producción?

- AuthN/AuthZ, tenant isolation.
- Observabilidad (latencia retrieval/LLM, tokens, hit-rate de citas).
- Hybrid search + reranker (funcionalidad opcional).
- Evaluación continua con métricas tipo faithfulness / context recall (p. ej. Ragas).
- OCR para PDFs escaneados.
- Secrets en vault; nunca en repo.

---

## 11. ¿Qué funcionalidades decidiste no implementar por tiempo?

- Hybrid search / reranking / streaming.
- Evaluación automatizada con framework de métricas (sí hay dataset manual).
- OCR.
- Multi-sesión avanzada en UI (sí hay `session_id` + endpoint history).

---

## 12. Persistencia del historial

**SQLite (SQLAlchemy)** para mensajes de sesión.
No se usa la vector DB para historial: el historial es relacional (sesión, rol, texto, sources JSON). Chroma solo guarda embeddings de chunks.

---

## 13. Desacoplamiento de proveedores (req. §9)

Puertos (`LLMProvider`, `EmbeddingProvider`, `VectorStore`) + implementaciones concretas.
La capa `services/rag.py` no importa clientes de Ollama/Chroma directamente.
Cambiar a otro LLM o a Qdrant implica una nueva implementación del puerto, sin reescribir el flujo RAG.
