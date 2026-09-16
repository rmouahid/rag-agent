# Chunk schema

`rag_agent.chunking.chunk.Chunk` is the shared contract between the
chunking layer, the vector store, and generation. Every chunking
strategy (fixed-size, sentence-aware, ...) must produce `Chunk`
instances; the vector store indexes them and generation reads them
back to build the answer context.

| Field      | Type   | Description |
|------------|--------|-------------|
| `id`       | `str`  | Deterministic id — SHA-256 of `source:position:text`, truncated to 16 hex chars. Same passage re-ingested from the same source at the same position always yields the same id, which chunk deduplication (#14) relies on. Auto-computed if not passed explicitly. |
| `text`     | `str`  | The chunk's text content. |
| `source`   | `str`  | Identifier of the originating document (file path or URL), matching `Document.source`. |
| `position` | `int`  | Character offset of the chunk's start within the source document's normalized text. |
| `metadata` | `dict` | Arbitrary metadata carried over from the source `Document` (e.g. `type`, `pages`, `headings`), plus any chunking-strategy-specific keys. |

## Relationship to `Document`

`rag_agent.ingestion.document.Document` (#8) is the input to
chunking — one `Document` per ingested source. A chunker (#12, #13)
splits a `Document` into a list of `Chunk`s, threading the source's
`metadata` through.
