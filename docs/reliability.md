# Persistence and Replay Safety

## Runtime

Set `DATABASE_URL` to the PostgreSQL connection string before starting Uvicorn.
The application lifespan opens a connection pool, runs the checkpointer and
ownership schema setup, builds the graph with `PostgresSaver`, and closes the
pool on shutdown. Without `DATABASE_URL`, `/generate` remains available but
`/chat` returns 503; it does not silently fall back to volatile checkpoints.
The graph builder's in-memory default is for isolated tests and standalone use.

Keep the PostgreSQL volume between restarts. `docker compose down -v` deletes
that volume and its data. Submit requests for a given conversation sequentially;
concurrent conversation turns are not serialized across application workers.

Clients must retain the `smartlab_session` cookie. Ownership hashes persist in
`conversation_owners` independently of process memory. Existing checkpoints
without ownership records are blocked, not automatically assigned to a new
client. This is anonymous session ownership, not account authentication.

Long-term memory uses the separate `user_memories` table. Deleting a checkpoint
does not delete these memories. `MemoryService.setup()` supports standalone use;
application startup also initializes the table. Saving the same user/key updates
the existing memory.

## Retrieval and Memory

The RAG route merges vector and PostgreSQL full-text matches by stable chunk ID
using reciprocal-rank fusion. It retrieves up to 20 candidates, reranks them
with the cross-encoder, and supplies the top three to the grounded answer model.
Citations retain the selected chunk IDs. Retrieval models load lazily on first
use; the database connection closes with the app lifespan.

Memory saving is explicit. First send a `/chat` request and retain its session
cookie, then use these endpoints with that same cookie:

- `POST /memories` with `{"key": "preferred_language", "value": "Rust"}` saves
	or updates a fact.
- `GET /memories` lists that session owner's saved facts.
- `DELETE /memories/preferred_language` deletes that saved fact.

Keys are at most 64 characters and start with an alphanumeric character or
underscore; subsequent characters may also include dots and hyphens. Values
must be nonblank and at most 1,000 characters. The API does not accept user IDs:
it derives the owner from a validated cookie and uses its hash for storage.

Each chat turn fetches the owner's 20 most recent memories. Direct and tool
reasoning receive at most 2,000 characters of whole JSON memory entries, marked
as untrusted user facts, not instructions. Direct prompts omit entries that
would exceed the gateway input limit. Memory is not document evidence, does not
expand tool permissions, and is not injected into the strictly grounded RAG
answer prompt. Recall works across conversations and restarts with the same
cookie; losing the anonymous session cookie loses access to its memories.

Deleting a memory stops automatic recall on subsequent turns. It does not erase
earlier messages or historical checkpoints, which may already contain that fact.
Prompt delivery is tested; DistilGPT2 remains a completion model and does not
guarantee accurate semantic recall or instruction following.

## Retries

Ollama connection and timeout failures get at most three attempts with 0.25s and
0.5s delays. Each transport uses a 60s timeout. Validation, permissions and model
HTTP errors are not retried. Entire graph runs and tool execution are not wrapped
in retries: replaying a graph can otherwise repeat completed side effects.

## Side-Effect Policy

Calculator and clock tools are explicitly read-only. Replaying them does not
mutate an external system, although clock results may differ. Dispatch rejects
side-effecting or unclassified tools before calling their implementation.

Before enabling a write tool, implement a durable operation key scoped to the
owner, conversation and logical tool call. Persist the canonical arguments hash
and outcome under a unique key; reject reuse with different arguments. Replays
must return the recorded result. For external effects, pass the same key to a
provider supporting idempotency, or use a transactional outbox. A local record
alone cannot guarantee exactly-once effects across a crash between an external
write and recording its result. Ambiguous operations must be reconciled, not
blindly retried. No write tool is currently enabled.

## Verification

With `DATABASE_URL` configured:

```sh
.venv/bin/python -m pytest tests/test_persistence.py tests/test_memory_service.py -q
```

The persistence tests start two independent Python processes, check ownership
before checkpoint reads, and verify that memory survives checkpoint deletion.