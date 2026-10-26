# SmartLab Chat

React + TypeScript chat workspace for the existing FastAPI application.

## Run Locally

Start the API at `http://127.0.0.1:8000` with its PostgreSQL and model services running. Then, from the repository root:

```sh
npm --prefix frontend ci
npm --prefix frontend run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`. Vite chooses another port if it is occupied.

To use a different API address, set `SMARTLAB_API_URL` when starting Vite:

```sh
SMARTLAB_API_URL=http://127.0.0.1:8001 npm --prefix frontend run dev
```

Requests to `/chat` and `/memories` are proxied on the same origin. The browser retains the API's HttpOnly session cookie; JavaScript never reads or stores that cookie. Use a consistent browser and hostname. Losing the cookie can make existing conversations inaccessible; start a new conversation to establish a new session.

## Checks

```sh
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run preview
```

Preview proxies API requests too, but is only a local build preview, not a production deployment. This addition does not change FastAPI, Docker, models, or database configuration. A production deployment needs a static file host and same-origin API reverse proxy.

## Behavior and Privacy

- Conversation titles, messages, source excerpts, and errors are stored in plaintext in this browser's local storage. Do not use a shared browser profile for sensitive conversations.
- The sidebar is browser-local, not a server history browser. The API has no conversation-list or history-read endpoint. Refresh restores completed local messages, but cannot recover a response that arrived while the page was closed.
- Rename, export, and removal operate locally. Removing a conversation does not delete its PostgreSQL checkpoints. Exported JSON includes its messages and sources.
- Saved memories use the server's session-scoped API. They are shared across conversations belonging to that session. Deleting a memory does not erase old transcripts or checkpoints.
- Only one chat request runs at a time. Enter sends; Shift+Enter inserts a newline. Requests time out after four minutes and are never automatically replayed, because the server may still be processing them.
- Sources are the excerpts supplied by the API. Markdown is rendered without raw HTML or remote images.
- "API connected" indicates API reachability, not a health check of every model or database connection.
- Streaming, uploads, account authentication, and cross-device history are not implemented because the backend does not expose those workflows.