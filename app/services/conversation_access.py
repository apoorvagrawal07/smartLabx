import secrets
from hashlib import sha256
from threading import Lock


SESSION_COOKIE = "smartlab_session"


class ConversationAccess:
    def __init__(self):
        self._owners: dict[str, str] = {}
        self._sessions: set[str] = set()
        self._lock = Lock()

    def owner_id(self, session_id: str | None) -> str:
        with self._lock:
            if not session_id or session_id not in self._sessions:
                raise PermissionError("Forbidden")
            return sha256(session_id.encode()).hexdigest()

    def authorize(self, conversation_id: str, session_id: str | None) -> str:
        with self._lock:
            owner = self._owners.get(conversation_id)
            if owner is not None:
                if session_id not in self._sessions or not secrets.compare_digest(owner, session_id):
                    raise PermissionError("Forbidden")
                return owner

            if session_id not in self._sessions:
                session_id = secrets.token_urlsafe(32)
                self._sessions.add(session_id)
            self._owners[conversation_id] = session_id
            return session_id


class PostgresConversationAccess:
    def __init__(self, pool):
        self.pool = pool

    def owner_id(self, session_id: str | None) -> str:
        if not session_id:
            raise PermissionError("Forbidden")
        session_hash = sha256(session_id.encode()).hexdigest()
        with self.pool.connection() as connection:
            known_session = connection.execute(
                "SELECT 1 FROM conversation_owners WHERE session_hash = %s LIMIT 1",
                (session_hash,),
            ).fetchone()
        if known_session is None:
            raise PermissionError("Forbidden")
        return session_hash

    def authorize(self, conversation_id: str, session_id: str | None) -> str:
        session_hash = sha256(session_id.encode()).hexdigest() if session_id else None
        with self.pool.connection() as connection, connection.transaction():
            owner = connection.execute(
                "SELECT session_hash FROM conversation_owners WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
            if owner is not None:
                if not session_hash or not owner["session_hash"] or not secrets.compare_digest(
                    owner["session_hash"], session_hash
                ):
                    raise PermissionError("Forbidden")
                return session_id

            known_session = connection.execute(
                "SELECT 1 FROM conversation_owners WHERE session_hash = %s LIMIT 1",
                (session_hash,),
            ).fetchone()
            if known_session is None:
                session_id = secrets.token_urlsafe(32)
                session_hash = sha256(session_id.encode()).hexdigest()

            connection.execute(
                """
                INSERT INTO conversation_owners (conversation_id, session_hash)
                VALUES (%s, %s) ON CONFLICT (conversation_id) DO NOTHING
                """,
                (conversation_id, session_hash),
            )
            owner = connection.execute(
                "SELECT session_hash FROM conversation_owners WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
            if not owner["session_hash"] or not secrets.compare_digest(owner["session_hash"], session_hash):
                raise PermissionError("Forbidden")
            return session_id