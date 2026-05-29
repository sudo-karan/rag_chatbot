import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

MAX_HISTORY_TURNS = 20


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer_accepted: bool = False
    history: list[dict] = field(default_factory=list)


_sessions: dict[str, Session] = {}


def create_session() -> Session:
    sid = str(uuid.uuid4())
    session = Session(session_id=sid)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def accept_disclaimer(session_id: str) -> bool:
    session = _sessions.get(session_id)
    if session:
        session.disclaimer_accepted = True
        return True
    return False


def add_to_history(session_id: str, role: str, content: str):
    session = _sessions.get(session_id)
    if not session:
        return
    session.history.append({"role": role, "content": content})
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(session.history) > max_msgs:
        session.history = session.history[-max_msgs:]


def get_history(session_id: str) -> list[dict]:
    session = _sessions.get(session_id)
    return session.history if session else []


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    return list(_sessions.keys())
