"""Local web UI for ScribeLocal (FastAPI + a single static page)."""

from __future__ import annotations


def create_app():
    """Build the FastAPI app. Imported lazily so the CLI works without FastAPI."""
    from .app import create_app as _create_app

    return _create_app()


def serve(host: str = "127.0.0.1", port: int = 8321) -> None:
    """Run the UI with uvicorn (blocking)."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


__all__ = ["create_app", "serve"]
