"""ScribeLocal CLI.

  scribe devices              list WASAPI input/loopback devices
  scribe record [--title T]   record until Ctrl+C (or --duration N seconds)
  scribe list                 list past sessions
  scribe process <session>    transcribe + summarize an existing recording
  scribe serve                run the local web UI (+ inbox watcher)
  scribe watch                watch the inbox folder for dropped audio
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from .config import load_config
from .session import create_session_dir, list_sessions, resolve_session, write_meta


def _meter(level: float, width: int = 24) -> str:
    filled = int(level * width)
    return "#" * filled + "-" * (width - filled)


def cmd_devices(_args) -> int:
    from .audio.devices import list_devices

    print(f"{'idx':>4}  {'type':<8}  {'rate':>6}  {'ch':>2}  name")
    for d in list_devices():
        kind = "LOOPBACK" if d.is_loopback else "input"
        print(f"{d.index:>4}  {kind:<8}  {d.sample_rate:>6}  {d.channels:>2}  {d.name}")
    return 0


def cmd_record(args) -> int:
    from .audio.recorder import DualRecorder

    cfg = load_config()
    title = args.title or "meeting"
    session_dir = create_session_dir(cfg.notes_dir, title)
    recorder = DualRecorder(
        session_dir,
        mic_device=cfg.get("audio.mic_device"),
        loopback_device=cfg.get("audio.loopback_device"),
        chunk_frames=cfg.get("audio.chunk_frames", 4096),
    )
    if recorder.warning:
        print(f"WARNING: {recorder.warning}", file=sys.stderr)

    print(f"Session: {session_dir}")
    for track in recorder.tracks:
        print(f"  [{track.label}] {track.device.name} @ {track.device.sample_rate} Hz")
    print("Recording... Ctrl+C to stop.")

    recorder.start()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        while True:
            time.sleep(0.2)
            elapsed = recorder.elapsed()
            meters = "  ".join(
                f"{t.label} [{_meter(t.level())}]" for t in recorder.tracks
            )
            mins, secs = divmod(int(elapsed), 60)
            print(f"\r  {mins:02d}:{secs:02d}  {meters}", end="", flush=True)
            if args.duration and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    print("\nStopping...")
    stats = recorder.stop()

    write_meta(
        session_dir,
        title=title,
        started_at=started_iso,
        duration_seconds=max(t["seconds"] for t in stats.values()),
        template=args.template or cfg.get("summarize.template"),
        tracks=stats,
        status="recorded",
    )
    for label, t in stats.items():
        print(f"  [{label}] {t['seconds']}s -> {t['path']}")
    print(f"Done. Process it with:  scribe process {session_dir.name}")
    return 0


def cmd_process(args) -> int:
    """Transcribe and summarize an already-recorded session."""
    from .pipeline import process_session
    from .summarize import SummarizeError
    from .transcribe import TranscriptionError

    if args.transcribe_only and args.summarize_only:
        print("ERROR: --transcribe-only and --summarize-only are mutually exclusive",
              file=sys.stderr)
        return 2

    cfg = load_config()
    try:
        session_dir = resolve_session(cfg.notes_dir, args.session)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Session: {session_dir}")
    try:
        process_session(
            session_dir,
            cfg,
            transcribe=not args.summarize_only,
            summarize=not args.transcribe_only,
            template=args.template,
            model=args.model,
            force=args.force,
            notify=lambda m: print(f"  {m}"),
        )
    except (TranscriptionError, SummarizeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_watch(args) -> int:
    """Watch the inbox folder and import anything dropped into it."""
    from .inbox import InboxSettings, InboxWatcher

    cfg = load_config()
    settings = InboxSettings.from_config(cfg)
    if args.interval:
        settings.poll_seconds = args.interval
    if args.no_summarize:
        settings.summarize = False
    if args.template:
        settings.template = args.template

    watcher = InboxWatcher(notify=lambda m: print(m, flush=True), settings=settings)

    if args.once:
        imported = watcher.scan_once()
        if not imported:
            print(f"inbox: nothing to import in {cfg.inbox_dir}")
        return 0

    print(f"Watching {cfg.inbox_dir}  (Ctrl+C to stop)")
    try:
        watcher.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def cmd_serve(args) -> int:
    """Run the local web UI."""
    cfg = load_config()
    host = args.host or cfg.get("server.host", "127.0.0.1")
    port = args.port or cfg.get("server.port", 8321)
    try:
        from .server import serve
    except ImportError as exc:
        print(
            f"ERROR: the web UI needs FastAPI and uvicorn ({exc}). Install them with:\n"
            '  pip install -e ".[server]"',
            file=sys.stderr,
        )
        return 1

    print(f"ScribeLocal UI on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        serve(host=host, port=port)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_list(_args) -> int:
    cfg = load_config()
    sessions = list_sessions(cfg.notes_dir)
    if not sessions:
        print(f"No sessions in {cfg.notes_dir}")
        return 0
    for s in sessions:
        flags = "".join(
            ch if ok else "-"
            for ch, ok in (("R", s["recorded"]), ("T", s["transcribed"]), ("S", s["summarized"]))
        )
        print(f"[{flags}]  {s['name']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scribe", description="Local meeting notes")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list WASAPI audio devices").set_defaults(func=cmd_devices)

    rec = sub.add_parser("record", help="record a session (Ctrl+C to stop)")
    rec.add_argument("--title", "-t", help="session title")
    rec.add_argument("--duration", "-d", type=float, help="stop after N seconds")
    rec.add_argument("--template", help="summary template to use later")
    rec.set_defaults(func=cmd_record)

    sub.add_parser("list", help="list sessions").set_defaults(func=cmd_list)

    proc = sub.add_parser("process", help="transcribe + summarize an existing recording")
    proc.add_argument("session", help="session folder name, path, or unique prefix")
    proc.add_argument("--template", help="summary template (overrides config and meta.json)")
    proc.add_argument("--model", help="Claude model id (overrides summarize.model)")
    proc.add_argument("--force", "-f", action="store_true", help="redo existing outputs")
    proc.add_argument(
        "--transcribe-only", action="store_true", help="stop after writing transcript.md"
    )
    proc.add_argument(
        "--summarize-only", action="store_true", help="use the existing transcript.md"
    )
    proc.set_defaults(func=cmd_process)

    srv = sub.add_parser("serve", help="run the local web UI (and the inbox watcher)")
    srv.add_argument("--host", help="bind address (default: server.host in config.yaml)")
    srv.add_argument("--port", "-p", type=int, help="port (default: server.port)")
    srv.set_defaults(func=cmd_serve)

    watch = sub.add_parser("watch", help="watch the inbox folder for dropped audio")
    watch.add_argument("--once", action="store_true", help="scan once and exit")
    watch.add_argument("--interval", type=float, help="seconds between scans")
    watch.add_argument("--template", help="summary template for imports")
    watch.add_argument(
        "--no-summarize", action="store_true", help="transcribe imports but skip the API call"
    )
    watch.set_defaults(func=cmd_watch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
