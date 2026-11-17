"""ScribeLocal CLI.

  scribe devices              list WASAPI input/loopback devices
  scribe record [--title T]   record until Ctrl+C (or --duration N seconds)
  scribe live [--title T]     record + live rolling transcript (Ctrl+C to stop)
  scribe list                 list past sessions
  scribe process <session>    transcribe + summarize an existing recording
  scribe serve                run the local web UI (+ inbox watcher)
  scribe watch                watch the inbox folder for dropped audio
"""

from __future__ import annotations

import argparse
import sys
import threading
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


def _remove_if_empty(session_dir) -> None:
    """Drop a session folder this command just created, if nothing was written to it.

    Only an empty folder is ever removed: create_session_dir() reuses an
    existing folder with the same minute and title, which may hold a recording.
    """
    try:
        if not any(session_dir.iterdir()):
            session_dir.rmdir()
    except OSError:
        pass


def cmd_record(args) -> int:
    from .audio.devices import DeviceSelectionError
    from .audio.recorder import DualRecorder

    cfg = load_config()
    title = args.title or "meeting"
    session_dir = create_session_dir(cfg.notes_dir, title)
    try:
        recorder = DualRecorder(
            session_dir,
            mic_device=cfg.get("audio.mic_device"),
            loopback_device=cfg.get("audio.loopback_device"),
            chunk_frames=cfg.get("audio.chunk_frames", 4096),
        )
    except DeviceSelectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        _remove_if_empty(session_dir)
        return 2
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
        mode="record",
    )
    for label, t in stats.items():
        print(f"  [{label}] {t['seconds']}s -> {t['path']}")
    print(f"Done. Process it with:  scribe process {session_dir.name}")
    return 0


def cmd_live(args) -> int:
    """Record + produce a live rolling transcript while the session runs.

    Reuses DualRecorder exactly as cmd_record does - audio_mic.wav and
    audio_system.wav are written identically, so `scribe process` already
    works on a live session's output with no special-casing. This command
    only adds a second, lower-fidelity transcript alongside that, for
    responsiveness while the call is happening; it does not run
    transcribe_session/summarize_session at all (that stays a `scribe
    process` step afterward, same as record mode).
    """
    from .audio.devices import DeviceSelectionError
    from .audio.recorder import DualRecorder
    from .live.context import build_context
    from .live.copilot import CopilotSettings, Suggestion, SuggestionEngine, make_distiller
    from .live.stream import LiveChunk, LiveSettings, LiveTranscriber
    from .transcribe import TranscriptionError

    cfg = load_config()
    title = args.title or "meeting"
    session_dir = create_session_dir(cfg.notes_dir, title)
    try:
        recorder = DualRecorder(
            session_dir,
            mic_device=cfg.get("audio.mic_device"),
            loopback_device=cfg.get("audio.loopback_device"),
            chunk_frames=cfg.get("audio.chunk_frames", 4096),
        )
    except DeviceSelectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        _remove_if_empty(session_dir)
        return 2
    if recorder.warning:
        print(f"WARNING: {recorder.warning}", file=sys.stderr)

    print(f"Session: {session_dir}")
    for track in recorder.tracks:
        print(f"  [{track.label}] {track.device.name} @ {track.device.sample_rate} Hz")

    live_settings = LiveSettings.from_config(cfg)
    print(
        f"Live transcript: model={live_settings.model}  "
        f"window={live_settings.window_seconds:.0f}s  "
        f"overlap={live_settings.overlap_seconds:.0f}s"
    )

    # Printing happens from multiple threads (the meter tick below, plus a
    # live-chunk/warning callback per track). A lock keeps individual print()
    # calls from interleaving mid-line; it does not - and doesn't need to -
    # impose any particular ORDER between them.
    #
    # Re-entrant on purpose: a streaming suggestion holds this for its whole
    # duration (on_start acquires, on_done releases), and the engine can emit
    # an error callback in between - which prints, from the same thread. A
    # plain Lock would deadlock there.
    print_lock = threading.RLock()

    def on_chunk(chunk: LiveChunk) -> None:
        with print_lock:
            tag = " [lagging]" if chunk.lagging else ""
            print(f"\n  [{chunk.speaker:<6}]{tag} {chunk.text}")

    def on_warning(track_label: str, message: str) -> None:
        with print_lock:
            print(f"\n  WARNING [{track_label}]: {message}", file=sys.stderr)

    # --- copilot: reference docs + suggestion engine -------------------------
    copilot_settings = CopilotSettings.from_config(cfg)
    if args.no_suggest:
        copilot_settings.enabled = False

    context = None
    if args.context:
        # The distiller only runs if the docs exceed the budget, and only once,
        # here at session start - never on the live suggestion path.
        distiller = None
        if copilot_settings.enabled:
            try:
                distiller = make_distiller(copilot_settings)
            except Exception as exc:  # no credentials: fall back to truncation
                print(f"  context: WARNING {exc}", file=sys.stderr)
        context = build_context(
            args.context,
            int(cfg.get("live.context_char_budget", 6000)),
            distiller=distiller,
            notify=lambda m: print(f"  {m}"),
        )

    engine = None
    if copilot_settings.enabled:
        print(
            f"Suggestions: model={copilot_settings.model}  "
            f"every {copilot_settings.interval_seconds:.0f}s of new speech  "
            f"max {copilot_settings.max_suggestion_chars} chars"
            + (f"  context={context.chars:,} chars" if context and not context.is_empty()
               else "  (no reference docs)")
        )
        print("  ^ each suggestion is a billed Anthropic API call")
    else:
        print("Suggestions: OFF (no API calls will be made)")

    # A suggestion streams token by token, so the whole thing holds print_lock
    # for its duration - otherwise the elapsed/meter line would overwrite it
    # mid-sentence. The meter simply pauses for the second or so it takes.
    suggestion_lock = threading.Lock()

    def on_suggest_start(reason: str) -> None:
        suggestion_lock.acquire()
        print_lock.acquire()
        print(f"\n  >> SUGGESTION [{reason}] ", end="", flush=True)

    def on_suggest_delta(piece: str) -> None:
        print(piece, end="", flush=True)

    def on_suggest_done(suggestion: Suggestion) -> None:
        try:
            if suggestion.error:
                # Already printed by on_suggest_error - just close the line so
                # the error isn't reported twice.
                print(flush=True)
            elif suggestion.cancelled:
                print("  [cancelled - session ended]", flush=True)
            elif suggestion.superseded:
                print("  [superseded by a newer trigger]", flush=True)
            elif not suggestion.text:
                print("(nothing useful right now)", flush=True)
            else:
                first = suggestion.first_token_ms
                total = suggestion.total_ms
                timing = ""
                if first is not None and total is not None:
                    timing = f"   [first word {first / 1000:.1f}s, complete {total / 1000:.1f}s]"
                print(timing, flush=True)
        finally:
            print_lock.release()
            suggestion_lock.release()

    def on_suggest_error(message: str) -> None:
        # The single owner of error output: per-call failures AND engine-level
        # notices (e.g. automatic suggestions standing down after repeated
        # failures) both arrive here. on_suggest_done deliberately doesn't
        # reprint them.
        with print_lock:
            print(f"\n  >> ERROR: {message}", flush=True)

    if copilot_settings.enabled:
        engine = SuggestionEngine(
            session_dir,
            copilot_settings,
            context,
            on_start=on_suggest_start,
            on_delta=on_suggest_delta,
            on_done=on_suggest_done,
            on_error=on_suggest_error,
        )

    def on_chunk_and_suggest(chunk: LiveChunk) -> None:
        on_chunk(chunk)
        if engine is not None:
            engine.note_chunk(chunk)

    live = LiveTranscriber(
        session_dir, live_settings, on_chunk=on_chunk_and_suggest, on_warning=on_warning
    )
    try:
        live.attach(recorder)  # loads the live model; must run before recorder.start()
    except TranscriptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        try:
            if not any(session_dir.iterdir()):
                session_dir.rmdir()
        except OSError:
            pass
        return 1

    print("Recording + live transcript... Ctrl+C to stop.")
    if engine is not None:
        engine.start()
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
            with print_lock:
                print(f"\r  {mins:02d}:{secs:02d}  {meters}", end="", flush=True)
            if args.duration and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    print("\nStopping...")
    stats = recorder.stop()  # joins the writer threads - every tap has now seen every buffer
    live_stats = live.stop()
    suggest_stats = engine.stop() if engine is not None else None

    write_meta(
        session_dir,
        title=title,
        started_at=started_iso,
        duration_seconds=max(t["seconds"] for t in stats.values()),
        template=args.template or cfg.get("summarize.template"),
        tracks=stats,
        status="recorded",
        mode="live",
        live_model=live_settings.model,
        live_stats=live_stats,
        context_docs=context.as_meta() if context else None,
        suggestions=suggest_stats,
    )
    for label, t in stats.items():
        print(f"  [{label}] {t['seconds']}s -> {t['path']}")
    for label, s in live_stats.items():
        print(f"  [{label}] live: {s['windows']} window(s) transcribed")
    print(f"Live transcript log: {session_dir / 'live_transcript.jsonl'}")
    if suggest_stats:
        median = suggest_stats.get("median_first_token_ms")
        print(
            f"Suggestions: {suggest_stats['delivered']} delivered, "
            f"{suggest_stats['calls']} API call(s), {suggest_stats['errors']} error(s), "
            f"{suggest_stats['superseded']} superseded"
        )
        if median is not None:
            print(f"  median time to first word: {median / 1000:.2f}s")
        print(f"  suggestions log: {session_dir / 'suggestions.jsonl'}")
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

    live = sub.add_parser(
        "live", help="record + live rolling transcript (Ctrl+C to stop)"
    )
    live.add_argument("--title", "-t", help="session title")
    live.add_argument("--duration", "-d", type=float, help="stop after N seconds")
    live.add_argument("--template", help="summary template to use later")
    live.add_argument(
        "--context",
        action="append",
        metavar="PATH",
        help="reference doc to ground suggestions in (repeatable): "
             "--context pricing.md --context playbook.md",
    )
    live.add_argument(
        "--no-suggest",
        action="store_true",
        help="record + live transcript only, no suggestion API calls",
    )
    live.set_defaults(func=cmd_live)

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
