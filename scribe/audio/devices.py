"""WASAPI device discovery: default mic and default-output loopback.

`audio.mic_device` / `audio.loopback_device` accept either an index or a name.
Indices are what PortAudio hands out, and they are not stable on Windows: they
renumber whenever a device or a piece of virtual-audio software appears or goes
away. A name survives that. Matching is case-insensitive against the names
`scribe devices` prints, and a unique substring is enough ("Brio").

A configured device that can't be resolved is an error, never a silent fallback
to some other device: recording the wrong microphone for an hour is worse than
refusing to start.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyaudiowpatch as pyaudio


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool


class DeviceSelectionError(Exception):
    """A configured mic/loopback device couldn't be resolved to exactly one device.

    The message is meant to be shown to the user as-is: it names the setting,
    says what went wrong, and lists the devices that would have been accepted.
    """


_ROLES = {
    # role: (config key, human label)
    "mic": ("audio.mic_device", "microphone"),
    "loopback": ("audio.loopback_device", "loopback device"),
}


def _to_info(raw: dict, is_loopback: bool) -> DeviceInfo:
    return DeviceInfo(
        index=raw["index"],
        name=raw["name"],
        channels=max(1, int(raw["maxInputChannels"])),
        sample_rate=int(raw["defaultSampleRate"]),
        is_loopback=is_loopback,
    )


def _wasapi_devices(p: pyaudio.PyAudio) -> list[DeviceInfo]:
    """Input-capable WASAPI devices, loopbacks flagged - what `scribe devices` lists."""
    try:
        wasapi_index = p.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    except OSError:
        wasapi_index = None
    devices: list[DeviceInfo] = []
    for i in range(p.get_device_count()):
        raw = p.get_device_info_by_index(i)
        if raw["maxInputChannels"] < 1:
            continue
        if wasapi_index is not None and raw["hostApi"] != wasapi_index:
            continue  # keep the list readable: WASAPI only
        devices.append(_to_info(raw, is_loopback=bool(raw.get("isLoopbackDevice"))))
    return devices


def _listing(devices: list[DeviceInfo]) -> str:
    if not devices:
        return "  (none found)"
    return "\n".join(f"  [{d.index}] {d.name}" for d in devices)


def match_device_name(name: str, candidates: list[DeviceInfo], role: str) -> DeviceInfo:
    """Pick the one candidate whose name matches `name`.

    An exact (case-insensitive) name wins outright, so a device called just
    "Microphone" stays selectable next to "Microphone (Brio 101)". Otherwise
    the name must be a substring of exactly one candidate.
    """
    key, label = _ROLES[role]
    wanted = name.strip().casefold()
    if not wanted:
        raise DeviceSelectionError(f"{key} is set to an empty name.")

    exact = [d for d in candidates if d.name.strip().casefold() == wanted]
    matches = exact or [d for d in candidates if wanted in d.name.casefold()]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceSelectionError(
            f'No {label} matching "{name}" ({key}). Available:\n'
            f"{_listing(candidates)}\n"
            f"Run `scribe devices` and set {key} in config.yaml to a name or index from that list."
        )
    raise DeviceSelectionError(
        f'"{name}" matches {len(matches)} {label}s ({key}):\n'
        f"{_listing(matches)}\n"
        f"Use more of the name so only one matches."
    )


def _normalize(spec: int | str) -> int | str:
    if isinstance(spec, bool):  # YAML `true`/`false` - an int subclass, never an index
        raise DeviceSelectionError(f"expected a device index or name, got {spec!r}")
    if isinstance(spec, int):
        return spec
    text = str(spec).strip()
    return int(text) if text.isdigit() else text


def resolve_device(p: pyaudio.PyAudio, spec: int | str, role: str) -> DeviceInfo:
    """Resolve a configured index or name to a device, or raise DeviceSelectionError."""
    key, label = _ROLES[role]
    is_loopback = role == "loopback"
    spec = _normalize(spec)

    if isinstance(spec, int):
        # Index: unchanged from before name matching existed, including that
        # it accepts any host API, not only the WASAPI devices `scribe devices`
        # lists - existing configs keep opening exactly what they opened.
        try:
            raw = p.get_device_info_by_index(spec)
        except (OSError, ValueError) as exc:
            candidates = [d for d in _wasapi_devices(p) if d.is_loopback == is_loopback]
            raise DeviceSelectionError(
                f"{key} is set to index {spec}, but there is no device {spec} ({exc}). "
                f"Indices change when devices are added or removed. Available {label}s:\n"
                f"{_listing(candidates)}\n"
                f"Set {key} to a name instead, e.g. part of one of the names above."
            ) from exc
        return _to_info(raw, is_loopback=is_loopback)

    candidates = [d for d in _wasapi_devices(p) if d.is_loopback == is_loopback]
    return match_device_name(spec, candidates, role)


def get_default_mic(p: pyaudio.PyAudio, override: int | str | None = None) -> DeviceInfo:
    if override is not None:
        return resolve_device(p, override, "mic")
    return _to_info(p.get_default_input_device_info(), is_loopback=False)


def get_default_loopback(
    p: pyaudio.PyAudio, override: int | str | None = None
) -> DeviceInfo | None:
    """Loopback device for the default output. Returns None if unavailable.

    None only ever means "no default loopback exists". A configured override
    that can't be resolved raises instead, so it can't quietly degrade to a
    mic-only recording.
    """
    if override is not None:
        return resolve_device(p, override, "loopback")
    try:
        return _to_info(p.get_default_wasapi_loopback(), is_loopback=True)
    except (OSError, LookupError):
        return None


def list_devices() -> list[DeviceInfo]:
    """All input-capable devices, loopbacks flagged."""
    with pyaudio.PyAudio() as p:
        return _wasapi_devices(p)
