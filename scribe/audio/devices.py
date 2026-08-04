"""WASAPI device discovery: default mic and default-output loopback."""

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


def _to_info(raw: dict, is_loopback: bool) -> DeviceInfo:
    return DeviceInfo(
        index=raw["index"],
        name=raw["name"],
        channels=max(1, int(raw["maxInputChannels"])),
        sample_rate=int(raw["defaultSampleRate"]),
        is_loopback=is_loopback,
    )


def get_default_mic(p: pyaudio.PyAudio, override_index: int | None = None) -> DeviceInfo:
    if override_index is not None:
        return _to_info(p.get_device_info_by_index(override_index), is_loopback=False)
    return _to_info(p.get_default_input_device_info(), is_loopback=False)


def get_default_loopback(
    p: pyaudio.PyAudio, override_index: int | None = None
) -> DeviceInfo | None:
    """Loopback device for the default output. Returns None if unavailable."""
    if override_index is not None:
        return _to_info(p.get_device_info_by_index(override_index), is_loopback=True)
    try:
        return _to_info(p.get_default_wasapi_loopback(), is_loopback=True)
    except (OSError, LookupError):
        return None


def list_devices() -> list[DeviceInfo]:
    """All input-capable devices, loopbacks flagged."""
    devices: list[DeviceInfo] = []
    with pyaudio.PyAudio() as p:
        try:
            wasapi_index = p.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        except OSError:
            wasapi_index = None
        for i in range(p.get_device_count()):
            raw = p.get_device_info_by_index(i)
            if raw["maxInputChannels"] < 1:
                continue
            if wasapi_index is not None and raw["hostApi"] != wasapi_index:
                continue  # keep the list readable: WASAPI only
            devices.append(_to_info(raw, is_loopback=bool(raw.get("isLoopbackDevice"))))
    return devices
