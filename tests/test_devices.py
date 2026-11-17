"""Tests for device selection by index or name (scribe/audio/devices.py).

No real audio: a FakePyAudio stands in for pyaudiowpatch.PyAudio and serves a
fixed device table shaped like the November failure - a Brio next to an Oculus
virtual mic, and a Realtek loopback next to an Elgato one.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.audio import devices  # noqa: E402
from scribe.audio.devices import DeviceSelectionError  # noqa: E402

WASAPI = 2
MME = 0


def _dev(index, name, inputs, host=WASAPI, loopback=False, rate=48000):
    return {
        "index": index,
        "name": name,
        "maxInputChannels": inputs,
        "defaultSampleRate": float(rate),
        "hostApi": host,
        "isLoopbackDevice": loopback,
    }


DEVICE_TABLE = [
    _dev(0, "Microphone (Brio 101)", 2),
    _dev(1, "Headset Microphone (Oculus Virtual Audio Device)", 1),
    _dev(2, "Microphone (Realtek(R) Audio)", 2),
    _dev(3, "Speakers (Realtek(R) Audio) [Loopback]", 2, loopback=True),
    _dev(4, "Elgato Game Capture [Loopback]", 2, loopback=True),
    _dev(5, "Speakers (Realtek(R) Audio)", 0),             # output only: never listed
    _dev(6, "Microphone (Brio 101)", 2, host=MME, rate=44100),  # same name, other host API
]


class FakePyAudio:
    def __init__(self, table=None, default_input=0, default_loopback=3):
        self.table = list(DEVICE_TABLE if table is None else table)
        self.default_input = default_input
        self.default_loopback = default_loopback
        self.calls: list[str] = []
        self.terminated = False

    def get_host_api_info_by_type(self, _type):
        return {"index": WASAPI}

    def get_device_count(self):
        return len(self.table)

    def get_device_info_by_index(self, index):
        self.calls.append(f"by_index:{index}")
        for raw in self.table:
            if raw["index"] == index:
                return raw
        raise OSError("Invalid device index")  # what PortAudio raises

    def get_default_input_device_info(self):
        self.calls.append("default_input")
        return self.get_device_info_by_index(self.default_input)

    def get_default_wasapi_loopback(self):
        self.calls.append("default_loopback")
        if self.default_loopback is None:
            raise LookupError("no loopback")
        return self.get_device_info_by_index(self.default_loopback)

    def terminate(self):
        self.terminated = True


class TestNameMatching(unittest.TestCase):
    def test_exact_name(self):
        d = devices.get_default_mic(FakePyAudio(), "Microphone (Brio 101)")
        self.assertEqual((d.index, d.name), (0, "Microphone (Brio 101)"))

    def test_exact_name_wins_over_substring_hits(self):
        table = [_dev(7, "Microphone", 1)] + DEVICE_TABLE
        d = devices.get_default_mic(FakePyAudio(table), "microphone")
        self.assertEqual(d.index, 7)

    def test_substring_is_case_insensitive(self):
        d = devices.get_default_mic(FakePyAudio(), "brio")
        self.assertEqual(d.index, 0)

    def test_substring_ignores_surrounding_whitespace(self):
        d = devices.get_default_mic(FakePyAudio(), "  Brio  ")
        self.assertEqual(d.index, 0)

    def test_name_match_is_wasapi_only(self):
        # The MME copy of the Brio (index 6) shares the name; `scribe devices`
        # never shows it, so a name must not resolve to it.
        d = devices.get_default_mic(FakePyAudio(), "Microphone (Brio 101)")
        self.assertEqual(d.sample_rate, 48000)
        self.assertNotEqual(d.index, 6)

    def test_mic_name_never_matches_a_loopback(self):
        # "Realtek" is in one mic name and one loopback name.
        mic = devices.get_default_mic(FakePyAudio(), "realtek")
        self.assertEqual(mic.index, 2)
        self.assertFalse(mic.is_loopback)

    def test_loopback_name_never_matches_a_mic(self):
        lb = devices.get_default_loopback(FakePyAudio(), "realtek")
        self.assertEqual(lb.index, 3)
        self.assertTrue(lb.is_loopback)

    def test_loopback_by_name_does_not_consult_the_default(self):
        p = FakePyAudio(default_loopback=4)  # Elgato is the Windows default
        lb = devices.get_default_loopback(p, "Realtek")
        self.assertEqual(lb.index, 3)
        self.assertNotIn("default_loopback", p.calls)


class TestNoMatch(unittest.TestCase):
    def test_no_match_raises_and_lists_candidates(self):
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_mic(FakePyAudio(), "Yeti")
        msg = str(ctx.exception)
        self.assertIn('"Yeti"', msg)
        self.assertIn("audio.mic_device", msg)
        self.assertIn("[0] Microphone (Brio 101)", msg)
        self.assertIn("[1] Headset Microphone (Oculus Virtual Audio Device)", msg)
        self.assertNotIn("[Loopback]", msg, "mic candidates must not list loopbacks")
        self.assertIn("scribe devices", msg)

    def test_no_match_never_falls_back_to_the_default_mic(self):
        p = FakePyAudio()
        with self.assertRaises(DeviceSelectionError):
            devices.get_default_mic(p, "Yeti")
        self.assertNotIn("default_input", p.calls)

    def test_unmatched_loopback_raises_instead_of_returning_none(self):
        # None means "record mic only" to DualRecorder - a typo must not do that.
        p = FakePyAudio()
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_loopback(p, "Focusrite")
        self.assertIn("audio.loopback_device", str(ctx.exception))
        self.assertIn("[4] Elgato Game Capture [Loopback]", str(ctx.exception))
        self.assertNotIn("default_loopback", p.calls)

    def test_empty_name_is_an_error(self):
        with self.assertRaises(DeviceSelectionError):
            devices.get_default_mic(FakePyAudio(), "   ")

    def test_yaml_boolean_is_an_error_not_index_one(self):
        with self.assertRaises(DeviceSelectionError):
            devices.get_default_mic(FakePyAudio(), True)


class TestAmbiguous(unittest.TestCase):
    def test_ambiguous_raises_and_lists_only_the_hits(self):
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_mic(FakePyAudio(), "microphone")
        msg = str(ctx.exception)
        self.assertIn("matches 3 microphones", msg)
        for line in ("[0] Microphone (Brio 101)",
                     "[1] Headset Microphone (Oculus Virtual Audio Device)",
                     "[2] Microphone (Realtek(R) Audio)"):
            self.assertIn(line, msg)
        self.assertNotIn("Elgato", msg)

    def test_ambiguous_loopback(self):
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_loopback(FakePyAudio(), "[loopback]")
        self.assertIn("matches 2 loopback devices", str(ctx.exception))

    def test_two_devices_with_the_same_exact_name_are_ambiguous(self):
        table = DEVICE_TABLE + [_dev(7, "Microphone (Brio 101)", 2)]
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_mic(FakePyAudio(table), "Microphone (Brio 101)")
        self.assertIn("matches 2", str(ctx.exception))


class TestIndexBackwardCompatibility(unittest.TestCase):
    def test_integer_index_opens_that_device(self):
        p = FakePyAudio()
        d = devices.get_default_mic(p, 2)
        self.assertEqual(d.index, 2)
        self.assertIn("by_index:2", p.calls)

    def test_index_still_reaches_non_wasapi_devices(self):
        # Before name matching, an index could point at any host API. Keep it.
        d = devices.get_default_mic(FakePyAudio(), 6)
        self.assertEqual((d.index, d.sample_rate), (6, 44100))

    def test_digit_string_is_treated_as_an_index(self):
        d = devices.get_default_mic(FakePyAudio(), "2")
        self.assertEqual(d.index, 2)

    def test_loopback_index_is_flagged_as_loopback(self):
        d = devices.get_default_loopback(FakePyAudio(), 3)
        self.assertTrue(d.is_loopback)

    def test_missing_index_raises_a_readable_error(self):
        with self.assertRaises(DeviceSelectionError) as ctx:
            devices.get_default_mic(FakePyAudio(), 53)
        msg = str(ctx.exception)
        self.assertIn("index 53", msg)
        self.assertIn("[0] Microphone (Brio 101)", msg)

    def test_none_still_means_the_windows_default(self):
        p = FakePyAudio(default_input=2, default_loopback=4)
        self.assertEqual(devices.get_default_mic(p, None).index, 2)
        self.assertEqual(devices.get_default_loopback(p, None).index, 4)

    def test_no_default_loopback_still_returns_none(self):
        self.assertIsNone(devices.get_default_loopback(FakePyAudio(default_loopback=None)))


class TestListing(unittest.TestCase):
    def test_wasapi_devices_skips_output_only_and_other_host_apis(self):
        listed = [d.index for d in devices._wasapi_devices(FakePyAudio())]
        self.assertEqual(listed, [0, 1, 2, 3, 4])


class TestDualRecorderSelection(unittest.TestCase):
    """DualRecorder wiring, with PyAudio swapped for the fake."""

    def setUp(self):
        import scribe.audio.recorder as recorder_module

        self.recorder_module = recorder_module
        self._saved = recorder_module.pyaudio.PyAudio
        self.fake = FakePyAudio()
        recorder_module.pyaudio.PyAudio = lambda: self.fake
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name) / "session"

    def tearDown(self):
        self.recorder_module.pyaudio.PyAudio = self._saved
        self._tmp.cleanup()

    def test_names_pick_the_intended_tracks(self):
        rec = self.recorder_module.DualRecorder(
            self.session_dir, mic_device="Brio", loopback_device="Realtek"
        )
        names = {t.label: t.device.name for t in rec.tracks}
        self.assertEqual(names, {
            "mic": "Microphone (Brio 101)",
            "system": "Speakers (Realtek(R) Audio) [Loopback]",
        })
        self.assertIsNone(rec.warning)

    def test_unresolvable_device_raises_and_releases_portaudio(self):
        with self.assertRaises(DeviceSelectionError):
            self.recorder_module.DualRecorder(self.session_dir, mic_device="Yeti")
        self.assertTrue(self.fake.terminated)
        self.assertFalse(self.session_dir.exists(), "constructor must not create the folder")


if __name__ == "__main__":
    unittest.main()
