"""Tests for scribe/live/context.py - reference-doc loading and budgeting.

No API and no credentials: build_context takes an injected `distiller`, so the
over-budget path is exercised with a plain function.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.live.context import (  # noqa: E402
    allocate_budget,
    build_context,
    read_documents,
    render,
)


class ContextTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.logged: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path


class TestReadDocuments(ContextTestCase):
    def test_reads_text_and_markdown(self):
        a = self.write("pricing.md", "# Pricing\nHoodies $28")
        b = self.write("notes.txt", "call notes")
        docs, warnings = read_documents([a, b])

        self.assertEqual([d.name for d in docs], ["pricing.md", "notes.txt"])
        self.assertEqual(docs[0].original_chars, len("# Pricing\nHoodies $28"))
        self.assertEqual(warnings, [])

    def test_missing_file_warns_but_does_not_raise(self):
        docs, warnings = read_documents([self.dir / "nope.md"])
        self.assertEqual(docs, [])
        self.assertIn("not found", warnings[0])

    def test_binary_types_are_refused_rather_than_guessed_at(self):
        pdf = self.dir / "sheet.pdf"
        pdf.write_bytes(b"%PDF-1.7 binary junk")
        docs, warnings = read_documents([pdf])

        self.assertEqual(docs, [])
        self.assertIn("unsupported context file type", warnings[0])

    def test_empty_and_directory_paths_are_skipped(self):
        empty = self.write("empty.md", "   \n")
        docs, warnings = read_documents([empty, self.dir])
        self.assertEqual(docs, [])
        self.assertEqual(len(warnings), 2)

    def test_one_bad_path_does_not_lose_the_good_ones(self):
        good = self.write("good.md", "keep me")
        docs, warnings = read_documents([self.dir / "gone.md", good])
        self.assertEqual([d.name for d in docs], ["good.md"])
        self.assertEqual(len(warnings), 1)


class TestBudgetAllocation(unittest.TestCase):
    def test_everything_fits_when_under_budget(self):
        self.assertEqual(allocate_budget([100, 200], 1000), [100, 200])

    def test_small_docs_release_their_unused_share(self):
        # The 10-char doc must not be truncated just because it was attached
        # next to a huge one; it takes 10 and leaves the rest to the big doc.
        allocation = allocate_budget([10, 10_000], 1000)
        self.assertEqual(allocation[0], 10)
        self.assertEqual(allocation[1], 990)

    def test_equal_oversized_docs_split_evenly(self):
        self.assertEqual(allocate_budget([5000, 5000], 1000), [500, 500])

    def test_no_docs_is_not_an_error(self):
        self.assertEqual(allocate_budget([], 1000), [])


class TestBuildContext(ContextTestCase):
    def build(self, paths, budget, **kw):
        return build_context(paths, budget, notify=self.logged.append, **kw)

    def test_no_paths_gives_an_empty_context(self):
        context = self.build([], 6000)
        self.assertTrue(context.is_empty())
        self.assertEqual(context.docs, [])

    def test_under_budget_docs_are_used_verbatim(self):
        path = self.write("pricing.md", "Hoodies $28, rush +15%")
        context = self.build([path], 6000)

        self.assertIn("Hoodies $28, rush +15%", context.text)
        self.assertIn("pricing.md", context.text)
        self.assertFalse(context.distilled)
        self.assertFalse(context.docs[0].truncated)

    def test_over_budget_without_a_distiller_truncates_and_warns(self):
        path = self.write("playbook.md", "para one.\n\n" + ("x" * 5000))
        context = self.build([path], 500)

        self.assertLessEqual(len(context.text), 700)  # block markup adds a little
        self.assertTrue(context.docs[0].truncated)
        self.assertTrue(any("truncated" in w for w in context.warnings))
        self.assertFalse(context.distilled)

    def test_over_budget_with_a_distiller_condenses_once(self):
        path = self.write("playbook.md", "y" * 9000)
        calls = []

        def distiller(text, target):
            calls.append((len(text), target))
            return "CONDENSED: hoodies $28, rush +15%"

        context = self.build([path], 400, distiller=distiller)

        self.assertEqual(len(calls), 1, "the distiller must run exactly once")
        self.assertTrue(context.distilled)
        self.assertIn("CONDENSED", context.text)

    def test_distiller_failure_falls_back_to_truncation(self):
        # No credentials must still leave a usable session, just a rougher one.
        path = self.write("playbook.md", "z" * 9000)

        def broken(text, target):
            raise RuntimeError("no API key")

        context = self.build([path], 400, distiller=broken)

        self.assertFalse(context.distilled)
        self.assertTrue(context.docs[0].truncated)
        self.assertTrue(any("no API key" in w for w in context.warnings))
        self.assertFalse(context.is_empty(), "fell back to nothing instead of truncating")

    def test_distiller_returning_nothing_falls_back_to_truncation(self):
        path = self.write("playbook.md", "z" * 9000)
        context = self.build([path], 400, distiller=lambda t, n: "   ")

        self.assertFalse(context.distilled)
        self.assertTrue(context.docs[0].truncated)

    def test_distiller_is_not_called_when_docs_already_fit(self):
        path = self.write("small.md", "tiny")
        called = []
        self.build([path], 6000, distiller=lambda t, n: called.append(1) or "x")
        self.assertEqual(called, [], "spent an API call on docs that already fit")

    def test_truncation_prefers_a_paragraph_boundary(self):
        path = self.write("doc.md", "first para.\n\n" + "second para. " * 200)
        context = self.build([path], 300)
        self.assertIn("truncated to fit", context.text)


class TestContextMeta(ContextTestCase):
    def test_meta_records_paths_but_never_contents(self):
        secret = "CONFIDENTIAL client pricing 12345"
        path = self.write("pricing.md", secret)
        context = build_context([path], 6000)

        meta = context.as_meta()
        self.assertNotIn(secret, str(meta), "document contents leaked into meta.json")
        self.assertEqual(meta["docs"][0]["name"], "pricing.md")
        self.assertEqual(meta["docs"][0]["chars"], len(secret))
        self.assertEqual(meta["docs"][0]["path"], str(path))


class TestRender(unittest.TestCase):
    def test_render_is_empty_for_no_docs(self):
        self.assertEqual(render([]), "")


if __name__ == "__main__":
    unittest.main()
