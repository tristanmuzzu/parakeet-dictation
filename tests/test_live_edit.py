"""Deterministic streaming tests without a model, microphone or clipboard."""
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_editor import EditResult, LiveEdit


class LiveEditTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
            if session._thread:
                session._thread.join(1)
                self.assertFalse(session._thread.is_alive())

    def session(self, function):
        session = LiveEdit(Mock(edit=function))
        self.sessions.append(session)
        return session

    def wait_attempt(self, session, text):
        with session._condition:
            self.assertTrue(session._condition.wait_for(lambda: session._attempted == text, 1))

    def test_work_starts_before_stop_and_ready_result_returns_immediately(self):
        session = self.session(lambda text, cancelled: EditResult("Run the tests.", "edited"))
        session.update("Um run the tests.")
        self.wait_attempt(session, "Um run the tests.")
        result = session.finish("Um run the tests.")
        self.assertEqual(result.text, "Run the tests.")
        self.assertLess(result.seconds, .1)

    def test_updates_coalesce_without_cancel_starvation(self):
        entered, release = threading.Event(), threading.Event()
        seen = []
        def edit(text, cancelled):
            seen.append(text)
            if len(seen) == 1:
                entered.set(); release.wait(1)
            return EditResult(text.removeprefix("Um "), "edited")
        session = self.session(edit)
        session.update("Um start.")
        self.assertTrue(entered.wait(1))
        session.update("Um start. Obsolete.")
        session.update("Um start. Final.")
        release.set()
        self.wait_attempt(session, "Um start. Final.")
        self.assertEqual(session.finish("Um start. Final.").text, "start. Final.")
        self.assertEqual(seen, ["Um start.", "Um start. Final."])

    def test_deadline_preserves_completed_prefix_and_every_raw_tail_word(self):
        stopped = threading.Event()
        first = "Um run the tests."
        def edit(text, cancelled):
            if text == first:
                return EditResult("Run the tests.", "edited")
            while not cancelled():
                time.sleep(.001)
            stopped.set()
            return EditResult("DO NOT USE LATE OUTPUT", "edited")
        session = self.session(edit)
        session.update(first); self.wait_attempt(session, first)
        final = first + " Do not delete anything, even if the tests fail."
        result = session.finish(final, wait=.05)
        self.assertEqual(result.text, "Run the tests. Do not delete anything, even if the tests fail.")
        self.assertEqual(result.reason, "edited_prefix")
        self.assertLess(result.seconds, .2)
        self.assertTrue(stopped.wait(1))

    def test_changed_asr_ending_invalidates_stale_edit(self):
        first = "Um use the staging server."
        final = "Um use the production server."
        session = self.session(lambda text, cancelled: EditResult(text.removeprefix("Um "), "edited"))
        session.update(first); self.wait_attempt(session, first)
        result = session.finish(final, wait=0)
        self.assertEqual(result.text, final)
        self.assertEqual(result.reason, "deadline")

    def test_partial_sentence_or_open_quote_cannot_supply_prefix(self):
        for first, rest in (("Um I would like", " a larger font."),
                            ('Um quote "First sentence.', ' Last sentence."')):
            with self.subTest(first=first):
                session = self.session(lambda text, cancelled: EditResult(text.removeprefix("Um "), "edited"))
                session.update(first); self.wait_attempt(session, first)
                self.assertEqual(session.finish(first + rest, wait=0).text, first + rest)

    def test_failed_model_and_cancel_do_not_expose_generated_text(self):
        session = self.session(lambda text, cancelled: EditResult(text, "negation"))
        self.assertEqual(session.finish("Do not deploy.").reason, "negation")
        session = self.session(lambda text, cancelled: EditResult("Deploy.", "edited"))
        result = session.finish("Do not deploy.", cancelled=lambda: True)
        self.assertEqual((result.text, result.reason), ("Do not deploy.", "cancelled"))


if __name__ == "__main__":
    unittest.main()
