"""Behavioral regression checks. Synthetic audio and isolated stores only."""
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import dictation as d
import linux_backend as backend
from app_store import AppStore


class Conn:
    def __init__(self,replies):self.replies=iter(replies);self.audio=[]
    def send(self,msg):self.audio.append(msg[1])
    def recv(self):return next(self.replies)


class Reliability(unittest.TestCase):
    def setUp(self):
        self.saved=(dict(d.state),d.session,d.app_store,d.active_take_id,d.prompt_editor)
        self.tmp=tempfile.TemporaryDirectory(prefix="parakeet-regression-")
        d.app_store=AppStore(self.tmp.name)
        d.state.update(recording=False,busy=False,continuous=False)
        d.worker_ready_evt.set()

    def tearDown(self):
        state,d.session,d.app_store,d.active_take_id,d.prompt_editor=self.saved
        d.state.clear();d.state.update(state)
        self.tmp.cleanup()

    def test_short_tail_reuses_previous_context(self):
        rng=np.random.default_rng(2)
        audio=np.concatenate([np.zeros(6400),rng.normal(0,.1,316800)]).astype("float32")
        conn=Conn([("text","unfinished"),("text","no anchor"),("text","complete ending")])
        sess=d.AsrSession(conn,threading.Lock())
        sess.frames=[audio]
        sess._drain_frames();sess.stop_feeding();sess.close();sess.join_consumer(3)
        self.assertFalse(sess._consumer.is_alive())
        self.assertEqual(sess.ordered_texts(),["complete ending"])
        self.assertGreater(len(conn.audio[-1]),len(conn.audio[0]))
        self.assertEqual(len(conn.audio[-1]),len(audio))

    def test_context_replaces_wrong_ending_without_duplicate_words(self):
        self.assertEqual(d.join_tail_context("Earlier sentence. If anything that we generally", "If anything that we genuinely still need to finish."), "Earlier sentence. If anything that we genuinely still need to finish.")
        self.assertIsNone(d.join_tail_context("we can do it and we can do it", "we can do it tomorrow"))
        self.assertIsNone(d.join_tail_context("Nothing in common", "A completely different sentence"))

    def test_context_decode_is_bounded_to_nearby_audio(self):
        conn=Conn([("text","Earlier sentence. If anything that we generally"),("text","If anything that we genuinely still need to finish.")])
        sess=d.AsrSession(conn,threading.Lock())
        sess.submit(np.ones(20*d.SAMPLE_RATE,dtype="float32"))
        with patch.object(sess.segmenter,"finalize",return_value=np.ones(int(1.6*d.SAMPLE_RATE),dtype="float32")):sess.stop_feeding()
        sess.close();sess.join_consumer(2)
        self.assertEqual(len(conn.audio),2)
        self.assertEqual(len(conn.audio[1]),int(5.6*d.SAMPLE_RATE))
        self.assertEqual(sess.ordered_texts(),["Earlier sentence. If anything that we genuinely still need to finish."])

    def test_reported_one_point_six_second_tail_gets_context(self):
        conn=Conn([("text","words ending generally"),("text","no anchor"),("text","words ending genuinely and the remaining five words")])
        sess=d.AsrSession(conn,threading.Lock())
        previous=np.random.default_rng(4).normal(0,.03,320000).astype("float32")
        tail=np.random.default_rng(5).normal(0,.015,25600).astype("float32")
        sess.submit(previous)
        with patch.object(sess.segmenter,"finalize",return_value=tail):sess.stop_feeding()
        sess.close();sess.join_consumer(3)
        self.assertEqual(len(conn.audio[-1]),len(previous)+len(tail))
        self.assertEqual(sess.ordered_texts(),["words ending genuinely and the remaining five words"])

    def test_paste_waits_until_capsule_has_unmapped(self):
        import queue
        with patch.object(d,"ui_q",queue.Queue()),patch.object(d,"overlay",Mock()):
            sess=Mock(id="take",cancelled=False);d.active_take_id="take"
            completed=threading.Event()
            thread=threading.Thread(target=lambda:(d.prepare_paste(sess,"sample"),completed.set()))
            thread.start()
            deadline=time.monotonic()+1
            while d.ui_q.empty() and time.monotonic()<deadline:time.sleep(.001)
            ov=Mock();d.pump_ui(ov)
            ov.hide.assert_called_once()
            ov.copy_to_clipboard.assert_not_called()
            self.assertFalse(completed.is_set())
            delay,callback=ov.root.after.call_args.args
            self.assertGreaterEqual(delay,80)
            callback();thread.join(1)
            ov.copy_to_clipboard.assert_called_once_with("sample")
            self.assertTrue(completed.is_set())

    def test_cancelled_session_cannot_read_next_buffer(self):
        conn=Conn([]);sess=d.AsrSession(conn,threading.Lock())
        sess.frames=[np.ones(480,dtype="float32")]
        sess._cursor=1;sess.cancel()
        with patch.object(d,"frames",[np.zeros(480),np.ones(480)]),patch.object(sess.segmenter,"push") as push:
            sess._drain_frames();push.assert_not_called()
        sess.close();sess.join_consumer(1)

    def test_voiced_empty_result_is_an_explicit_error(self):
        conn=Conn([("text","good"),("text",""),("text","")])
        sess=d.AsrSession(conn,threading.Lock())
        wave=np.random.default_rng(1).normal(0,.1,16000).astype("float32")
        sess.submit(wave);sess.submit(wave);sess.close();sess.join_consumer(3)
        self.assertEqual(sess.ordered_texts(),["good"])
        self.assertIn(1,sess.errors)

    def test_send_is_bounded_too(self):
        release=threading.Event();conn=Mock()
        conn.send.side_effect=lambda _:release.wait(2)
        try:
            with patch.object(d,"ASR_TIMEOUT",.03),patch.object(d,"retire_worker") as retire:
                with self.assertRaises(TimeoutError):d.recognize_request(conn,threading.Lock(),np.zeros(10))
                retire.assert_called_once_with(conn)
        finally:release.set()

    def test_idle_monitor_rechecks_recent_activity(self):
        with patch.object(d,"_last_activity",time.time()),patch.object(d,"worker_proc") as proc:
            d.unload_idle_worker(999)
            proc.terminate.assert_not_called()
            proc.join.assert_not_called()

    def test_cancel_while_processing_disables_late_session(self):
        sess=Mock();d.session=sess;d.state.update(recording=False,busy=True,continuous=True)
        d.cancel()
        sess.cancel.assert_called_once();self.assertFalse(d.state["busy"])
        self.assertFalse(d.state["continuous"]);self.assertIsNone(d.session)

    def test_partial_take_is_saved_but_never_pasted(self):
        from prompt_editor import EditResult
        d.app_store.update("light_edit",True);d.prompt_editor=Mock()
        d.prompt_editor.edit.return_value=EditResult("First sentence", "edited")
        conn=Conn([("text","First sentence"),("asr_error","failed")])
        sess=d.AsrSession(conn,threading.Lock())
        sess.frames=[np.random.default_rng(1).normal(0,.1,16000).astype("float32")]
        sess._cursor=1
        sess.submit(sess.frames[0]);sess.submit(sess.frames[0])
        d.session=sess;d.state["busy"]=True
        with patch.object(backend,"copy_to_clipboard",return_value=True) as copy,patch.object(d,"paste_text") as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess)
            paste.assert_not_called()
            copy.assert_called_once_with("First sentence")
        self.assertEqual(d.app_store.recent()[0]["status"],"incomplete")
        self.assertEqual(d.app_store.recent()[0]["text"],"First sentence")
        # A later ASR failure may follow speculative work; it must never be used.
        if sess.live_edit:
            self.assertTrue(sess.live_edit._closed.is_set())

    def test_uncertain_delivery_never_sends_enter(self):
        d.app_store.update("voice_commands",True);d.state["continuous"]=True
        d.app_store.update("light_edit",True);d.prompt_editor=Mock()
        sess=d.AsrSession(Conn([("text","Example sentence send")]),threading.Lock())
        sess.frames=[np.random.default_rng(3).normal(0,.1,16000).astype("float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("uncertain","Ready")),patch.object(d,"press_command_key") as key,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess);key.assert_not_called()
        self.assertEqual(d.app_store.recent()[0]["status"],"uncertain")
        d.prompt_editor.edit.assert_not_called()

    def test_stale_inline_command_does_not_survive_tail(self):
        d.app_store.update("voice_commands",True);d.state["continuous"]=True
        sess=d.AsrSession(Conn([("text","Please send the example tomorrow")]),threading.Lock())
        sess._cmd_action="enter";sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("inserted","Inserted")),patch.object(d,"press_command_key") as key,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess);key.assert_not_called()

    def test_no_destructive_cleanup(self):
        self.assertEqual(d.clean_text("I had had enough"),"I had had enough")
        self.assertEqual(d.clean_text("Wir treffen uns um acht"),"Wir treffen uns um acht")

    def test_prompt_cleanup_preserves_literals_and_instructions(self):
        for text in (
            'Return exactly "uh,  um" and nothing else.',
            "Replace 'erm' with 'uh'.",
            "Keep `uh =  42` unchanged.",
            "Keep `uh =  42",
            "```python\nuh =  42\nprint(uh)\n```",
            "```python\nuh =  42\nprint(uh)",
            "Return “uh” or ‘erm’ exactly.",
            "Keep --uh and $uh and /uh and https://example.org/?uh=1.",
            "npm must not change src/uh.py or uh-oh or uh_value or UH.",
            "Do not delete anything. Keep 0.05, -12 and version 3.14.",
            "Make it blue, actually make it green. I really really mean it.",
            "First instruction.\n\nSecond instruction.",
        ):
            with self.subTest(text=text):self.assertEqual(d.clean_text(text),text)
        self.assertEqual(d.clean_text('Uh, keep "uh" and, erm, do not send.'),
                         'keep "uh" and, do not send.')
        self.assertEqual(d.clean_text("uh, uh, Please keep the final words."),
                         "Please keep the final words.")

    def test_cleanup_retains_raw_prompt_in_history(self):
        d.app_store.update("cleanup",True)
        original='Uh, do not change "uh" or the number 42.'
        sess=d.AsrSession(Conn([("text",original)]),threading.Lock())
        sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("dispatched","Ready")) as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess)
        self.assertEqual(paste.call_args.args[0],'do not change "uh" or the number 42.')
        self.assertEqual(d.app_store.recent()[0]["raw"],original)

    def test_light_edit_pastes_once_and_retains_raw(self):
        from prompt_editor import EditResult
        d.app_store.update("light_edit",True)
        original="Um make it blue, no actually red."
        sess=d.AsrSession(Conn([("text",original)]),threading.Lock())
        sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        def edit(text, cancelled):
            return EditResult("Make it red.","edited",.5)
        d.prompt_editor=Mock();d.prompt_editor.edit.side_effect=edit
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("dispatched","Ready")) as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess)
        paste.assert_called_once()
        self.assertEqual(paste.call_args.args[0],"Make it red.")
        saved=d.app_store.recent()[0]
        self.assertEqual(saved["raw"],original)
        self.assertEqual(saved["text"],"Make it red.")

    def test_asr_starts_editing_while_recording_and_reuses_it_on_stop(self):
        from prompt_editor import EditResult
        entered=threading.Event()
        original="Um I I want the tests first."
        def edit(text, cancelled):
            entered.set()
            return EditResult("I want the tests first.","edited",.01)
        d.app_store.update("light_edit",True)
        d.prompt_editor=Mock();d.prompt_editor.edit.side_effect=edit
        sess=d.AsrSession(Conn([("text",original)]),threading.Lock())
        sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;d.session=sess
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("dispatched","Ready")) as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            sess.submit(sess.frames[0])
            self.assertTrue(entered.wait(1))
            paste.assert_not_called()
            self.assertEqual(d.app_store.recent(),[])
            d._finish_take(sess)
        paste.assert_called_once()
        self.assertEqual(paste.call_args.args[0],"I want the tests first.")
        d.prompt_editor.edit.assert_called_once()
        self.assertEqual(d.app_store.recent()[0]["raw"],original)

    def test_editor_scheduling_failure_does_not_fail_recognition(self):
        sess=d.AsrSession(Conn([("text","Keep every word.")]),threading.Lock())
        with patch.object(sess,"queue_edit",side_effect=RuntimeError("Cannot start editor")):
            sess.submit(np.ones(16000,dtype="float32"))
            sess.close();sess.join_consumer(2)
        self.assertEqual(sess.ordered_texts(),["Keep every word."])
        self.assertFalse(sess.errors)
        self.assertFalse(sess._worker_failed)

    def test_cancel_during_light_edit_never_pastes(self):
        from prompt_editor import EditResult
        d.app_store.update("light_edit",True)
        sess=d.AsrSession(Conn([("text","Um keep this.")]),threading.Lock())
        sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        sess.close();sess.join_consumer(2)
        def edit(text, cancelled):
            sess.cancel()
            return EditResult("Keep this.","edited")
        d.prompt_editor=Mock();d.prompt_editor.edit.side_effect=edit
        with patch.object(d,"paste_text") as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess)
        paste.assert_not_called()
        self.assertEqual(d.app_store.recent()[0]["status"],"cancelled")

    def test_stale_editing_event_cannot_cover_new_recording(self):
        import queue
        d.active_take_id="new"
        with patch.object(d,"ui_q",queue.Queue()):
            d.ui_q.put(("editing","old"))
            overlay=Mock();d.pump_ui(overlay)
            overlay.show_pill.assert_not_called()

    def test_unexpected_editor_failure_still_pastes_original(self):
        d.app_store.update("light_edit",True)
        original="Please run the tests."
        sess=d.AsrSession(Conn([("text",original)]),threading.Lock())
        sess.frames=[np.ones(16000,dtype="float32")]
        sess._cursor=1;sess.submit(sess.frames[0]);d.session=sess
        d.prompt_editor=Mock();d.prompt_editor.edit.side_effect=RuntimeError("failed")
        with patch.object(d,"paste_text",return_value=backend.DeliveryResult("dispatched","Ready")) as paste,patch.object(d,"save_transcript"),patch.object(d,"apply_dictionary",side_effect=lambda s:s):
            d._finish_take(sess)
        self.assertEqual(paste.call_args.args[0],original)
        self.assertEqual(d.app_store.recent()[0]["text"],original)

    def test_fuzzy_phrase_cannot_consume_numbers_or_punctuation(self):
        _,entries,_,_=d._parse_dictionary("stable release")
        for s in ("stable 42 release","stable. Release","stable\nrelease"):
            self.assertEqual(d._apply_fuzzy(s,entries),(s,0))

    def test_immediate_voice_is_not_noise_floor(self):
        seg=d.Segmenter();seg.push(np.random.default_rng(4).normal(0,.08,64000).astype("float32"))
        self.assertGreater(seg._speech_samples,16000*3)

    def test_clipboard_failure_is_reported(self):
        with patch.object(backend,"copy_to_clipboard",return_value=False),patch.object(backend,"_chord") as chord:
            result=backend.paste_text("sample")
            self.assertEqual(result.status,"clipboard_failed");chord.assert_not_called()

    def test_held_modifiers_do_not_inject(self):
        with patch.object(backend,"copy_to_clipboard",return_value=True),patch.object(backend,"_wait_modifiers",return_value=False),patch.object(backend,"_chord") as chord,patch.object(backend.time,"sleep"):
            self.assertEqual(backend.paste_text("sample").status,"uncertain");chord.assert_not_called()

    def test_injection_is_not_proof_and_never_double_pastes(self):
        with patch.object(backend,"copy_to_clipboard",return_value=True),patch.object(backend,"_wait_modifiers",return_value=True),patch.object(backend,"_focused_snapshot",return_value=None),patch.object(backend,"_chord",return_value=True) as chord,patch.object(backend,"_atspi_insert") as fallback,patch.object(backend.time,"sleep"):
            result=backend.paste_text("sample","ctrl+shift+v",verify=True)
            self.assertEqual(result.status,"uncertain")
            chord.assert_called_once_with("ctrl","shift","v");fallback.assert_not_called()

    def test_normal_paste_never_waits_for_accessibility(self):
        with patch.object(backend,"copy_to_clipboard",return_value=True),patch.object(backend,"_wait_modifiers",return_value=True),patch.object(backend,"_focused_snapshot") as snapshot,patch.object(backend,"_chord",return_value=True) as chord,patch.object(backend.time,"sleep"):
            result=backend.paste_text("sample")
            self.assertEqual(result.status,"dispatched")
            self.assertFalse(result.confirmed)
            chord.assert_called_once_with("ctrl","v")
            snapshot.assert_not_called()

    def test_success_and_unconfirmed_paste_both_hide_capsule(self):
        import queue
        for outcome in ("inserted","recover"):
            with patch.object(d,"ui_q",queue.Queue()):
                d.active_take_id="take"
                d.ui_q.put(("outcome","take",outcome,"message"))
                overlay=Mock()
                d.pump_ui(overlay)
                overlay.hide.assert_called_once()
                overlay.show.assert_not_called()
                overlay.show_pill.assert_not_called()

    def test_background_feeder_transcribes_before_stop(self):
        conn=Conn([("text","Background words")])
        sess=d.AsrSession(conn,threading.Lock())
        speech=np.random.default_rng(9).normal(0,.12,64000).astype("float32")
        sess.frames=[np.concatenate([np.zeros(6400),speech,np.zeros(16000)]).astype("float32")]
        sess.start_feeder()
        try:
            deadline=time.monotonic()+2
            while not sess.results and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(sess.ordered_texts(),["Background words"])
            self.assertTrue(sess._feeding)
        finally:
            sess.stop_feeding();sess.close();sess.join_consumer(2)

    def test_native_clipboard_is_not_overwritten_by_wayland_helper(self):
        with patch.object(backend,"copy_to_clipboard") as copy,patch.object(backend,"_wait_modifiers",return_value=True),patch.object(backend,"_chord",return_value=True),patch.object(backend.time,"sleep"):
            self.assertEqual(backend.paste_text("sample",clipboard_ready=True).status,"dispatched")
            copy.assert_not_called()

    def test_exact_selection_replacement_can_be_confirmed(self):
        import gi
        gi.require_version("Atspi","2.0")
        from gi.repository import Atspi
        node=Mock()
        with patch.object(Atspi.Text,"get_text",return_value="new tail"):
            self.assertTrue(backend._insertion_matches((node,"old tail",0,3),"new"))
        with patch.object(Atspi.Text,"get_text",return_value="new old tail"):
            self.assertFalse(backend._insertion_matches((node,"old tail",0,3),"new"))

    def test_preferences_and_history_survive_reopen(self):
        d.app_store.update("cleanup",True)
        take=d.app_store.add("Final","Original",18,"uncertain")
        other=AppStore(self.tmp.name)
        self.assertTrue(other.settings["cleanup"])
        self.assertEqual(other.recent()[0]["raw"],"Original")
        other.mark(take,"copied")
        self.assertEqual(d.app_store.recent()[0]["status"],"copied")

    def test_legacy_import_is_idempotent(self):
        path=Path(self.tmp.name)/"old.log"
        content="[2026-09-07 10:00:00]\nA saved thought.\n\n"
        path.write_text(content)
        d.app_store.import_legacy(path);d.app_store.import_legacy(path)
        self.assertEqual(len(d.app_store.recent()),1)
        self.assertEqual(path.read_text(),content)

    def test_shortcut_installer_handles_no_conflict(self):
        import os
        import subprocess
        binary=Path(self.tmp.name)/"gsettings"
        binary.write_text("#!/bin/sh\ncase \"$1\" in\nget) echo '@as []';;\nlist-recursively) echo \"test key ['<Super>x']\";;\nset) echo WRITE;;\nesac\n")
        binary.chmod(0o755)
        env=dict(os.environ,PATH=self.tmp.name+os.pathsep+os.environ["PATH"])
        script=Path(__file__).resolve().parents[1]/"install-gnome-hotkeys.sh"
        result=subprocess.run(["bash",str(script)],env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count("WRITE"),13)

if __name__=="__main__":unittest.main()
