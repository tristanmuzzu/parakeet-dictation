"""Contextual chunk matching: preserve words, repetitions and pipe ownership."""
import sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dictation as d

class Conn:
    def __init__(self,replies,on_recv=None):
        self.replies=iter(replies);self.sent=[];self.pending=False;self.on_recv=on_recv
    def send(self,msg):
        assert not self.pending
        self.pending=True;self.sent.append(msg)
    def recv(self):
        assert self.pending
        self.pending=False;r=next(self.replies)
        if self.on_recv:self.on_recv(len(self.sent))
        return r

class SentenceContextTests(unittest.TestCase):
    def setUp(self):
        self.saved=dict(d.state);d.state['continuous']=False;d.worker_ready_evt.set()
    def tearDown(self):d.state.clear();d.state.update(self.saved)
    def run_session(self,conn,count=2,context=4):
        s=d.AsrSession(conn,threading.Lock(),context_seconds=context)
        for _ in range(count):s.submit(np.zeros(10*d.SAMPLE_RATE,dtype='float32'))
        s.close();s.join_consumer(5)
        self.assertFalse(s._consumer.is_alive());self.assertFalse(conn.pending)
        return s
    def test_continuation_changes_only_boundary_and_keeps_previous_spelling(self):
        old='Keep PostgreSQL. I would like a shape that gets.'
        joined=d.join_chunk_context(old,'a shape that gets smaller and smaller.')
        self.assertEqual(joined,('Keep PostgreSQL. I would like a shape that gets','smaller and smaller.'))
    def test_complete_sentences_remain_separate(self):
        self.assertEqual(d.join_chunk_context('Do not disable encryption.','Do not disable encryption. Keep the preview off.'),('Do not disable encryption.','Keep the preview off.'))
    def test_changed_negation_number_or_missing_old_suffix_is_rejected(self):
        for old,new in [('Please do not disable encryption.','Please do disable encryption. Keep going.'),('Set timeout to 42 seconds.','Set timeout to 24 seconds. Keep going.'),('Please take a look at the image.','Please take a look at the preview and continue.')]:
            self.assertIsNone(d.join_chunk_context(old,new))
    def test_repeated_anchors_and_missing_continuations_are_rejected(self):
        for old,new in [('We can do it and we can do it.','We can do it tomorrow.'),('We go we.','We go we go we tomorrow.'),('Do not disable encryption.','Do not disable encryption.')]:
            self.assertIsNone(d.join_chunk_context(old,new))
    def test_quotes_are_not_stripped_or_moved(self):
        self.assertIsNone(d.join_chunk_context('Say "keep the final words."','Keep the final words and continue.'))
    def test_success_reuses_bounded_original_audio_and_preserves_order(self):
        conn=Conn([('text','Please take a look at.'),('text','take a look at the preview image.'),('text','the preview image. Keep the final words.')])
        s=self.run_session(conn,count=3)
        self.assertEqual(s.ordered_texts(),['Please take a look at','the preview image.','Keep the final words.'])
        self.assertEqual([len(m[1]) for m in conn.sent],[160000,224000,224000])
        self.assertFalse(s.errors)
    def test_mismatch_falls_back_without_changing_prior_words(self):
        conn=Conn([('text','Set timeout to 42 seconds.'),('text','Set timeout to 24 seconds. Keep going.'),('text','Keep going.')])
        s=self.run_session(conn)
        self.assertEqual(s.ordered_texts(),['Set timeout to 42 seconds.','Keep going.'])
        self.assertEqual([len(m[1]) for m in conn.sent],[160000,224000,160000])
    def test_new_repeated_instructions_survive_without_an_extra_decode(self):
        conn=Conn([('text','Keep the final words.'),('text','Keep the final words. Keep the final words again.')])
        s=self.run_session(conn)
        self.assertEqual(' '.join(s.ordered_texts()).count('Keep the final words'),2)
        self.assertEqual(len(conn.sent),2)
    def test_cancel_drains_context_reply_without_modifying_previous_result(self):
        holder={}
        conn=Conn([('text','Please take a look at.'),('text','take a look at the preview.')],lambda n:holder['s'].cancel() if n==2 else None)
        s=d.AsrSession(conn,threading.Lock(),context_seconds=4);holder['s']=s
        for _ in range(3):s.submit(np.zeros(16000,dtype='float32'))
        s.close();s.join_consumer(5)
        self.assertFalse(conn.pending);self.assertEqual(len(conn.sent),2)
        self.assertEqual(s.ordered_texts(),['Please take a look at.'])
    def test_context_worker_failure_is_reported_without_partial_update(self):
        conn=Conn([('text','Please take a look at.'),('asr_error','native failure')])
        s=self.run_session(conn,count=3)
        self.assertEqual(s.ordered_texts(),['Please take a look at.'])
        self.assertEqual(s.errors[1],'native failure');self.assertIn(2,s.errors)
    def test_default_is_scoped_to_cohere_normal_dictation(self):
        for model,continuous,expected in [('cohere-transcribe-en',False,64000),('parakeet',False,0),('cohere-transcribe-en',True,0)]:
            d.state['continuous']=continuous
            with patch.object(d,'IS_LINUX',True),patch.object(d,'app_store',SimpleNamespace(settings={'speech_model':model,'voice_commands':False})):
                s=d.AsrSession(Conn([]),threading.Lock());s.close();s.join_consumer(1)
                self.assertEqual(s.context_samples,expected)

if __name__=='__main__':unittest.main()
