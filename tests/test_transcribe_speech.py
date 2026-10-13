"""Native ASR ownership, exact output and explicit failure contracts."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from transcribe_speech import TranscribeSpeech,PROFILES,RUNTIME,model_assets
from setup_transcribe_speech import reuse_tree


class NativeSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(setattr,sys,'path',sys.path.copy())
        self.env=patch.dict(os.environ);self.env.start();self.addCleanup(self.env.stop)
        files={}
        for role,name in [('model','model.gguf'),('library','lib.so'),('binding','python/transcribe_cpp/__init__.py')]:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test')
            files[role]=dict(path=str(path),size=4)
        (self.root/'.local-speech').mkdir()
        self.manifest=self.root/'.local-speech/cohere-transcribe-en.json'
        self.manifest.write_text(json.dumps(dict(PROFILES['cohere-transcribe-en'],runtime=RUNTIME,backend='vulkan',files=files)))
        self.native=Mock();self.native.native_version.return_value='0.2.4'
        self.model=self.native.Model.return_value
        self.model.backend='Vulkan0';self.model.variant='cohere-transcribe-03-2026'
        self.session=self.model.session.return_value
        self.module=patch.dict(sys.modules,{'transcribe_cpp':self.native});self.module.start();self.addCleanup(self.module.stop)

    def open(self):return TranscribeSpeech(self.root,'cohere-transcribe-en')

    def test_exact_transcription_no_rewriting_and_one_owned_session(self):
        text='Do not reset the database. Forty-two. No, twenty-four. Keep the final words.'
        self.session.run.return_value.text=text
        asr=self.open()
        self.assertEqual(asr.recognize(np.ones(1600,dtype='float32')),text)
        self.assertEqual(self.session.run.call_args.kwargs,dict(language='en',timestamps='none'))
        self.assertEqual(self.native.Model.call_args.kwargs,dict(backend='vulkan'))
        self.model.session.assert_called_once_with(n_threads=4)
        asr.close();asr.close()
        self.session.close.assert_called_once();self.model.close.assert_called_once()

    def test_missing_and_wrong_profile_fail_before_native_loading(self):
        (self.root/'model.gguf').unlink()
        with self.assertRaises(RuntimeError):self.open()
        self.native.Model.assert_not_called()
        with self.assertRaises(ValueError):model_assets(self.root,'unknown')

    def test_cpu_fallback_is_rejected_and_model_is_freed(self):
        self.model.backend='CPU'
        with self.assertRaisesRegex(RuntimeError,'acceleration'):self.open()
        self.model.session.assert_not_called();self.model.close.assert_called_once()

    def test_session_failure_frees_model(self):
        self.model.session.side_effect=RuntimeError('allocation failed')
        with self.assertRaisesRegex(RuntimeError,'allocation'):self.open()
        self.model.close.assert_called_once()

    def test_invalid_audio_and_closed_session_are_explicit(self):
        asr=self.open()
        for audio,rate in [(np.ones(16),48000),(np.ones((2,16)),16000),(np.array([np.nan]),16000)]:
            with self.assertRaises(ValueError):asr.recognize(audio,rate)
        self.assertEqual(asr.recognize(np.zeros(0)),'')
        self.session.run.assert_not_called()
        asr.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):asr.recognize(np.ones(16))

    def test_incomplete_native_output_never_becomes_success(self):
        self.session.run.side_effect=RuntimeError('output truncated')
        asr=self.open()
        with self.assertRaisesRegex(RuntimeError,'truncated'):asr.recognize(np.ones(1600))
        asr.close()

    def test_immutable_runtime_reuse_preserves_mapped_files(self):
        source=self.root/'source';target=self.root/'target'
        source.mkdir();target.mkdir()
        (source/'lib.so').write_bytes(b'library');(target/'lib.so').write_bytes(b'library')
        before=(target/'lib.so').stat();reuse_tree(source,target);after=(target/'lib.so').stat()
        self.assertEqual((before.st_ino,before.st_mtime_ns),(after.st_ino,after.st_mtime_ns))
        (source/'lib.so').write_bytes(b'other')
        with self.assertRaisesRegex(RuntimeError,'differs'):reuse_tree(source,target)
        self.assertEqual((target/'lib.so').read_bytes(),b'library')

if __name__=='__main__':unittest.main()
