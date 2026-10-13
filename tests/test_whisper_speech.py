"""Local-only loading and complete-transcript behavior for optional Whisper."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from whisper_speech import WhisperSpeech,whisper_assets


class WhisperTests(unittest.TestCase):
    def test_missing_or_mismatched_assets_fail_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):WhisperSpeech(tmp)
            root=Path(tmp);(root/'.local-speech').mkdir()
            (root/'.local-speech'/'whisper-turbo.json').write_text('{"model":"wrong"}')
            with self.assertRaises(RuntimeError):whisper_assets(tmp)

    def test_local_cpu_load_never_downloads(self):
        module=Mock()
        with patch('whisper_speech.whisper_assets',return_value=(Path('/test/assets'),None)),patch.dict(sys.modules,{'faster_whisper':module}):
            WhisperSpeech('/test')
        module.WhisperModel.assert_called_once_with('/test/assets',device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,local_files_only=True)

    def test_preserves_all_segment_words_without_translation_or_history(self):
        speech=WhisperSpeech.__new__(WhisperSpeech);speech.model=Mock()
        speech.model.transcribe.return_value=(iter([Mock(text=' Do not delete. '),Mock(text=' Keine Dateien löschen.')]),None)
        self.assertEqual(speech.recognize(np.ones(16000,dtype='float32')),'Do not delete. Keine Dateien löschen.')
        opts=speech.model.transcribe.call_args.kwargs
        self.assertEqual(opts['task'],'transcribe');self.assertIsNone(opts['language'])
        self.assertFalse(opts['condition_on_previous_text']);self.assertTrue(opts['vad_filter'])

    def test_mid_stream_failure_does_not_return_partial_text(self):
        def segments():
            yield Mock(text='Do not')
            raise RuntimeError('decode failed')
        speech=WhisperSpeech.__new__(WhisperSpeech);speech.model=Mock()
        speech.model.transcribe.return_value=(segments(),None)
        with self.assertRaisesRegex(RuntimeError,'decode failed'):speech.recognize(np.ones(16000))

    def test_invalid_audio_is_rejected(self):
        speech=WhisperSpeech.__new__(WhisperSpeech);speech.model=Mock()
        for data,rate in ((np.ones((2,2)),16000),(np.array([float('inf')]),16000),(np.ones(10),8000)):
            with self.assertRaises(ValueError):speech.recognize(data,rate)
        speech.model.transcribe.assert_not_called()


if __name__=='__main__':unittest.main()
